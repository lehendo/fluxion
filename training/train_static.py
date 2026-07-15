"""
fluxion/training/train_static.py

DDP training loop for the static-ensemble pivot: OT-CFM from fresh Gaussian
noise to individual equilibrium-ensemble frames (no frame-to-frame pairing),
in frozen Chai-1 latent space. Reuses FluxionDiT, FlowMatcher (unchanged —
already generic over what z0 is), checkpointing, EMA, and the gate-LR-
multiplier optimizer setup from training/train.py; only the dataset and the
z0 source (freshly sampled noise instead of a real previous frame) differ.

Launch with torchrun:
    torchrun --nproc_per_node=NUM_GPUS training/train_static.py \
        --scratch-dir /scratch/arjunc4/fluxion_data \
        --proteins 1a62_A,1ab1_A

SLURM preemption: send SIGUSR1 to the process group and this loop will
checkpoint cleanly before the walltime kill arrives.
"""

import os
import sys
import signal
import argparse
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from training.checkpoint import save_checkpoint, load_checkpoint
from training.train import (
    get_param_groups,
    compute_group_grad_norms,
    update_ema,
    warmup_lambda,
)
from data_pipeline.dataset import build_multi_protein_static_dataset, collate_fn_static

PREEMPTION_REQUESTED = False


def handle_sigusr1(signum, frame) -> None:
    global PREEMPTION_REQUESTED
    PREEMPTION_REQUESTED = True
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"[Rank {rank}] SLURM preemption signal received. "
        "Preparing to checkpoint and exit...",
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Distributed setup
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # 2. SLURM signal handler
    # ------------------------------------------------------------------
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # ------------------------------------------------------------------
    # 3. Model + EMA + optimizer + warmup scheduler
    # ------------------------------------------------------------------
    model     = FluxionDiT().to(device)
    ema_model = deepcopy(model).requires_grad_(False)

    optimizer = torch.optim.AdamW(
        get_param_groups(model, args.weight_decay, args.lr, args.gate_lr_multiplier),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lambda(step, args.warmup_steps),
    )

    matcher = FlowMatcher()

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint (before wrapping in DDP)
    # ------------------------------------------------------------------
    checkpoint_dir = Path(args.scratch_dir) / "checkpoints_static"
    step, start_epoch = load_checkpoint(
        checkpoint_dir / "checkpoint.pt",
        model,
        ema_model,
        optimizer,
        scheduler,
        device,
    )

    model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------
    # 5. Dataset + DataLoader — individual frames, no pairing
    # ------------------------------------------------------------------
    latents_dir = Path(args.scratch_dir) / "latents"

    norm_stats_path = latents_dir / "norm_stats.pt"
    if not norm_stats_path.exists():
        proteins_flag = f' --proteins "{args.proteins}"' if args.proteins else ""
        raise FileNotFoundError(
            f"Normalization stats not found at {norm_stats_path}. Run:\n"
            f"  python data_pipeline/compute_norm_stats.py "
            f"--latents-dir {latents_dir}{proteins_flag}\n"
            f"before training."
        )

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None
    dataset = build_multi_protein_static_dataset(latents_dir, protein_ids=protein_ids)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_static,
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    model.train()

    loss_log_path = Path(args.scratch_dir) / "loss_log_static.csv"
    if global_rank == 0 and not loss_log_path.exists():
        with open(loss_log_path, "w") as f:
            f.write("step,epoch,loss,lr,gate_lr,total_grad_norm,gate_grad_norm\n")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for z1, pair, _frame_idx, padding_mask in dataloader:

            # Preemption check — save on rank 0, then all ranks exit cleanly
            if PREEMPTION_REQUESTED:
                if global_rank == 0:
                    save_checkpoint(
                        checkpoint_dir, step, epoch,
                        model.module, ema_model, optimizer, scheduler,
                    )
                    print(f"Graceful exit at step {step}.", flush=True)
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)

            z1           = z1.to(device,           non_blocking=True)
            pair         = pair.to(device,         non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            # Fresh Gaussian noise as z0 each step — the static-ensemble
            # objective is noise -> equilibrium sample, not frame -> frame.
            # FlowMatcher itself needs no changes: it is already agnostic
            # to what z0 represents.
            z0 = torch.randn_like(z1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)

            loss.backward()

            grad_norms = compute_group_grad_norms(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            update_ema(ema_model, model.module)
            step += 1

            if global_rank == 0 and step % args.log_interval == 0:
                lr_now      = scheduler.get_last_lr()[0]
                gate_lr_now = scheduler.get_last_lr()[2]
                loss_val = loss.item()
                print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f} | "
                    f"LR: {lr_now:.2e} | GateLR: {gate_lr_now:.2e} | "
                    f"GradNorm: {grad_norms['total_grad_norm']:.4f} | "
                    f"GateGradNorm: {grad_norms['gate_grad_norm']:.6f}",
                    flush=True,
                )
                with open(loss_log_path, "a") as f:
                    f.write(
                        f"{step},{epoch},{loss_val:.6f},{lr_now:.2e},{gate_lr_now:.2e},"
                        f"{grad_norms['total_grad_norm']:.6f},{grad_norms['gate_grad_norm']:.8f}\n"
                    )

            if (args.checkpoint_interval > 0
                    and step % args.checkpoint_interval == 0
                    and global_rank == 0):
                save_checkpoint(
                    checkpoint_dir, step, epoch,
                    model.module, ema_model, optimizer, scheduler,
                )

            if args.max_steps > 0 and step >= args.max_steps:
                if global_rank == 0:
                    save_checkpoint(
                        checkpoint_dir, step, epoch,
                        model.module, ema_model, optimizer, scheduler,
                    )
                    print(f"Reached --max-steps={args.max_steps}. Exiting at step {step}.",
                          flush=True)
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)

        # End-of-epoch checkpoint (rank 0 only)
        if global_rank == 0:
            save_checkpoint(
                checkpoint_dir, step, epoch + 1,
                model.module, ema_model, optimizer, scheduler,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir",    type=str,   required=True)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--gate-lr-multiplier", type=float, default=10.0,
                        help="LR multiplier for AdaLN-zero gate/final-projection "
                             "params — see training/train.py's get_param_groups "
                             "docstring for why.")
    parser.add_argument("--weight-decay",   type=float, default=1e-5)
    parser.add_argument("--warmup-steps",   type=int,   default=2000)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--log-interval",   type=int,   default=50)
    parser.add_argument("--proteins",             type=str,   default=None,
                        help="Comma-separated protein IDs to include (default: all)")
    parser.add_argument("--checkpoint-interval", type=int,   default=0,
                        help="Save checkpoint every N steps (0 = off, use for scavenger)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Checkpoint and exit after this many total steps "
                             "(0 = unlimited, run full --epochs). Use for short "
                             "diagnostic runs.")
    args = parser.parse_args()
    main(args)
