"""
fluxion/training/train.py

DDP Training Loop with OT-CFM, EMA, and SLURM Signal Handling.

Launch with torchrun:
    torchrun --nproc_per_node=NUM_GPUS training/train.py \
        --scratch-dir /scratch/arjunc4/fluxion_data \
        --protein-id 1AKE

SLURM preemption: send SIGUSR1 to the process group and this loop will
checkpoint cleanly before the walltime kill arrives.
"""

import os
import sys
import random
import signal
import argparse
from copy import deepcopy
from pathlib import Path

# Ensure repo root is on sys.path regardless of invocation directory,
# so that `from training.*` and `from data_pipeline.*` resolve correctly
# when torchrun sets the working directory to something unexpected.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from training.checkpoint import save_checkpoint, load_checkpoint
from data_pipeline.dataset import build_multi_protein_dataset, collate_fn

# ---------------------------------------------------------------------------
# Global preemption flag — set by SIGUSR1 handler, checked each batch
# ---------------------------------------------------------------------------
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


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """In-place EMA update: ema ← decay·ema + (1-decay)·model."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    gate_lr_multiplier: float = 1.0,
) -> list[dict]:
    """
    Three groups:
      - "decay":    regular >=2D weights (attention/FFN/projections), with
                    weight decay, at the base learning rate.
      - "no_decay": remaining 1-D params (biases), no weight decay, at the
                    base learning rate.
      - "gate":     every zero-initialized "wake-up" parameter — the AdaLN-
                    zero modulation weight+bias in every block and in
                    final_layer, plus final_layer's own output-projection
                    weight+bias — no weight decay, at
                    gate_lr_multiplier * base learning rate.

    Why the gate group needs its own (higher) LR
    -----------------------------------------------
    Every block computes x = x + gate * f(x), where f is attention or FFN.
    The gradient reaching f's OWN internal parameters is scaled by gate's
    CURRENT value (chain rule through the multiplication) — while gate ~ 0,
    those parameters get essentially no usable signal. final_layer.linear
    plays the same throttling role for the ENTIRE stack: its still-small
    weight multiplies the gradient flowing backward into every earlier
    block, including their gates. But the gate/final-projection parameters
    themselves receive gradient through a DIFFERENT, undiminished path (as
    an outer multiplier on f(x)/norm(x), not gated by their own value) — so
    they can, and empirically do, learn on their own regardless of how
    starved the rest of the network is. Giving them a higher LR accelerates
    them past this cold-start bottleneck without over-accelerating the rest
    of the network's still mostly noise-dominated gradient.
    """
    decay, no_decay, gate = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adaLN_mod" in name or "final_layer.linear" in name:
            gate.append(param)
        elif param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0,           "lr": lr},
        {"params": gate,     "weight_decay": 0.0,           "lr": lr * gate_lr_multiplier},
    ]


def compute_group_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Must be called after loss.backward() and before optimizer.zero_grad().

    Splits the L2 gradient norm into two groups:
      "total_grad_norm": every parameter's gradient — this is the same
          quantity torch.nn.utils.clip_grad_norm_ measures pre-clip.
      "gate_grad_norm": only the AdaLN-zero gate/final-projection parameters
          (those that start at exactly zero — see get_param_groups for why).

    If gate_grad_norm is a substantial fraction of (or roughly equal to)
    total_grad_norm, that means the OPPOSITE of what "starved gates" would
    look like: virtually all gradient signal is concentrated in the gates,
    and the rest of the network (attention/FFN internals) is what's
    starved — because their gradient is chain-rule-multiplied by the still-
    tiny gate value that sits between them and the loss. This was confirmed
    empirically on this model (2026-07): gate_grad_norm tracked
    total_grad_norm almost exactly at every logged step.
    """
    total_sq = 0.0
    gate_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = param.grad.detach().double().pow(2).sum().item()
        total_sq += sq
        if "adaLN_mod" in name or "final_layer.linear" in name:
            gate_sq += sq
    return {
        "total_grad_norm": total_sq ** 0.5,
        "gate_grad_norm": gate_sq ** 0.5,
    }


def warmup_lambda(step: int, warmup_steps: int) -> float:
    """Linear warmup, then constant LR."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return 1.0


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
    checkpoint_dir = Path(args.scratch_dir) / "checkpoints"
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
    # 5. Dataset + DataLoader
    # ------------------------------------------------------------------
    latents_dir = Path(args.scratch_dir) / "latents"

    # Normalization stats are required, not optional. FlowMatcher's MSE loss
    # regresses directly against these latents; left unnormalized, a handful
    # of high-variance channels dominate the loss and starve gradient signal
    # for everything else — see data_pipeline/compute_norm_stats.py docstring.
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
    dataset = build_multi_protein_dataset(latents_dir, protein_ids=protein_ids)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    model.train()

    loss_log_path = Path(args.scratch_dir) / "loss_log.csv"
    if global_rank == 0 and not loss_log_path.exists():
        with open(loss_log_path, "w") as f:
            f.write("step,epoch,loss,lr,gate_lr,total_grad_norm,gate_grad_norm\n")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for (z0, z1), pair, _frame_idx, padding_mask in dataloader:

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

            z0           = z0.to(device,           non_blocking=True)
            z1           = z1.to(device,           non_blocking=True)
            pair         = pair.to(device,         non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)

            loss.backward()

            # Measured pre-clip so gate_grad_norm reflects the natural,
            # unclipped gradient reaching the AdaLN-zero parameters — the
            # signal we actually care about diagnosing, not what's left of
            # it after the global clip has already rescaled everything.
            grad_norms = compute_group_grad_norms(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            update_ema(ema_model, model.module)
            step += 1

            if global_rank == 0 and step % args.log_interval == 0:
                # Param groups are [decay, no_decay, gate] — see get_param_groups.
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

            # Diagnostic/short-run cap — checkpoint and exit cleanly once the
            # requested step count is hit, same graceful-exit pattern as the
            # SIGUSR1 preemption handler. Lets a quick instrumentation run
            # (e.g. a few thousand steps) be validated cheaply before
            # committing to a full multi-day run.
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
                             "params, which sit behind a self-throttling multiplicative "
                             "bottleneck and need to grow faster than the rest of the "
                             "network to unblock it (default 10x base LR)")
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
