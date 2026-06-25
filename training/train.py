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
import random
import signal
import sys
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from training.checkpoint import save_checkpoint, load_checkpoint
from data_pipeline.dataset import LatentDataset

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


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1. Distributed setup
    # ------------------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # 2. SLURM signal handler
    # ------------------------------------------------------------------
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # ------------------------------------------------------------------
    # 3. Model + EMA + optimizer
    # ------------------------------------------------------------------
    model = FluxionDiT().to(device)
    ema_model = deepcopy(model).requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        scheduler=None,
        device=device,
    )

    model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------
    # 5. Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = LatentDataset(args.scratch_dir, args.protein_id)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for (z0, z1), pair, _frame_idx in dataloader:

            # Preemption check — save on rank 0, then all ranks exit cleanly
            if PREEMPTION_REQUESTED:
                if global_rank == 0:
                    save_checkpoint(
                        checkpoint_dir, step, epoch,
                        model.module, ema_model, optimizer, scheduler=None,
                    )
                    print(f"Graceful exit at step {step}.", flush=True)
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)

            z0 = z0.to(device, non_blocking=True)
            z1 = z1.to(device, non_blocking=True)
            pair = pair.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = matcher.compute_loss(model, z0, z1, pair)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model.module)
            step += 1

            if global_rank == 0 and step % args.log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}",
                    flush=True,
                )

        # End-of-epoch checkpoint (rank 0 only)
        if global_rank == 0:
            save_checkpoint(
                checkpoint_dir, step, epoch + 1,
                model.module, ema_model, optimizer, scheduler=None,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=str, required=True)
    parser.add_argument("--protein-id", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    main(args)
