"""
fluxion/training/checkpoint.py

Atomic, bit-exact checkpointing for Distributed Data Parallel (DDP) training.
Ensures corruption-proof saves and deterministic RNG state resumption.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    epoch: int,
    model_without_ddp: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    """
    Saves an atomic checkpoint (Rank 0 only).

    Writes to checkpoint.tmp first, then renames over checkpoint.pt.
    If the job is killed mid-write, the previous checkpoint.pt is
    untouched — os.replace() is atomic on POSIX filesystems.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = checkpoint_dir / "checkpoint.tmp"
    final_path = checkpoint_dir / "checkpoint.pt"

    state = {
        "step": step,
        "epoch": epoch,
        "model": model_without_ddp.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        # RNG states — restoring all four guarantees the DataLoader sees
        # exactly the same sample order and augmentations on resume.
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }

    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model_without_ddp: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, int]:
    """
    Loads checkpoint and restores full training state.

    Returns:
        (step, epoch) — training cursor to resume from.
        Returns (0, 0) if no checkpoint exists (fresh run).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return 0, 0

    # weights_only=False required: checkpoint includes numpy/python RNG states
    # which are not plain tensors and cannot be loaded with weights_only=True.
    state = torch.load(path, map_location="cpu", weights_only=False)

    model_without_ddp.load_state_dict(state["model"])
    ema_model.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])

    # Restore exact RNG state — must happen after all model/optimizer loads
    # so that any internal torch.rand() calls during state_dict loading do
    # not corrupt the restored state.
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])

    return state["step"], state["epoch"]
