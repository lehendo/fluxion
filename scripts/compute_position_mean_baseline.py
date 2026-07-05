#!/usr/bin/env python3
"""
scripts/compute_position_mean_baseline.py

For each protein, computes how much of the frame-to-frame displacement
(u_t = z1 - z0) is explained by a purely STATIC per-(protein, residue
position) mean — no dynamics, no z_t, no timestep, just "this residue tends
to drift this direction on average" — and compares it to the global
zero-velocity baseline on the exact same (z0, z1) pairs the model trains on.

Why this matters
-----------------
FlowMatcher's Bayes-optimal predictor is v*(z_t, pair, t) = E[u_t | z_t, pair, t].
The static per-position mean is a SPECIAL CASE of what a well-trained model
with pair conditioning could in principle learn (pair is per-protein/
per-position static information) — so it's a clean, cheap upper-bound probe:

  - If the static mean barely beats the zero baseline, that's strong evidence
    frame-to-frame displacement is close to physically unpredictable from a
    single structural snapshot with no velocity/momentum history —
    consecutive MD frames may be near-independent thermal noise at this
    stride, and no further optimizer tuning will manufacture a meaningfully
    lower loss out of this (z_t, pair, t) -> u_t formulation. That argues for
    richer conditioning (e.g. a short history of frames) in the next phase,
    not more training-loop debugging.

  - If the static mean DOES beat zero substantially while the trained model
    does not, there IS learnable structure the current model+training still
    isn't capturing — worth continued optimization debugging.

Usage:
    python scripts/compute_position_mean_baseline.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS" \
        --lag         1

--lag sweeps the frame offset between z0/z1 (default 1 = native ATLAS
10ps/frame stride). See scripts/sweep_lag_baseline.sh to run this across many
lag values in parallel SLURM array tasks, and
scripts/summarize_lag_sweep.py to collect the results into one table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.dataset import LatentDataset


def _collate_no_pair(batch):
    """
    This computation never touches `pair`, but the standard collate_fn still
    pads and torch.stacks it into a [B, L, L, d_pair] tensor every batch —
    at the confirmed Chai-1 crop size (L=512, d_pair=256) that's several GB
    per batch for no benefit. Skip it entirely: stack only z0/z1 (safe here
    since every sample in a single-protein dataset shares the same L, so no
    padding is ever needed).
    """
    z0_list, z1_list = [], []
    for (z0, z1), _pair, _fidx in batch:
        z0_list.append(z0)
        z1_list.append(z1)
    return torch.stack(z0_list), torch.stack(z1_list)


@torch.no_grad()
def compute_position_mean_baseline(
    protein_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict[str, float]:
    """
    Single-pass, full-coverage computation over a single protein's dataset.

    Uses the standard sum-of-squares decomposition to get the position-mean
    MSE without a second pass over the data:

        sum((u_i - mean)^2) = sum(u_i^2) - n * mean^2
                             = sum(u_i^2) - (sum(u_i))^2 / n

    So accumulating sum(u) and sum(u^2) in one pass is enough to compute both
    the zero-baseline MSE (from sum(u^2) alone) and the position-mean MSE
    (from the identity above) — no need to revisit the data with the mean
    already known. Halves runtime on real cluster-scale latent stores (tens
    of thousands of frames, HDF5 decompression-bound) versus a naive
    compute-mean-then-compute-residual two-pass approach.

    Uses the exact same (z0, z1) pairs the model trains on (replica-safe
    pairing, normalization already applied if the dataset was built with
    norm_stats).

    Returns:
        {"zero_baseline_mse": float, "position_mean_mse": float,
         "n_samples": int, "L": int}
    """
    loader = DataLoader(
        protein_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate_no_pair, drop_last=False,
    )

    sum_u: torch.Tensor | None = None
    sum_u_sq = 0.0
    n_samples = 0
    for z0, z1 in loader:
        u = (z1 - z0).double()  # [B, L, d]
        if sum_u is None:
            sum_u = torch.zeros(u.shape[1], u.shape[2], dtype=torch.float64)
        sum_u += u.sum(dim=0)
        sum_u_sq += u.pow(2).sum().item()
        n_samples += u.shape[0]

    if n_samples == 0:
        raise RuntimeError("Dataset produced no samples.")

    L, d = sum_u.shape
    zero_baseline_mse = sum_u_sq / (n_samples * L * d)

    # sum((u_i - mean)^2) = sum(u_i^2) - (sum(u_i))^2 / n, applied per
    # position/channel and summed.
    resid_sq = sum_u_sq - sum_u.pow(2).sum().item() / n_samples
    position_mean_mse = resid_sq / (n_samples * L * d)

    return {
        "zero_baseline_mse": zero_baseline_mse,
        "position_mean_mse": position_mean_mse,
        "n_samples": n_samples,
        "L": L,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--proteins", type=str, default=None,
                         help="Comma-separated protein IDs (default: all found)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lag", type=int, default=1,
                         help="Frame offset between z0/z1 (default 1 = native "
                              "ATLAS 10ps/frame stride)")
    args = parser.parse_args()

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None

    if protein_ids is None:
        protein_ids = sorted(
            d.name for d in args.latents_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists() and (d / "pair.pt").exists()
        )

    norm_stats_path = args.latents_dir / "norm_stats.pt"
    norm_stats = None
    if norm_stats_path.exists():
        norm_stats = torch.load(norm_stats_path, weights_only=True)
        print(f"[posmean] Loaded normalization stats from {norm_stats_path}", flush=True)
    else:
        print(f"[posmean] No normalization stats found at {norm_stats_path} — "
              f"proceeding WITHOUT latent normalization.", flush=True)

    total_zero_sq = 0.0
    total_mean_sq = 0.0
    total_n = 0

    print(f"[posmean] lag={args.lag} frames ({args.lag * 10}ps at native ATLAS stride)",
          flush=True)
    header = f"{'Protein':<14} {'ZeroMSE':>12} {'PosMeanMSE':>12} {'%Reduction':>12} {'N':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for protein_id in protein_ids:
        # load_pair=False: this computation never touches pair, and skipping
        # it avoids loading/holding pair.pt (hundreds of MB per protein) for
        # nothing — see _collate_no_pair.
        protein_dataset = LatentDataset(
            args.latents_dir, protein_id, load_pair=False, norm_stats=norm_stats,
            lag=args.lag,
        )

        result = compute_position_mean_baseline(
            protein_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        zero_mse = result["zero_baseline_mse"]
        mean_mse = result["position_mean_mse"]
        pct = (1 - mean_mse / zero_mse) * 100 if zero_mse > 0 else 0.0
        n = result["n_samples"]

        print(f"{protein_id:<14} {zero_mse:>12.6f} {mean_mse:>12.6f} {pct:>11.2f}% {n:>10}",
              flush=True)

        total_zero_sq += zero_mse * n * result["L"]
        total_mean_sq += mean_mse * n * result["L"]
        total_n += n * result["L"]

    print("-" * len(header), flush=True)
    overall_zero = total_zero_sq / total_n
    overall_mean = total_mean_sq / total_n
    overall_pct = (1 - overall_mean / overall_zero) * 100 if overall_zero > 0 else 0.0
    print(f"\n[summary] Aggregate zero-baseline MSE:        {overall_zero:.6f}")
    print(f"[summary] Aggregate position-mean baseline MSE: {overall_mean:.6f}")
    print(f"[summary] Reduction from static position mean:  {overall_pct:.2f}%")
    print("\n[summary] If this reduction is small (a few percent), frame-to-frame")
    print("[summary] displacement is likely close to unpredictable from a single")
    print("[summary] structural snapshot — the pilot's flat loss curve is probably")
    print("[summary] a physical floor, not a remaining training bug.")


if __name__ == "__main__":
    main()
