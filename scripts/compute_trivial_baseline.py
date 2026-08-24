#!/usr/bin/env python3
"""
scripts/compute_trivial_baseline.py

Computes the MSE loss achieved by the trivial "predict no motion" model
(v_pred = 0 everywhere), measured directly from real training data, so it can
be compared against the model's observed training loss in loss_log.csv.

Why this matters
-----------------
FlowMatcher's regression target is u_t = z1 - z0 (constant regardless of t —
see training/flow_matching.py). A model that always outputs zero velocity
achieves loss = mean(u_t^2), with zero learned conditioning whatsoever. This
is the floor any model that's actually using its inputs (pair bias, current
latent, timestep) must beat by a meaningful margin. If the observed training
loss converges to (or hovers at/above) this baseline, the model has not
learned anything beyond "predict no motion" — which usually means training
hasn't run long enough yet, the LR/schedule needs tuning, or there's still a
data/gradient bug to chase down. If it's meaningfully below the baseline,
that's direct evidence real dynamics are being learned.

Usage:
    python scripts/compute_trivial_baseline.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins "$PILOT_PROTEINS" \
        --n-batches 200 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.dataset import build_multi_protein_dataset, collate_fn_no_pair


def compute_trivial_baseline_mse(
    dataloader: DataLoader,
    n_batches: int,
) -> tuple[float, int]:
    """
    Streams up to n_batches from dataloader and computes mean((z1 - z0)^2)
    over valid (non-padded) positions — the loss a zero-velocity predictor
    would achieve.

    Returns:
        (baseline_mse, n_samples_seen)
    """
    total_sq = 0.0
    total_n = 0
    n_samples = 0

    it = iter(dataloader)
    for i in range(n_batches):
        try:
            (z0, z1), _pair, _frame_idx, padding_mask = next(it)
        except StopIteration:
            it = iter(dataloader)
            (z0, z1), _pair, _frame_idx, padding_mask = next(it)

        u_t = z1 - z0  # [B, L, d]
        valid = (~padding_mask).float().unsqueeze(-1)  # [B, L, 1]
        sq = (u_t ** 2) * valid

        total_sq += sq.sum().item()
        total_n += valid.sum().item() * u_t.shape[-1]
        n_samples += z0.shape[0]

    if total_n == 0:
        raise RuntimeError("No valid (non-padded) tokens seen — check dataloader/padding mask.")

    return total_sq / total_n, n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--proteins", type=str, default=None,
                         help="Comma-separated protein IDs (default: all found)")
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="DataLoader worker processes (default 4; lower this on "
                              "memory-constrained nodes)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None
    dataset: Dataset = build_multi_protein_dataset(
        args.latents_dir, protein_ids=protein_ids, load_pair=False)

    torch.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_no_pair,
        drop_last=True,
    )

    print(f"[baseline] Sampling {args.n_batches} batches of {args.batch_size} "
          f"({args.n_batches * args.batch_size} pairs)...", flush=True)

    baseline, n_samples = compute_trivial_baseline_mse(dataloader, args.n_batches)

    print(f"\n[baseline] Trivial (v_pred=0) MSE over {n_samples} sampled pairs: {baseline:.6f}")
    print("[baseline] Compare this to the recent training loss in loss_log.csv —")
    print("[baseline] the model must sit meaningfully BELOW this number to prove")
    print("[baseline] it is using its conditioning (pair bias, z_t, t) at all.")


if __name__ == "__main__":
    main()
