#!/usr/bin/env python3
"""
scripts/compute_velocity_baseline.py

Tests whether EXPLICIT VELOCITY — estimated from two consecutive past frames
(z_prev, z0) — predicts the future state z1 BETTER than the current state z0
ALONE. This is the last cheap, linear check of whether ANY simple function of
available data carries dynamics-relevant information, before treating the
current representation (a single Chai-1 snapshot) as a dead end for this
task.

Why not just regress u = z1-z0 on v = z0-z_prev?
----------------------------------------------------
That's the obvious first instinct, but it repeats EXACTLY the confound found
in compute_restoring_force_baseline.py: u and v both contain "z0" (with
opposite signs), so under TRUE INDEPENDENCE (no real structure at all) this
regression has its own mechanical ceiling — R^2 = 25% at every lag, verified
algebraically before writing this script. Regressing one difference on
another difference is still confounded, just by a different constant.

The fix used here: a proper two-predictor OLS regression of z1 itself
(never differenced) on BOTH z0 and z_prev:

    z1  ~=  a + k0*z0 + k1*z_prev

compared against the single-predictor z0-only regression already computed
by compute_restoring_force_baseline.py's z1_predictability_mse. Neither
target nor predictors are ever differenced, so there is no shared-term
mechanical artifact at all — the ONLY way k1 (z_prev's coefficient) comes
out nonzero is if z_prev carries genuine information about z1 beyond what
z0 already provides. Solved via the standard 2-predictor normal equations,
per (position, channel), from one streaming pass of raw sufficient
statistics (n, sums, sums of squares/cross-products of z_prev, z0, z1).

Why this matters
------------------
Three independent diagnostics have already shown the SINGLE current frame
z0 carries ~0% linear information about the future state z1
(compute_restoring_force_baseline.py's z1_predictability_mse, 2026-07-12:
rho^2 = 0.00% across all 14 pilot proteins). But velocity is not observable
from a single snapshot — it requires at least two frames. Chai-1's
single-token representation was trained purely for STATIC structure
prediction and has never seen a temporal training signal; there's no a
priori reason a single frame would encode momentum. If z_prev also adds
nothing here, that's strong evidence the representation itself — not the
prediction lag, not the training loop — is the bottleneck.

Interpretation
---------------
- z0_and_velocity_mse close to z0_only_mse (little/no further reduction):
  z_prev carries no additional linear information beyond z0. Combined with
  everything else, this points at a genuinely different representation or
  approach being needed, not further tuning of the current one.
- z0_and_velocity_mse meaningfully below z0_only_mse: real momentum-based
  structure exists that richer conditioning (explicit velocity/history
  input to the model) could exploit — a concrete, actionable architecture
  change, not a dead end.

Usage:
    python scripts/compute_velocity_baseline.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS" \
        --lag 1 --velocity-lag 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.dataset import LatentDataset


def _collate_triplet(batch):
    """This computation never uses `pair` — skip the standard collate_fn's
    expensive pad+stack of it (see compute_position_mean_baseline.py)."""
    zprev_list, z0_list, z1_list = [], [], []
    for (z_prev, z0, z1), _pair, _fidx in batch:
        zprev_list.append(z_prev)
        z0_list.append(z0)
        z1_list.append(z1)
    return torch.stack(zprev_list), torch.stack(z0_list), torch.stack(z1_list)


@torch.no_grad()
def compute_velocity_baseline(
    protein_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict[str, float]:
    """
    Single-pass computation of:
      - z1_variance_mse:     baseline, predicting z1 by its own mean
      - z0_only_mse:         best single-predictor OLS fit of z1 on z0
      - z0_and_velocity_mse: best two-predictor OLS fit of z1 on (z0, z_prev)

    All per-(position, channel), via raw sufficient statistics — see module
    docstring for why z1 is regressed directly (never differenced).

    Excludes samples with an artificial self-pair at EITHER boundary: tail
    (z0 == z1, no valid future frame) or head (z_prev == z0, no valid past
    frame) — see compute_position_mean_baseline.py's docstring for why these
    dilute statistics if left in.

    Returns:
        {"z1_variance_mse": float, "z0_only_mse": float,
         "z0_and_velocity_mse": float, "n_samples": int,
         "n_excluded_self_pairs": int, "L": int}
    """
    loader = DataLoader(
        protein_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate_triplet, drop_last=False,
    )

    sum_zprev: torch.Tensor | None = None
    sum_zprev_sq: torch.Tensor | None = None
    sum_z0: torch.Tensor | None = None
    sum_z0_sq: torch.Tensor | None = None
    sum_z1: torch.Tensor | None = None
    sum_z1_sq: torch.Tensor | None = None
    sum_zprev_z0: torch.Tensor | None = None
    sum_zprev_z1: torch.Tensor | None = None
    sum_z0_z1: torch.Tensor | None = None
    n_samples = 0
    n_excluded_self_pairs = 0

    for z_prev, z0, z1 in loader:
        zp = z_prev.double()
        z0d = z0.double()
        z1d = z1.double()

        if sum_zprev is None:
            shape = (z0d.shape[1], z0d.shape[2])
            sum_zprev    = torch.zeros(shape, dtype=torch.float64)
            sum_zprev_sq = torch.zeros(shape, dtype=torch.float64)
            sum_z0       = torch.zeros(shape, dtype=torch.float64)
            sum_z0_sq    = torch.zeros(shape, dtype=torch.float64)
            sum_z1       = torch.zeros(shape, dtype=torch.float64)
            sum_z1_sq    = torch.zeros(shape, dtype=torch.float64)
            sum_zprev_z0 = torch.zeros(shape, dtype=torch.float64)
            sum_zprev_z1 = torch.zeros(shape, dtype=torch.float64)
            sum_z0_z1    = torch.zeros(shape, dtype=torch.float64)

        head_selfpair = (zp - z0d).abs().sum(dim=(1, 2)) == 0
        tail_selfpair = (z1d - z0d).abs().sum(dim=(1, 2)) == 0
        valid = ~(head_selfpair | tail_selfpair)
        if not valid.all():
            n_excluded_self_pairs += int((~valid).sum().item())
            zp, z0d, z1d = zp[valid], z0d[valid], z1d[valid]
        if zp.shape[0] == 0:
            continue

        sum_zprev    += zp.sum(dim=0)
        sum_zprev_sq += zp.pow(2).sum(dim=0)
        sum_z0       += z0d.sum(dim=0)
        sum_z0_sq    += z0d.pow(2).sum(dim=0)
        sum_z1       += z1d.sum(dim=0)
        sum_z1_sq    += z1d.pow(2).sum(dim=0)
        sum_zprev_z0 += (zp * z0d).sum(dim=0)
        sum_zprev_z1 += (zp * z1d).sum(dim=0)
        sum_z0_z1    += (z0d * z1d).sum(dim=0)
        n_samples += zp.shape[0]

    if n_samples == 0:
        raise RuntimeError("Dataset produced no non-self-paired samples.")

    L, d = sum_z0.shape
    n = n_samples

    # Centered sums of squares/products, per (l,c):
    ss_zprev    = sum_zprev_sq - sum_zprev.pow(2) / n
    ss_z0       = sum_z0_sq    - sum_z0.pow(2)    / n
    ss_z1       = sum_z1_sq    - sum_z1.pow(2)    / n
    sp_zprev_z0 = sum_zprev_z0 - (sum_zprev * sum_z0) / n
    sp_zprev_z1 = sum_zprev_z1 - (sum_zprev * sum_z1) / n
    sp_z0_z1    = sum_z0_z1    - (sum_z0    * sum_z1) / n

    z1_variance_mse = ss_z1.sum().item() / (n * L * d)

    # Single-predictor (z0 only) — same identity as compute_restoring_force_
    # baseline.py's z1_predictability_mse.
    safe_ss_z0 = ss_z0.clone()
    zero_var_z0 = safe_ss_z0 <= 0
    safe_ss_z0[zero_var_z0] = 1.0
    explained_z0_only = sp_z0_z1.pow(2) / safe_ss_z0
    explained_z0_only[zero_var_z0] = 0.0
    z0_only_resid = (ss_z1 - explained_z0_only).clamp(min=0.0)
    z0_only_mse = z0_only_resid.sum().item() / (n * L * d)

    # Two-predictor (z0 AND z_prev): solve the 2x2 normal equations per (l,c).
    #   [ss_z0     sp_zprev_z0] [k0]   [sp_z0_z1   ]
    #   [sp_zprev_z0  ss_zprev] [k1] = [sp_zprev_z1]
    det = ss_z0 * ss_zprev - sp_zprev_z0.pow(2)
    safe_det = det.clone()
    zero_det = safe_det.abs() <= 1e-12
    safe_det[zero_det] = 1.0
    k0 = (ss_zprev * sp_z0_z1 - sp_zprev_z0 * sp_zprev_z1) / safe_det
    k1 = (ss_z0 * sp_zprev_z1 - sp_zprev_z0 * sp_z0_z1) / safe_det
    k0[zero_det] = 0.0
    k1[zero_det] = 0.0

    two_predictor_explained = k0 * sp_z0_z1 + k1 * sp_zprev_z1
    two_predictor_resid = (ss_z1 - two_predictor_explained).clamp(min=0.0)
    z0_and_velocity_mse = two_predictor_resid.sum().item() / (n * L * d)

    return {
        "z1_variance_mse": z1_variance_mse,
        "z0_only_mse": z0_only_mse,
        "z0_and_velocity_mse": z0_and_velocity_mse,
        "n_samples": n_samples,
        "n_excluded_self_pairs": n_excluded_self_pairs,
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
                         help="Frame offset for the future target z1 (default 1 = 10ps)")
    parser.add_argument("--velocity-lag", type=int, default=1,
                         help="Frame offset for the past frame z_prev (default 1 = 10ps)")
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
        print(f"[velocity] Loaded normalization stats from {norm_stats_path}", flush=True)
    else:
        print(f"[velocity] No normalization stats found at {norm_stats_path} — "
              f"proceeding WITHOUT latent normalization.", flush=True)

    total_z1var_sq = 0.0
    total_z0only_sq = 0.0
    total_z0vel_sq = 0.0
    total_n = 0

    print(f"[velocity] lag={args.lag} frames ({args.lag * 10}ps), "
          f"velocity_lag={args.velocity_lag} frames ({args.velocity_lag * 10}ps)", flush=True)
    header = (f"{'Protein':<14} {'Z1Var':>12} {'Z0OnlyMSE':>12} {'Z0+VelMSE':>12} "
              f"{'%vsZ0Only':>11} {'N':>10}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for protein_id in protein_ids:
        protein_dataset = LatentDataset(
            args.latents_dir, protein_id, load_pair=False, norm_stats=norm_stats,
            lag=args.lag, velocity_lag=args.velocity_lag,
        )
        result = compute_velocity_baseline(
            protein_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        z1var  = result["z1_variance_mse"]
        z0only = result["z0_only_mse"]
        z0vel  = result["z0_and_velocity_mse"]
        n = result["n_samples"]
        n_excluded = result["n_excluded_self_pairs"]

        pct_vs_z0only = (1 - z0vel / z0only) * 100 if z0only > 0 else 0.0

        print(f"{protein_id:<14} {z1var:>12.6f} {z0only:>12.6f} {z0vel:>12.6f} "
              f"{pct_vs_z0only:>10.2f}% {n:>10}"
              + (f"  (excluded {n_excluded})" if n_excluded else ""),
              flush=True)

        total_z1var_sq  += z1var  * n * result["L"]
        total_z0only_sq += z0only * n * result["L"]
        total_z0vel_sq  += z0vel  * n * result["L"]
        total_n += n * result["L"]

    print("-" * len(header), flush=True)
    overall_z1var  = total_z1var_sq  / total_n
    overall_z0only = total_z0only_sq / total_n
    overall_z0vel  = total_z0vel_sq  / total_n
    overall_pct = (1 - overall_z0vel / overall_z0only) * 100 if overall_z0only > 0 else 0.0

    print(f"\n[summary] Aggregate z1 variance (baseline):            {overall_z1var:.6f}")
    print(f"[summary] Aggregate z0-only predictability MSE:        {overall_z0only:.6f}")
    print(f"[summary] Aggregate z0+velocity predictability MSE:    {overall_z0vel:.6f}")
    print(f"[summary] Incremental reduction from adding velocity:  {overall_pct:.2f}%")
    print("\n[summary] This isolates whether the explicit velocity z0-z_prev adds ANY")
    print("[summary] predictive value beyond the current state z0 alone (already ~0%,")
    print("[summary] per compute_restoring_force_baseline.py). If this is also ~0%, no")
    print("[summary] simple linear function of the available data predicts the future —")
    print("[summary] strong evidence the representation itself (a single frozen Chai-1")
    print("[summary] snapshot), not just the conditioning scheme, is the bottleneck.")


if __name__ == "__main__":
    main()
