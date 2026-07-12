#!/usr/bin/env python3
"""
scripts/compute_restoring_force_baseline.py

Tests for Ornstein-Uhlenbeck-style mean-reversion: does a residue's CURRENT
deviation from its own time-averaged position predict its NEXT displacement
(a restoring force pulling it back toward equilibrium), beyond what the
static per-position mean (scripts/compute_position_mean_baseline.py) already
explains?

Why this is a different, more relevant test
---------------------------------------------
The position-mean baseline found 0% explainable structure at every lag from
10ps to 50ns (2026-07-07 sweep) — but that only rules out a FIXED,
time-independent bias per position. It says nothing about state-dependent
structure: whether the current frame z_t itself carries information about
where the system goes next. The real FlowMatcher model conditions on z_t
directly (v_theta(z_t, pair, t)), so this is a much closer proxy for what the
model could actually exploit than the position-mean test was.

Method: for each (position, channel), fit the single best linear predictor
    u_t = z1 - z0  ~=  k * (z0 - mu) + b
via ordinary least squares (mu = time-averaged z0, k = restoring-force
coefficient — k < 0 means "pulls back toward equilibrium", the OU signature),
computed in one streaming pass via raw sufficient statistics. OLS with an
intercept is invariant to constant shifts of the predictor, so regressing
directly on z0 (instead of z0 - mu) gives the identical k* and residual
without needing mu computed in advance:

    Cov(u, z0 - mu) = Cov(u, z0)          (mu is a constant, shift-invariant)
    Var(z0 - mu)    = Var(z0)             (same)
    k*  = Cov(u, z0) / Var(z0)
    SSE = SS_u - Cov(u,z0)^2 / Var(z0)    (standard OLS residual identity)

Reports the residual MSE after this fit against both the zero baseline and
the position-mean baseline, so the INCREMENTAL value of state-dependent
conditioning is visible on top of what a static bias already explains
(nothing, per the lag sweep). Excludes artificial self-pairs the same way
compute_position_mean_baseline.py does — see that module's docstring.

Interpretation
---------------
- If restoring-force MSE is close to position-mean MSE (no further
  reduction): z0 itself carries no LINEAR information about u_t either.
  Combined with the flat lag-sweep result, this would be strong evidence
  that Fluxion's OT-CFM formulation has no first-order-linear signal to
  exploit from (z_t, pair, t) at all in this latent space — arguing for a
  genuinely different conditioning (explicit velocity/history) rather than
  continued tuning of the current single-snapshot setup.
- If restoring-force MSE drops meaningfully below position-mean MSE: real,
  state-dependent (if linear) structure exists that a model conditioning on
  z_t could in principle learn — reopens the "this is an optimization/
  architecture problem, not a physical floor" hypothesis with much stronger
  justification than before.

A subtlety that matters for reading the real result correctly
-----------------------------------------------------------------
u = z1 - z0 mechanically CONTAINS "-z0" as a term, so regressing u on z0
picks up a baseline negative correlation even under a NULL model with zero
restoring force — but only if z0 and z1 are independent. For our actual
regime (highly autocorrelated positions, rho = Corr(z0,z1) very close to 1 —
consistent with the tiny Var(u_t) already measured), a pure random walk
z1 = z0 + independent_noise (no restoring force at all) gives Cov(u,z0) =
Cov(noise,z0) = 0 exactly, so this mechanical effect vanishes in the
high-rho regime that actually applies here. It only becomes large for LOW
autocorrelation (e.g. testing against literally independent frames, which is
NOT representative of this data and would overstate "signal" spuriously —
tests in tests/test_compute_restoring_force_baseline.py validate against a
proper random-walk null, not i.i.d. frames, for exactly this reason). At the
already-measured rho for lag=1 (~0.9996), the expected mechanical reduction
under a TRUE null is (1-rho)/2 ~ 0.02% — negligible. A measured reduction
much larger than that is real signal, not this artifact.

Usage:
    python scripts/compute_restoring_force_baseline.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS" \
        --lag         1
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
    """See compute_position_mean_baseline.py — this computation never uses
    `pair`, so skip the standard collate_fn's expensive pad+stack of it."""
    z0_list, z1_list = [], []
    for (z0, z1), _pair, _fidx in batch:
        z0_list.append(z0)
        z1_list.append(z1)
    return torch.stack(z0_list), torch.stack(z1_list)


@torch.no_grad()
def compute_restoring_force_baseline(
    protein_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict[str, float]:
    """
    Single-pass computation of the best per-(position, channel) linear fit
    of u_t = z1 - z0 against z0 — see module docstring for the OLS algebra.

    Returns:
        {"zero_baseline_mse": float, "position_mean_mse": float,
         "restoring_force_mse": float, "n_samples": int,
         "n_excluded_self_pairs": int, "L": int}
    """
    loader = DataLoader(
        protein_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate_no_pair, drop_last=False,
    )

    sum_z0: torch.Tensor | None = None    # [L, d]
    sum_z0_sq: torch.Tensor | None = None  # [L, d]
    sum_u: torch.Tensor | None = None      # [L, d]
    sum_u_sq: torch.Tensor | None = None   # [L, d]  — per-channel, needed for per-(l,c) OLS
    sum_uz0: torch.Tensor | None = None    # [L, d]
    n_samples = 0
    n_excluded_self_pairs = 0

    for z0, z1 in loader:
        z0d = z0.double()
        u = (z1 - z0).double()

        if sum_z0 is None:
            shape = (u.shape[1], u.shape[2])
            sum_z0    = torch.zeros(shape, dtype=torch.float64)
            sum_z0_sq = torch.zeros(shape, dtype=torch.float64)
            sum_u     = torch.zeros(shape, dtype=torch.float64)
            sum_u_sq  = torch.zeros(shape, dtype=torch.float64)
            sum_uz0   = torch.zeros(shape, dtype=torch.float64)

        valid = u.abs().sum(dim=(1, 2)) > 0
        if not valid.all():
            n_excluded_self_pairs += int((~valid).sum().item())
            u, z0d = u[valid], z0d[valid]
        if u.shape[0] == 0:
            continue

        sum_z0    += z0d.sum(dim=0)
        sum_z0_sq += z0d.pow(2).sum(dim=0)
        sum_u     += u.sum(dim=0)
        sum_u_sq  += u.pow(2).sum(dim=0)
        sum_uz0   += (u * z0d).sum(dim=0)
        n_samples += u.shape[0]

    if n_samples == 0:
        raise RuntimeError("Dataset produced no non-self-paired samples.")

    L, d = sum_z0.shape
    n = n_samples

    # Per-(l,c) sums-of-squares/products about the mean (n * Var / n * Cov):
    ss_z0 = sum_z0_sq - sum_z0.pow(2) / n   # n * Var(z0), per (l,c)
    ss_u  = sum_u_sq  - sum_u.pow(2)  / n   # n * Var(u),  per (l,c) — this IS the
                                             # position-mean residual, per (l,c)
    sp_uz0 = sum_uz0 - (sum_u * sum_z0) / n  # n * Cov(u, z0), per (l,c)

    zero_baseline_mse = sum_u_sq.sum().item() / (n * L * d)
    position_mean_mse = ss_u.sum().item() / (n * L * d)

    # OLS residual SSE = SS_u - SP_uz0^2 / SS_z0, per (l,c). Guard against
    # zero-variance z0 at some (rare/degenerate) position — no valid slope
    # there, so that position falls back to its position-mean residual.
    safe_ss_z0 = ss_z0.clone()
    zero_var_mask = safe_ss_z0 <= 0
    safe_ss_z0[zero_var_mask] = 1.0  # avoid div-by-zero; numerator is 0 there too's not guaranteed, handle below
    explained = sp_uz0.pow(2) / safe_ss_z0
    explained[zero_var_mask] = 0.0   # no reduction possible where z0 has no variance
    restoring_force_resid = (ss_u - explained).clamp(min=0.0)  # numerical safety
    restoring_force_mse = restoring_force_resid.sum().item() / (n * L * d)

    return {
        "zero_baseline_mse": zero_baseline_mse,
        "position_mean_mse": position_mean_mse,
        "restoring_force_mse": restoring_force_mse,
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
        print(f"[restoring] Loaded normalization stats from {norm_stats_path}", flush=True)
    else:
        print(f"[restoring] No normalization stats found at {norm_stats_path} — "
              f"proceeding WITHOUT latent normalization.", flush=True)

    total_zero_sq  = 0.0
    total_mean_sq  = 0.0
    total_force_sq = 0.0
    total_n = 0

    print(f"[restoring] lag={args.lag} frames ({args.lag * 10}ps at native ATLAS stride)",
          flush=True)
    header = (f"{'Protein':<14} {'ZeroMSE':>12} {'PosMeanMSE':>12} {'RestoringMSE':>14} "
              f"{'%vsZero':>10} {'%vsPosMean':>12} {'N':>10}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for protein_id in protein_ids:
        protein_dataset = LatentDataset(
            args.latents_dir, protein_id, load_pair=False, norm_stats=norm_stats,
            lag=args.lag,
        )
        result = compute_restoring_force_baseline(
            protein_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        zero_mse  = result["zero_baseline_mse"]
        mean_mse  = result["position_mean_mse"]
        force_mse = result["restoring_force_mse"]
        n = result["n_samples"]
        n_excluded = result["n_excluded_self_pairs"]

        pct_vs_zero = (1 - force_mse / zero_mse) * 100 if zero_mse > 0 else 0.0
        pct_vs_mean = (1 - force_mse / mean_mse) * 100 if mean_mse > 0 else 0.0

        print(f"{protein_id:<14} {zero_mse:>12.6f} {mean_mse:>12.6f} {force_mse:>14.6f} "
              f"{pct_vs_zero:>9.2f}% {pct_vs_mean:>11.2f}% {n:>10}"
              + (f"  (excluded {n_excluded})" if n_excluded else ""),
              flush=True)

        total_zero_sq  += zero_mse  * n * result["L"]
        total_mean_sq  += mean_mse  * n * result["L"]
        total_force_sq += force_mse * n * result["L"]
        total_n += n * result["L"]

    print("-" * len(header), flush=True)
    overall_zero  = total_zero_sq  / total_n
    overall_mean  = total_mean_sq  / total_n
    overall_force = total_force_sq / total_n
    overall_pct_zero = (1 - overall_force / overall_zero) * 100 if overall_zero > 0 else 0.0
    overall_pct_mean = (1 - overall_force / overall_mean) * 100 if overall_mean > 0 else 0.0

    print(f"\n[summary] Aggregate zero-baseline MSE:          {overall_zero:.6f}")
    print(f"[summary] Aggregate position-mean baseline MSE:  {overall_mean:.6f}")
    print(f"[summary] Aggregate restoring-force baseline MSE: {overall_force:.6f}")
    print(f"[summary] Reduction vs. zero baseline:            {overall_pct_zero:.2f}%")
    print(f"[summary] Reduction vs. position-mean baseline:   {overall_pct_mean:.2f}%")
    print("\n[summary] The second number is the one that matters: it isolates whether")
    print("[summary] the CURRENT state (z0) predicts the next displacement, beyond")
    print("[summary] what a fixed per-position bias already explains (which was ~0%).")
    print("[summary] If this is also ~0%, z0 carries no linear signal about u_t either —")
    print("[summary] strong evidence the current single-snapshot conditioning has no")
    print("[summary] first-order signal to exploit, and richer conditioning (explicit")
    print("[summary] velocity/history) is needed, not more training-loop tuning.")


if __name__ == "__main__":
    main()
