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

IMPORTANT — a confound in restoring_force_mse, and the fix
-------------------------------------------------------------
u = z1 - z0 mechanically CONTAINS "-z0" as a term. Regressing u on z0
therefore hits a MECHANICAL ceiling of exactly 50% "reduction" under TRUE
INDEPENDENCE (z0, z1 uncorrelated) — not just under genuine restoring-force
dynamics. Worse: a "perfect, instantaneous restoring force" (system fully
resets to equilibrium every single lag step) is mathematically IDENTICAL to
independence, so restoring_force_mse alone cannot tell real fast dynamics
apart from pure memorylessness — both land at 50%. (Discovered 2026-07-12:
every one of 14 pilot proteins landed at 49.95-50.24%, an implausibly tight
cluster for real biological variation — the signature of hitting this
mechanical ceiling, not of measuring real per-protein dynamics.)

The fix: z1_predictability_mse regresses z1 DIRECTLY on z0 (not u = z1-z0 on
z0), which has no such self-inclusion term. Reduction vs. z1_variance_mse
(predicting z1 by its own mean) is exactly rho^2, the squared correlation
between z0 and z1 — 0% at true independence, growing only with genuine
retained information. THIS is the number that actually answers "does the
current state predict the future state," unconfounded. Derived from the
same sufficient statistics already collected for restoring_force_mse, no
extra data pass needed.

Interpretation (use z1_predictability_mse / z1_variance_mse, not
restoring_force_mse / position_mean_mse, for the real verdict)
-------------------------------------------------------------------
- If the z1-predictability reduction is close to 0%: z0 carries no linear
  information about z1 at all — full memorylessness at this lag. Combined
  with the flat lag-sweep result, this is strong evidence Fluxion's OT-CFM
  formulation has no first-order-linear signal to exploit from (z_t, pair, t)
  in this latent space at any tested timescale — arguing for a genuinely
  different conditioning (explicit velocity/history, or a different
  representation) rather than continued tuning of the current setup.
- If it's meaningfully positive: real, state-dependent linear structure
  exists that a model conditioning on z_t could in principle learn.

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
    of u_t = z1 - z0 against z0, AND the unconfounded z0->z1 predictability
    check — see module docstring for the OLS algebra and why both are
    reported (restoring_force_mse alone is confounded; z1_predictability_mse
    is the one to actually read).

    Returns:
        {"zero_baseline_mse": float, "position_mean_mse": float,
         "restoring_force_mse": float, "z1_variance_mse": float,
         "z1_predictability_mse": float, "n_samples": int,
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

    # --- Unconfounded predictability check ---
    # regressing u = z1 - z0 on z0 hits a mechanical 50% "reduction" ceiling
    # under TRUE independence (z0, z1 uncorrelated), because u mechanically
    # contains "-z0" as a term — see module docstring. That makes
    # restoring_force_mse ABOVE unable to distinguish "genuine fast
    # mean-reversion" from "no memory at all." Sidestep this entirely by
    # regressing z1 directly on z0 instead of regressing u on z0 — this
    # gives exactly rho^2 (squared correlation) with NO such artifact: 0%
    # at true independence, growing only with genuine retained information.
    # Derived from the SAME sufficient statistics already collected above
    # (z1 = z0 + u), no second pass over the data needed.
    sum_z1    = sum_z0 + sum_u
    sum_z1_sq = sum_z0_sq + 2 * sum_uz0 + sum_u_sq
    sum_z0z1  = sum_z0_sq + sum_uz0

    ss_z1   = sum_z1_sq - sum_z1.pow(2) / n       # n * Var(z1), per (l,c)
    sp_z0z1 = sum_z0z1  - (sum_z0 * sum_z1) / n    # n * Cov(z0,z1), per (l,c)

    z1_variance_mse = ss_z1.sum().item() / (n * L * d)  # baseline: predict z1 by its own mean

    z1_explained = sp_z0z1.pow(2) / safe_ss_z0
    z1_explained[zero_var_mask] = 0.0
    z1_predict_resid = (ss_z1 - z1_explained).clamp(min=0.0)
    z1_predictability_mse = z1_predict_resid.sum().item() / (n * L * d)

    return {
        "zero_baseline_mse": zero_baseline_mse,
        "position_mean_mse": position_mean_mse,
        "restoring_force_mse": restoring_force_mse,
        "z1_variance_mse": z1_variance_mse,
        "z1_predictability_mse": z1_predictability_mse,
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
    total_z1var_sq  = 0.0
    total_z1pred_sq = 0.0
    total_n = 0

    print(f"[restoring] lag={args.lag} frames ({args.lag * 10}ps at native ATLAS stride)",
          flush=True)
    print("[restoring] NOTE: %vsPosMean (from u=z1-z0 regressed on z0) is confounded — "
          "see module docstring. Read %z1Predict (unconfounded, = rho^2) for the real answer.",
          flush=True)
    header = (f"{'Protein':<14} {'ZeroMSE':>12} {'PosMeanMSE':>12} {'RestoringMSE':>14} "
              f"{'%vsPosMean':>12} {'%z1Predict':>12} {'N':>10}")
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
        z1var_mse  = result["z1_variance_mse"]
        z1pred_mse = result["z1_predictability_mse"]
        n = result["n_samples"]
        n_excluded = result["n_excluded_self_pairs"]

        pct_vs_mean = (1 - force_mse / mean_mse) * 100 if mean_mse > 0 else 0.0
        pct_z1predict = (1 - z1pred_mse / z1var_mse) * 100 if z1var_mse > 0 else 0.0

        print(f"{protein_id:<14} {zero_mse:>12.6f} {mean_mse:>12.6f} {force_mse:>14.6f} "
              f"{pct_vs_mean:>11.2f}% {pct_z1predict:>11.2f}% {n:>10}"
              + (f"  (excluded {n_excluded})" if n_excluded else ""),
              flush=True)

        total_z1var_sq  += z1var_mse  * n * result["L"]
        total_z1pred_sq += z1pred_mse * n * result["L"]
        total_zero_sq  += zero_mse  * n * result["L"]
        total_mean_sq  += mean_mse  * n * result["L"]
        total_force_sq += force_mse * n * result["L"]
        total_n += n * result["L"]

    print("-" * len(header), flush=True)
    overall_zero   = total_zero_sq   / total_n
    overall_mean   = total_mean_sq   / total_n
    overall_force  = total_force_sq  / total_n
    overall_z1var  = total_z1var_sq  / total_n
    overall_z1pred = total_z1pred_sq / total_n
    overall_pct_mean     = (1 - overall_force  / overall_mean)  * 100 if overall_mean  > 0 else 0.0
    overall_pct_z1predict = (1 - overall_z1pred / overall_z1var) * 100 if overall_z1var > 0 else 0.0

    print(f"\n[summary] Aggregate zero-baseline MSE:            {overall_zero:.6f}")
    print(f"[summary] Aggregate position-mean baseline MSE:    {overall_mean:.6f}")
    print(f"[summary] Aggregate restoring-force baseline MSE:  {overall_force:.6f}  "
          f"(CONFOUNDED — see note above)")
    print(f"[summary] Reduction vs. position-mean (confounded): {overall_pct_mean:.2f}%")
    print()
    print(f"[summary] Aggregate z1 variance (baseline):        {overall_z1var:.6f}")
    print(f"[summary] Aggregate z1 predictability MSE:         {overall_z1pred:.6f}")
    print(f"[summary] Unconfounded z0->z1 predictability (rho^2): {overall_pct_z1predict:.2f}%")
    print("\n[summary] The LAST number is the one that matters: it isolates whether the")
    print("[summary] CURRENT state z0 predicts the FUTURE state z1 at all, with no")
    print("[summary] mechanical artifact. If it's ~0%, z0 carries no linear signal about")
    print("[summary] z1 — strong evidence the current single-snapshot conditioning has no")
    print("[summary] first-order signal to exploit, and richer conditioning (explicit")
    print("[summary] velocity/history) is needed, not more training-loop tuning.")


if __name__ == "__main__":
    main()
