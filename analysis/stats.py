"""
analysis/stats.py

Pure-function statistics for the identity-relative rescoring (Phase 1).

Conventions
-----------
* Every improvement is reported against IDENTITY (z1_hat = z0), never the mean:
      improvement% = 100 * (1 - MSE_model / MSE_identity)
  Positive = better than persistence. Zero = no better than copying the input.
* Folds are leave-one-PDB-ENTRY-out. Chains of the same PDB entry (1bxy_A/B,
  1b2s_E/F) are siblings; holding out one while training on the other leaks.
* Uncertainty: the independent unit is the PDB entry, so the primary CI is a
  bootstrap over entries. Within one protein, frames are autocorrelated, so
  per-protein CIs resample contiguous time blocks, never individual frames.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def pdb_entry(protein_id: str) -> str:
    """'1bxy_A' -> '1bxy'. The leave-out unit."""
    return protein_id.split("_")[0]


def entry_groups(protein_ids: list[str]) -> dict[str, list[str]]:
    """Maps PDB entry -> its protein/chain ids (sorted, deterministic)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for p in protein_ids:
        groups[pdb_entry(p)].append(p)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def pct_improvement(mse_model: float, mse_ref: float) -> float:
    """100*(1 - model/ref). Always name `ref` when reporting."""
    if mse_ref <= 0:
        raise ValueError(f"reference MSE must be > 0, got {mse_ref}")
    return 100.0 * (1.0 - mse_model / mse_ref)


def lag_frames_to_label(lag_frames: int, stride_ps: float = 10.0) -> str:
    """1 -> '10 ps', 100 -> '1 ns'. ATLAS native stride is 10 ps/frame."""
    ps = lag_frames * stride_ps
    if ps >= 1000:
        ns = ps / 1000
        return f"{ns:g} ns"
    return f"{ps:g} ps"


def time_blocks(replica: np.ndarray, frame: np.ndarray, block_len: int) -> np.ndarray:
    """
    Integer block id per row: rows in the same replica whose frame index falls in
    the same `block_len`-frame window share a block. Resampling whole blocks
    respects temporal autocorrelation (resampling single frames would not).
    """
    keys = [f"{int(r)}:{int(f) // block_len}" for r, f in zip(replica, frame)]
    _, ids = np.unique(keys, return_inverse=True)
    return ids


def block_bootstrap_improvement(
    sqerr_model: np.ndarray,
    sqerr_ref: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Point estimate and 95% percentile CI of 100*(1 - mean(model)/mean(ref)),
    resampling whole time blocks with replacement. Returns (est, lo, hi).
    """
    sqerr_model = np.asarray(sqerr_model, dtype=np.float64)
    sqerr_ref = np.asarray(sqerr_ref, dtype=np.float64)
    n_blocks = int(block_ids.max()) + 1
    sum_m = np.bincount(block_ids, weights=sqerr_model, minlength=n_blocks)
    sum_r = np.bincount(block_ids, weights=sqerr_ref, minlength=n_blocks)
    cnt = np.bincount(block_ids, minlength=n_blocks).astype(np.float64)
    est = pct_improvement(sum_m.sum() / cnt.sum(), sum_r.sum() / cnt.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    bm, br, bc = sum_m[draws].sum(1), sum_r[draws].sum(1), cnt[draws].sum(1)
    boots = 100.0 * (1.0 - (bm / bc) / (br / bc))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def entry_bootstrap_mean(
    per_protein: dict[str, float],
    n_boot: int = 10000,
    seed: int = 0,
    stat: str = "mean",
) -> tuple[float, float, float]:
    """
    Primary across-protein summary. `per_protein` maps protein id -> its
    improvement (%). Resamples PDB ENTRIES with replacement (all chains of a
    drawn entry come along), so sibling chains are never treated as
    independent. Returns (point estimate, lo, hi) for the mean (or median).
    """
    groups = entry_groups(list(per_protein))
    entries = list(groups)
    fn = np.mean if stat == "mean" else np.median
    est = float(fn(list(per_protein.values())))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(entries), size=len(entries))
        vals = [per_protein[p] for i in pick for p in groups[entries[i]]]
        boots[b] = fn(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0
