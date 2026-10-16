"""analysis/b1_controls.py: per-dimension error decompositions for the B1-10ps controls C1 (whitened metric), C2 (component-index decomposition) and C3 (per-channel
shrinkage without a free intercept). Pure torch/numpy. All predictors are fit on the TRAINING pairs only and scored on the held-out pairs, per dimension.
"""
from __future__ import annotations
import numpy as np
import torch
from analysis.probes import fit_linear_diag, predict_linear_diag

PREDICTORS = ("identity", "mean", "shrink", "linear_diag", "noint")
BANDS = [(0, 5), (5, 20), (20, 60), (60, 150), (150, 384)]       # component-index bands [1-5], [6-20], [21-60], [61-150], [151-384]


def per_dim_mse(train, test, whiten_std=None):
    """train = (z0, z1) concatenated over the training replicas, test = (t0, t1); all [n, k]. If whiten_std ([k]) is given, every dimension is divided by it first.
    Returns {predictor: per-dimension MSE on the test pairs [k]}. 'shrink' is the single-alpha predictor alpha*z0, 'noint' is one alpha_c per dimension and no intercept
    (scores are centered on the training frames, so the shrinkage target is the training mean 0), 'linear_diag' a free slope and intercept per dimension."""
    z0, z1 = train; t0, t1 = test
    if whiten_std is not None:
        s = whiten_std.unsqueeze(0)
        z0, z1, t0, t1 = z0 / s, z1 / s, t0 / s, t1 / s
    alpha = float((z0 * z1).sum() / (z0 * z0).sum())
    alpha_c = (z0.double() * z1.double()).sum(0) / (z0.double() ** 2).sum(0).clamp_min(1e-30)
    a, k0, _ = fit_linear_diag(z0, z1)
    preds = {"identity": t0, "mean": z1.mean(0, keepdim=True).expand_as(t1), "shrink": alpha * t0,
             "linear_diag": predict_linear_diag(t0, a, k0), "noint": (alpha_c.float().unsqueeze(0)) * t0}
    return {n: ((p.double() - t1.double()) ** 2).mean(0).numpy() for n, p in preds.items()}


def pct_vs_identity(mse_by_fold, pred):
    """B1 convention: 100 x (1 - mean over folds of total MSE(pred) / mean over folds of total MSE(identity)); total = mean over dimensions."""
    num = np.mean([m[pred].mean() for m in mse_by_fold]); den = np.mean([m["identity"].mean() for m in mse_by_fold])
    return float(100 * (1 - num / den))


def band_contrib(mse_by_fold, pred, bands=BANDS):
    """Variance-weighted contribution of each component band to pct_vs_identity (points). Bands sum to pct_vs_identity. Truncated at k."""
    k = mse_by_fold[0]["identity"].shape[0]
    mid = np.mean([m["identity"] for m in mse_by_fold], 0); mp = np.mean([m[pred] for m in mse_by_fold], 0)
    tot = mid.sum()
    out = [float(100 * (mid[lo:min(hi, k)] - mp[lo:min(hi, k)]).sum() / tot) for lo, hi in bands if lo < k]
    half = k // 2
    return out, float(100 * (mid[:half] - mp[:half]).sum() / tot), float(100 * (mid[half:] - mp[half:]).sum() / tot)


def band_unweighted(mse_by_fold, pred, bands=BANDS):
    """Mean over the dimensions of a band of the per-dimension % improvement over identity (the view that whitening makes the metric see)."""
    k = mse_by_fold[0]["identity"].shape[0]
    mid = np.mean([m["identity"] for m in mse_by_fold], 0); mp = np.mean([m[pred] for m in mse_by_fold], 0)
    per = 100 * (1 - mp / np.maximum(mid, 1e-30))
    return [float(per[lo:min(hi, k)].mean()) for lo, hi in bands if lo < k]
