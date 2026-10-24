"""analysis/axis_stability.py: stability of PCA axes across replicas (diagnostics S-a and S-b of the B1-10ps controls). Pure torch. Rows are frames."""
from __future__ import annotations
import torch


def fit_axes(X, k: int):
    """Top-k principal axes of X [n, D] via the Gram matrix. Returns (mean [1, D], axes [D, k] unit-norm, variances [k] = eigenvalue / (n - 1))."""
    mu = X.mean(0, keepdim=True); A = X - mu
    w, V = torch.linalg.eigh((A @ A.T).double())
    idx = torch.argsort(w, descending=True)[:k]; w, V = w[idx].clamp_min(1e-12), V[:, idx]
    return mu, (A.T @ V.float()) / w.sqrt().float(), (w / (X.shape[0] - 1)).float()


def principal_cos2(UA, UB, m: int) -> float:
    """Mean cos^2 of the principal angles between the top-m subspaces spanned by the columns of UA and UB (1 = identical subspaces)."""
    s = torch.linalg.svdvals((UA[:, :m].T @ UB[:, :m]).double())
    return float((s ** 2).mean())


def axis_cos(UA, UB):
    """|cos| between the i-th axes of two bases, for each index i [k]."""
    return (UA * UB).sum(0).abs()


def capture_ratio(Xh, U, m: int) -> float:
    """Variance of the held-out frames Xh [n, D] (about their own mean) captured by the top-m subspace of U, divided by the variance captured by Xh's own best m-dim subspace."""
    A = Xh - Xh.mean(0, keepdim=True)
    own = torch.linalg.eigvalsh((A @ A.T).double()).sort(descending=True).values[:m].sum()
    return float(((A @ U[:, :m]).double() ** 2).sum() / own)


def heldout_variance(Xh, mu, U):
    """Variance of the held-out frames' scores on each training axis (scores centered on the TRAINING mean, as in the probes) [k]."""
    S = (Xh - mu) @ U
    return (S ** 2).mean(0)
