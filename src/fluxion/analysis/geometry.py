"""analysis/geometry.py: pure-numpy structural helpers."""
from __future__ import annotations
import numpy as np


def kabsch_pairwise_rmsd(X: np.ndarray) -> np.ndarray:
    n, L, _ = X.shape
    X = X - X.mean(1, keepdims=True)
    sq = (X ** 2).sum((1, 2))
    out = np.zeros((n, n))
    for i in range(n):
        H = np.einsum("ad,jae->jde", X[i], X)                        # [n, 3, 3]
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        S2 = S.copy(); S2[:, 2] *= d
        out[i] = np.sqrt(np.maximum(sq[i] + sq - 2 * S2.sum(1), 0) / L)
    return out


def kabsch_align(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Optimally superpose every frame of X [n, L, 3] onto ref [L, 3] (rotation + translation, no reflection)."""
    ref_c = ref - ref.mean(0, keepdims=True)
    Xc = X - X.mean(1, keepdims=True)
    H = np.einsum("nad,ae->nde", Xc, ref_c)                         # [n, 3, 3]
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.zeros_like(H); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = d
    R = np.einsum("nij,njk,nkl->nil", U, D, Vt)                     # [n, 3, 3] maps Xc -> ref frame (row vectors: Xc @ R)
    return np.einsum("nad,nde->nae", Xc, R) + ref.mean(0, keepdims=True)


def superposed_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two conformations a, b [L, 3] after optimal superposition of a onto b (rotation + translation)."""
    aligned = kabsch_align(a[None], b)[0]
    return float(np.sqrt(((aligned - b) ** 2).sum(-1).mean()))


def pca_fit(Xtr, k: int):
    """Top-k principal axes of Xtr (torch tensor, rows = frames) via the Gram matrix of the centered frames (no D x D covariance;
    float32 products, double only for the small Gram eigenproblem). Returns (mean [1, D], unit-norm axes [D, k])."""
    import torch
    mu = Xtr.mean(0, keepdim=True)
    A = Xtr - mu
    w, V = torch.linalg.eigh((A @ A.T).double())
    idx = torch.argsort(w, descending=True)[:k]
    w, V = w[idx].clamp_min(1e-12), V[:, idx]
    return mu, (A.T @ V.float()) / w.sqrt().float()


def pca_project(X, mu, comp):
    """Scores of the rows of X on axes fitted by pca_fit."""
    return (X - mu) @ comp


def pca_scores(Xtr, Xall, k: int, chunk: int = 500):
    """Scores of Xall on the top-k principal axes of Xtr (torch tensors, rows = frames)."""
    import torch
    mu, comp = pca_fit(Xtr, k)
    return torch.cat([pca_project(Xall[i:i + chunk], mu, comp) for i in range(0, Xall.shape[0], chunk)])
