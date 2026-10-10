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
