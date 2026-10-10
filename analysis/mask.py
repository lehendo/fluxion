"""
analysis/mask.py

Real-token mask for Chai-1 bucket-padded latents.

The mask is DERIVED FROM SEQUENCE LENGTH: the first `n_res` positions are real tokens, the
remaining `L_bucket - n_res` are bucket padding. Zero frame-variance is used ONLY to assert
that the mask is right, never to build it -- building it from variance would also discard
genuinely rigid residues (near-zero displacement), biasing MSE upward exactly where the
dynamics are least interesting.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def n_residues(sequences_json: Path, protein: str) -> int:
    data = json.loads(Path(sequences_json).read_text())
    if protein not in data:
        raise KeyError(f"{protein} missing from {sequences_json}; run scripts/fetch_atlas_sequences.py")
    return int(data[protein]["n_residues"])


def real_mask(L_bucket: int, n_res: int) -> torch.Tensor:
    if not 0 < n_res <= L_bucket:
        raise ValueError(f"need 0 < n_res <= L_bucket, got n_res={n_res}, L_bucket={L_bucket}")
    return torch.arange(L_bucket) < n_res


def verify_mask(mask: torch.Tensor, frames: torch.Tensor, tol: float = 1e-12) -> dict:
    """
    Asserts (a) the number of masked positions equals L_bucket - n_res and (b) every masked
    position is constant across `frames` [n, L, d]. Returns diagnostics, including how many
    REAL positions also have zero variance (expected 0; if not, they are kept, not masked).
    """
    L = mask.shape[0]
    var_pos = frames.var(dim=0).sum(dim=-1)                      # [L]
    padded = ~mask
    n_pad = int(padded.sum())
    assert n_pad == L - int(mask.sum()), "mask count inconsistent with L_bucket - n_res"
    nonconst = int((var_pos[padded] > tol).sum())
    assert nonconst == 0, (
        f"{nonconst} of {n_pad} masked (padding) positions vary across frames: the sequence-length "
        "mask is wrong for this protein (tokenization differs from residue count?)")
    return {"n_padded": n_pad, "n_real": int(mask.sum()),
            "n_real_with_zero_variance": int((var_pos[mask] <= tol).sum())}
