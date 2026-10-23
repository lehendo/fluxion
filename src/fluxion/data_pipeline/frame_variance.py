"""Frame-variance gate for latent extraction (torch only, so it is importable and testable without MDAnalysis or chai-lab)."""

from __future__ import annotations

import torch


FRAME_VARIANCE_MIN = 1e-2


def check_frame_variance(single_a: torch.Tensor, single_b: torch.Tensor, frame_b: int,
                         threshold: float = FRAME_VARIANCE_MIN) -> float:
    """Frame-variance gate: the trunk's single representation of two distinct frames must differ.

    Returns mean|a - b|. Raises RuntimeError when it is below `threshold`, i.e. when coordinate injection has
    silently gone inert and every frame would map to (nearly) the same latent. This is the permanent regression
    check for that failure: it fails loudly within minutes instead of after hours of extraction.
    """
    if single_a.shape != single_b.shape:
        raise ValueError(f"single representations must have the same shape, got {tuple(single_a.shape)} vs {tuple(single_b.shape)}")
    mean_abs_diff = (single_a.float() - single_b.float()).abs().mean().item()
    if mean_abs_diff < threshold:
        raise RuntimeError(
            f"Frame-variance gate failed: frame 0 and frame {frame_b} produced near-identical trunk output "
            f"(mean|diff|={mean_abs_diff:.6f}, threshold {threshold}). Coordinate injection is very likely inert "
            f"again; refusing to extract a degenerate dataset."
        )
    return mean_abs_diff
