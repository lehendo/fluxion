"""
fluxion/training/flow_matching.py

Optimal Transport Conditional Flow Matching (OT-CFM) for consecutive MD frames.
Learns the vector field driving conformational dynamics.
"""

import torch
import torch.nn as nn


class FlowMatcher:
    def __init__(self, eps: float = 1e-5):
        """
        Args:
            eps: Small buffer to prevent numerical instability at t=0 and t=1
        """
        self.eps = eps

    def compute_loss(
        self,
        model: nn.Module,
        z0: torch.Tensor,
        z1: torch.Tensor,
        pair: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model:        FluxionDiT network
            z0:           Token representations at frame i   [B, L, d_single]
            z1:           Token representations at frame i+1 [B, L, d_single]
            pair:         Static structural conditioning      [B, L, L, d_pair]
            padding_mask: True = padded position             [B, L] bool or None

        Returns:
            Scalar MSE loss averaged over valid (non-padded) positions.
        """
        B = z0.shape[0]
        device = z0.device

        t = torch.rand((B,), device=device) * (1.0 - 2 * self.eps) + self.eps
        t_expand = t.unsqueeze(1).unsqueeze(2)

        z_t = (1.0 - t_expand) * z0 + t_expand * z1
        u_t = z1 - z0

        v_pred = model(z_t, pair, t, padding_mask)

        loss = torch.nn.functional.mse_loss(v_pred, u_t, reduction="none")  # [B, L, d]

        if padding_mask is not None:
            # Zero out padded positions; average only over valid tokens per sample.
            valid = (~padding_mask).float().unsqueeze(-1)   # [B, L, 1]
            n_valid = valid.sum(dim=1)                       # [B, 1]
            loss = (loss * valid).sum(dim=(1, 2)) / (n_valid.squeeze(-1) * loss.shape[-1])
            return loss.mean()

        return loss.mean(dim=(1, 2)).mean()
