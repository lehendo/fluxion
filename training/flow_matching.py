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
    ) -> torch.Tensor:
        """
        Args:
            model: The FluxionDiT network
            z0: Token representations at frame i   [B, L, d_model]
            z1: Token representations at frame i+1 [B, L, d_model]
            pair: Static structural geometry        [B, L, L, d_pair]

        Returns:
            MSE loss between predicted velocity and target ODE velocity.
        """
        B = z0.shape[0]
        device = z0.device

        # 1. Sample continuous timestep t ~ U(eps, 1-eps)
        t = torch.rand((B,), device=device) * (1.0 - 2 * self.eps) + self.eps

        # Reshape t for broadcasting [B, 1, 1]
        t_expand = t.unsqueeze(1).unsqueeze(2)

        # 2. Optimal Transport Interpolant (straight-line path)
        # z_t = (1 - t) * z0 + t * z1
        z_t = (1.0 - t_expand) * z0 + t_expand * z1

        # 3. Target Velocity Field (derivative of z_t with respect to t)
        # u_t = d/dt [(1-t)*z0 + t*z1] = z1 - z0
        u_t = z1 - z0

        # 4. Model Prediction
        v_pred = model(z_t, pair, t)

        # 5. Regression Loss: E[ || v_pred - u_t ||^2 ]
        loss = torch.nn.functional.mse_loss(v_pred, u_t, reduction="none")

        # Average over sequence length, then batch
        return loss.mean(dim=(1, 2)).mean()
