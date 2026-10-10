"""
fluxion/training/flow_matching.py

Conditional Flow Matching (CFM) loss: regresses the straight-line velocity
field between a source z0 and target z1 in Chai-1 latent space. Deliberately
agnostic to what z0/z1 represent — train.py pairs z0=frame i, z1=frame i+1
(the original temporal formulation); train_static.py pairs z0=fresh Gaussian
noise, z1=an equilibrium-ensemble frame (the current static-ensemble
formulation, see train_static.py's module docstring). Optimal-transport
coupling of (z0, z1) pairs, when used, happens upstream of this class (see
ot_coupling.py) — this module only computes the loss for whatever pairing
it's handed.
"""

import torch
import torch.nn as nn


class FlowMatcher:
    def __init__(self, eps: float = 1e-5, t_dist: str = "uniform", beta_a: float = 0.5, beta_b: float = 0.5):
        """
        Args:
            eps: Small buffer to prevent numerical instability at t=0 and t=1
            t_dist: how t is drawn. "uniform" (default; the behaviour every earlier run used) or
                    "beta" (t ~ Beta(beta_a, beta_b); the default 0.5/0.5 puts mass near t = 0 and t = 1).
                    Phase 3a of the TMLR work changes ONLY this. No importance reweighting is applied:
                    the loss is the same MSE, sampled at different t.
        """
        if t_dist not in ("uniform", "beta"):
            raise ValueError(f"t_dist must be 'uniform' or 'beta', got {t_dist!r}")
        self.eps = eps
        self.t_dist = t_dist
        self._beta = torch.distributions.Beta(torch.tensor(float(beta_a)), torch.tensor(float(beta_b))) if t_dist == "beta" else None

    def sample_t(self, B: int, device) -> torch.Tensor:
        u = torch.rand((B,), device=device) if self._beta is None else self._beta.sample((B,)).to(device)
        return u * (1.0 - 2 * self.eps) + self.eps

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
            z0:           Source point in latent space (noise or frame i)   [B, L, d_single]
            z1:           Target point in latent space (an equilibrium
                          sample or frame i+1)                              [B, L, d_single]
            pair:         Structural conditioning                          [B, L, L, d_pair]
            padding_mask: True = padded position                          [B, L] bool or None

        Returns:
            Scalar MSE loss averaged over valid (non-padded) positions.
        """
        B = z0.shape[0]
        device = z0.device

        t = self.sample_t(B, device)
        t_expand = t.unsqueeze(1).unsqueeze(2)

        z_t = (1.0 - t_expand) * z0 + t_expand * z1
        u_t = z1 - z0

        v_pred = model(z_t, pair, t, padding_mask)

        loss = torch.nn.functional.mse_loss(v_pred, u_t, reduction="none")  # [B, L, d]

        if padding_mask is not None:
            # Zero out padded positions; average only over valid tokens per sample.
            valid   = (~padding_mask).float().unsqueeze(-1)          # [B, L, 1]
            n_valid = valid.sum(dim=1).clamp(min=1.0)                # [B, 1] — clamp avoids 0/0
            loss    = (loss * valid).sum(dim=(1, 2)) / (n_valid.squeeze(-1) * loss.shape[-1])
            return loss.mean()

        return loss.mean(dim=(1, 2)).mean()
