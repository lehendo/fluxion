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
    def __init__(self, eps: float = 1e-5, t_dist: str = "uniform", beta_a: float = 0.5, beta_b: float = 0.5,
                 t_clip: tuple[float, float] | None = None, t_mode: str = "clamp"):
        """
        Args:
            eps: Small buffer to prevent numerical instability at t=0 and t=1
            t_dist: how t is drawn. "uniform" (default; the behaviour every earlier run used) or
                    "beta" (t ~ Beta(beta_a, beta_b); the default 0.5/0.5 puts mass near t = 0 and t = 1).
                    Phase 3a of the TMLR work changes ONLY this. No importance reweighting is applied:
                    the loss is the same MSE, sampled at different t.
            t_mode: "clamp" (default) or "truncate" - how t_clip is applied. "truncate" renormalizes the drawn distribution to [lo, hi] by rejection (no
                    boundary point masses; the endpoint weight of the Beta is kept but reduced). Used only by the Phase 3 stability diagnostic.
            t_clip: optional (lo, hi). If given, t is CLAMPED to [lo, hi] after drawing (eps is then unused). This is a clamp, not a truncation:
                    the mass outside [lo, hi] becomes point masses at lo and hi (about 17% at each end for Beta(0.25, 0.25) with lo = 0.01, 1% for uniform).
                    (A first implementation truncated instead; that removes the endpoint mass and cuts the predicted attainable gain, see the log.)
        """
        if t_dist not in ("uniform", "beta"):
            raise ValueError(f"t_dist must be 'uniform' or 'beta', got {t_dist!r}")
        self.eps = eps
        self.t_dist = t_dist
        self._beta = torch.distributions.Beta(torch.tensor(float(beta_a)), torch.tensor(float(beta_b))) if t_dist == "beta" else None
        if t_clip is not None and not 0.0 <= t_clip[0] < t_clip[1] <= 1.0:
            raise ValueError(f"t_clip must satisfy 0 <= lo < hi <= 1, got {t_clip}")
        if t_mode not in ("clamp", "truncate"):
            raise ValueError(f"t_mode must be 'clamp' or 'truncate', got {t_mode!r}")
        self.t_clip = t_clip
        self.t_mode = t_mode

    def _draw(self, B: int, device) -> torch.Tensor:
        return torch.rand((B,), device=device) if self._beta is None else self._beta.sample((B,)).to(device)

    def sample_t(self, B: int, device) -> torch.Tensor:
        if self.t_clip is None:
            return self._draw(B, device) * (1.0 - 2 * self.eps) + self.eps
        lo, hi = self.t_clip
        t = self._draw(B, device)
        if self.t_mode == "clamp":
            return t.clamp(lo, hi)
        bad = (t < lo) | (t > hi)
        while bad.any():                                    # rejection sampling = truncation to [lo, hi]
            t[bad] = self._draw(int(bad.sum()), device)
            bad = (t < lo) | (t > hi)
        return t

    def compute_loss(
        self,
        model: nn.Module,
        z0: torch.Tensor,
        z1: torch.Tensor,
        pair: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_stats: bool = False,
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
            denom   = n_valid.squeeze(-1) * loss.shape[-1]
            loss    = (loss * valid).sum(dim=(1, 2)) / denom
            if return_stats:                                          # v = 0 loss on the same samples: the 'identity loss'
                v0 = ((u_t.float() ** 2) * valid).sum(dim=(1, 2)) / denom
                return loss.mean(), v0.mean()
            return loss.mean()

        if return_stats:
            return loss.mean(dim=(1, 2)).mean(), (u_t.float() ** 2).mean(dim=(1, 2)).mean()
        return loss.mean(dim=(1, 2)).mean()
