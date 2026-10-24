#!/usr/bin/env python3
"""A runnable miniature of the paper's three measurement hazards, on synthetic latents (CPU, a few seconds).

The synthetic "trajectory" is a stationary reversible AR(1) process per channel, observed through bucket padding whose constant changes between
extraction shards (hazard 1), with independent per-frame extraction noise (hazard 2), and scored against identity and the mean at several lags (hazard 3).
It also prints the attenuation bounds of the flow-matching objective (paper Section 5.4). Real data are not shipped; see the README for the real pipeline.

    python examples/synthetic_hazards.py
"""
from __future__ import annotations

import numpy as np
import torch

from fluxion.analysis.mask import real_mask
from fluxion.analysis.noise_recompose import corrected_gain
from fluxion.analysis.theory import attenuation_R, f_closed_form, rho_from_f0

RNG = np.random.default_rng(0)
N_FRAMES, N_RES, BUCKET, D = 4000, 24, 32, 12
RHO_STEP = 0.90          # lag-1 autocorrelation of the signal
NOISE_VAR = 0.08         # extraction-noise variance relative to a unit-variance signal
SHARD = 500              # frames per extraction shard


def simulate():
    """Signal [T, L, D] with unit variance (AR(1) per channel), padded to BUCKET with a per-shard constant, plus extraction noise on real tokens."""
    e = RNG.standard_normal((N_FRAMES, N_RES, D)) * np.sqrt(1 - RHO_STEP ** 2)
    s = np.zeros_like(e); s[0] = RNG.standard_normal((N_RES, D))
    for t in range(1, N_FRAMES):
        s[t] = RHO_STEP * s[t - 1] + e[t]
    z = np.zeros((N_FRAMES, BUCKET, D))
    z[:, :N_RES] = s + RNG.standard_normal(s.shape) * np.sqrt(NOISE_VAR)       # hazard 2: noise in the real tokens
    for k in range(0, N_FRAMES, SHARD):                                         # hazard 1: the padded value changes between shards
        z[k:k + SHARD, N_RES:] = RNG.standard_normal((1, BUCKET - N_RES, D)) * 1.5
    return torch.tensor(z, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)


def improvement(mse_model: float, mse_ref: float) -> float:
    return 100.0 * (1.0 - mse_model / mse_ref)


def hazard_padding(z: torch.Tensor) -> None:
    """Identity against the mean baseline, counting padded tokens against masking them."""
    z0, z1 = z[:-1], z[1:]
    mean = z.mean(0, keepdim=True)                          # the protein's own per-position mean, over all frames
    mask = real_mask(BUCKET, N_RES)
    out = {}
    for name, m in (("counted", torch.ones(BUCKET, dtype=torch.bool)), ("masked", mask)):
        e_id = ((z1 - z0)[:, m] ** 2).mean().item(); e_mean = ((z1 - mean)[:, m] ** 2).mean().item()
        out[name] = improvement(e_id, e_mean)
    print(f"[padding]  identity's improvement over the mean baseline: counted {out['counted']:6.1f}%   masked {out['masked']:6.1f}%   inflation {out['counted'] - out['masked']:5.1f} points")


def hazard_noise(z: torch.Tensor, s: torch.Tensor) -> None:
    """The one-parameter shrinkage gain at lag 1, with and without the extraction noise."""
    mask = real_mask(BUCKET, N_RES)
    def gain(x):
        x = x - x.mean(0, keepdim=True); a, b = x[:-1], x[1:]
        alpha = (a * b).sum() / (a * a).sum()
        return improvement(((b - alpha * a) ** 2).mean().item(), ((b - a) ** 2).mean().item())
    g_noisy, g_clean = gain(z[:, mask]), gain(s)
    nu = NOISE_VAR / (1 + NOISE_VAR)                                         # noise share of the observed variance
    g_corr = 100 * float(corrected_gain(g_noisy / 100, nu))
    print(f"[noise]    shrinkage gain over identity: noisy {g_noisy:5.2f}%   noise-free truth {g_clean:5.2f}%   corrected with the known noise share {g_corr:5.2f}%")


def hazard_baselines(s: torch.Tensor) -> None:
    """MSE(identity) / MSE(mean) = 2 (1 - rho(lag)): the mean baseline is the harder to beat once rho < 1/2."""
    x = s - s.mean(0, keepdim=True); row = []
    for lag in (1, 3, 7, 15, 40, 120):
        a, b = x[:-lag], x[lag:]
        ratio = ((b - a) ** 2).mean().item() / (b ** 2).mean().item()
        row.append(f"lag {lag:3d}: {ratio:4.2f}")
    print("[baseline] identity / mean MSE   " + "   ".join(row) + "   (> 1: identity is the worse baseline)")


def attenuation() -> None:
    """Flow matching between two frames, uniform t: only 33 to 43% of the endpoint signal is attainable."""
    print("[theory]   attenuation 1/R(rho) of the uniform-t objective: " + "   ".join(f"rho={r}: {1 / attenuation_R(r):.2f}" for r in (0.0, 0.5, 0.7, 0.9, 0.999)) + "   (bounds 2.33 to 3.0)")
    f0 = 0.1523; rho = rho_from_f0(f0)
    print(f"[theory]   with the paper's measured f(0) = {f0:.4f}: rho_eff = {rho:.3f}, f(0.5) = {float(f_closed_form(0.5, rho)):.3f} (the exact velocity vanishes at the midpoint)")


def main() -> None:
    z, s = simulate()
    hazard_padding(z); hazard_noise(z, s); hazard_baselines(s); attenuation()


if __name__ == "__main__":
    main()
