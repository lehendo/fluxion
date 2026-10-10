"""
analysis/theory.py

Closed forms for the conditional-flow-matching regression target between the two ends of a stationary,
time-reversible pair (z0, z1) (see theory.md). Gaussian case with correlation rho (unit variance):

    delta = z1 - z0,  z_t = (1-t) z0 + t z1
    Cov(delta, z_t) = -(1-rho)(1-2t),  Var(z_t) = 1 - 2(1-rho) t (1-t),  Var(delta) = 2(1-rho)
    fraction of Var(delta) explained by the optimal E[delta | z_t]:
        f(t) = f0 (1-2t)^2 / (1 - 2(1-rho) t (1-t)),        f0 = f(0) = (1-rho)/2
    uniform-t average:  f_bar = f0 * R(rho),   R(rho) = int_0^1 (1-2t)^2 / (1 - 2(1-rho) t (1-t)) dt
R decreases from ~0.4292 (rho = 0) to 1/3 (rho -> 1): the random-t objective can attain only 33-43% of the signal
available at the endpoint, i.e. the attenuation factor 1/R lies in [2.33, 3.0] for ANY stationary reversible Gaussian.
For per-channel correlations rho_c the same bounds hold (f_bar/f0 is a weighted average of R(rho_c)).
"""
from __future__ import annotations

import numpy as np


def f_closed_form(t, rho: float):
    t = np.asarray(t, dtype=float)
    f0 = (1.0 - rho) / 2.0
    return f0 * (1.0 - 2.0 * t) ** 2 / (1.0 - 2.0 * (1.0 - rho) * t * (1.0 - t))


def attenuation_R(rho: float, n: int = 200001) -> float:
    t = np.linspace(0.0, 1.0, n)
    y = (1.0 - 2.0 * t) ** 2 / (1.0 - 2.0 * (1.0 - rho) * t * (1.0 - t))
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0)


def rho_from_f0(f0: float) -> float:
    """Effective correlation implied by the endpoint explained fraction (f0 = (1-rho)/2)."""
    return 1.0 - 2.0 * f0
