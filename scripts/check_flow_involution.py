#!/usr/bin/env python3
"""Numerical check of the identity-flow proposition on Gaussian mixtures and generic odd fields (paper Section 5.2, Appendix A.6)."""
from __future__ import annotations
import argparse
import numpy as np
from scipy.integrate import solve_ivp


def make_mixture(rng, d: int, k: int, spread: float):
    comps = []
    for _ in range(k):
        a, b = rng.standard_normal(d) * spread, rng.standard_normal(d) * spread
        A = rng.standard_normal((2 * d, 2 * d)); S = A @ A.T / (2 * d) + 0.3 * np.eye(2 * d)
        comps.append((a, b, S)); P = np.block([[np.zeros((d, d)), np.eye(d)], [np.eye(d), np.zeros((d, d))]])
        comps.append((b, a, P @ S @ P.T))                                   # the swapped copy: makes the joint law swap-symmetric (reversible, equal marginals)
    return comps


def field(comps, d: int):
    I = np.eye(d)
    def v(t, x):
        logw, mus, cond = [], [], []
        for a, b, S in comps:
            S00, S01, S11 = S[:d, :d], S[:d, d:], S[d:, d:]
            m = (1 - t) * a + t * b
            Q = (1 - t) ** 2 * S00 + t * (1 - t) * (S01 + S01.T) + t ** 2 * S11
            # Cov(delta, z_t) = Cov(z1 - z0, (1-t) z0 + t z1)
            C = (1 - t) * (S01.T - S00) + t * (S11 - S01)
            Qi = np.linalg.inv(Q); r = x - m
            logw.append(-0.5 * r @ Qi @ r - 0.5 * np.linalg.slogdet(Q)[1])
            cond.append((b - a) + C @ Qi @ r)
        logw = np.array(logw); w = np.exp(logw - logw.max()); w /= w.sum()
        return sum(wi * ci for wi, ci in zip(w, cond))
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--spread", type=float, default=1.5)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for d in a.dims:
        worst_inv, worst_id = 0.0, 0.0
        for _ in range(a.trials):
            comps = make_mixture(rng, d, 2, a.spread); v = field(comps, d)
            # sample starting points from the mixture's own marginal at t = 0 (first coordinates)
            for _ in range(8):
                j = rng.integers(len(comps)); a0, b0, S = comps[j]
                x0 = rng.multivariate_normal(a0, S[:d, :d])
                s1 = solve_ivp(v, [0, 1], x0, rtol=1e-11, atol=1e-12); x1 = s1.y[:, -1]
                s2 = solve_ivp(v, [0, 1], x1, rtol=1e-11, atol=1e-12); x2 = s2.y[:, -1]
                worst_inv = max(worst_inv, np.abs(x2 - x0).max()); worst_id = max(worst_id, np.abs(x1 - x0).max())
        print(f"d = {d}: max |Phi(Phi(x)) - x| = {worst_inv:.2e}   max |Phi(x) - x| = {worst_id:.2e}   (swap-symmetric 2-component Gaussian mixtures, exact field)")


if __name__ == "__main__":
    main()
