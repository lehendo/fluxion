#!/usr/bin/env python3
"""
scripts/phase3_predict.py

PRE-REGISTERED predictions of the attenuation account for the Phase 3a arms (written before any arm is trained).
For a t-sampling density w, with t CLAMPED to [0.01, 0.99], the attainable flow-loss reduction (fraction of E||delta||^2 explained, v = 0 reference) is
    E_w[f] = integral w(t) f(t) dt,
with f(t) from (a) the Gaussian reversible closed form at rho_eff = 1 - 2 f(0) = 0.695 (analysis/theory.py) and (b) the MEASURED f(t) curve of the optimal
per-channel LINEAR estimator of E[delta | z_t] (results/phase1/headline.json, 17 held-out folds), linearly interpolated. (b) is a lower bound on what a nonlinear model can reach.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy import stats
from analysis.theory import f_closed_form

LO, HI = 0.01, 0.99
ARMS = {"uniform": stats.uniform(), "beta(0.25,0.25)": stats.beta(0.25, 0.25)}


def clamp_expect(dist, ffun, t):
    """E[f(clip(t, LO, HI))] for t ~ dist: interior cell masses plus point masses at LO and HI (the CLAMP actually implemented in training/flow_matching.py)."""
    p_lo, p_hi = float(dist.cdf(LO)), float(1 - dist.cdf(HI))
    edges = np.concatenate([[LO], (t[1:] + t[:-1]) / 2, [HI]])
    m = np.diff(dist.cdf(edges))
    return float((m * ffun(t)).sum() + p_lo * ffun(np.array(LO)) + p_hi * ffun(np.array(HI)))


def trunc_expect(dist, ffun, t):
    """For the record: the same expectation if t were TRUNCATED (renormalized) to [LO, HI] instead of clamped."""
    edges = np.concatenate([[LO], (t[1:] + t[:-1]) / 2, [HI]])
    m = np.diff(dist.cdf(edges)); m /= m.sum()
    return float((m * ffun(t)).sum())


def main() -> None:
    h = json.loads(Path("results/phase1/headline.json").read_text())
    tg = np.array(h["t_grid"]); fm = np.array([h["per_t_mean_ci"][str(x)][0] for x in h["t_grid"]])
    rho = 1 - 2 * h["direct0_mean"]
    t = np.linspace(LO, HI, 4001)
    f_cf = lambda x: f_closed_form(np.asarray(x, dtype=float), rho)
    f_ms = lambda x: np.interp(np.asarray(x, dtype=float), tg, fm)
    out = {"t_range": [LO, HI], "mode": "clamp", "rho_eff": rho, "arms": {}}
    for name, d in ARMS.items():
        out["arms"][name] = {"closed_form": clamp_expect(d, f_cf, t), "measured_linear_curve": clamp_expect(d, f_ms, t),
                             "mass_at_each_clamp_bound": float(d.cdf(LO)),
                             "FOR_THE_RECORD_if_truncated_closed_form": trunc_expect(d, f_cf, t)}
    for k in ("closed_form", "measured_linear_curve"):
        out["arms"]["ratio_treatment_over_uniform_" + k] = out["arms"]["beta(0.25,0.25)"][k] / out["arms"]["uniform"][k]
    out["arms"]["ratio_if_truncated_closed_form"] = out["arms"]["beta(0.25,0.25)"]["FOR_THE_RECORD_if_truncated_closed_form"] / out["arms"]["uniform"]["FOR_THE_RECORD_if_truncated_closed_form"]
    Path("results/phase3").mkdir(parents=True, exist_ok=True)
    Path("results/phase3/predictions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
