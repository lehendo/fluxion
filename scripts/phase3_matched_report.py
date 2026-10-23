#!/usr/bin/env python3
"""Reading of the arms against the matched reference (paper Section 6.1, Appendix A.7)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

R = Path("paper_results/phase3")


def arm_masses(t_cfg, grid):
    lo, hi = t_cfg["clip"]; d = stats.uniform() if t_cfg["dist"] == "uniform" else stats.beta(t_cfg["beta_a"], t_cfg["beta_b"])
    edges = np.concatenate([[lo], (grid[1:] + grid[:-1]) / 2, [hi]]); m = np.diff(d.cdf(edges)); m[0] += d.cdf(lo); m[-1] += 1 - d.cdf(hi)
    return m / m.sum()


def entry_mean_curve(d, key):
    by = {}
    for k, v in d["flow"].items():
        if k.startswith("held_out:") and key in v:
            by.setdefault(k.split(":")[1].split("_")[0], []).append(np.array(v[key], float))
    return np.mean([np.mean(v, 0) for v in by.values()], 0)


def main() -> None:
    mc = json.load(open(R / "matched_linear_curve.json")); f = np.array(mc["entry_mean"]); tg = np.array(mc["t_grid"])
    h = json.load(open("paper_results/phase1/headline.json")); pm = h["per_t_mean_ci"]; ht = np.array([float(k) for k in pm]); fv = np.array([pm[k][0] for k in pm]); o = np.argsort(ht); f19 = np.interp(tg, ht[o], fv[o])
    print(f"matched f(0.01) {100 * f[0]:.2f}%  f(0.5) {100 * f[20]:.2f}%  f(0.99) {100 * f[-1]:.2f}%   vs 19-protein curve {100 * f19[0]:.2f} / {100 * f19[20]:.2f} / {100 * f19[-1]:.2f}")
    runs = [("uniform s0", "eval_100", "uniform"), ("uniform s1", "eval_s1_100", "uniform"), ("uniform s2", "eval_s2_100", "uniform"), ("beta s0", "eval_100", "beta"), ("3c (bs16)", "eval_3c_100", "bs16")]
    print(f"{'run':12s} {'weights':5s} | max|R-f_matched| | mean(R-f_matched) | mean(R-f_19) | attained  predicted_matched (ratio) | predicted_19 | reading")
    for name, fn, arm in runs:
        d = json.load(open(R / f"{fn}.json")); m = arm_masses(d["t_cfg"][arm], tg)
        for wt in ("ema", "raw"):
            c = entry_mean_curve(d, f"{arm}_{wt}"); att = float((m * c).sum()); pred_m = float((m * f).sum()); pred_19 = float((m * f19).sum())
            D = 100 * np.abs(c - f).max(); ms = 100 * (c - f).mean(); m19 = 100 * (c - f19).mean()
            reading = "ON" if D <= 0.6 else ("BELOW" if ms < -0.6 else ("ABOVE" if ms > 0.6 else "OFF (mixed)"))
            print(f"{name:12s} {wt:5s} | {D:6.2f}           | {ms:+6.2f}            | {m19:+6.2f}       | {att:.4f}   {pred_m:.4f} ({att / pred_m:.2f})        | {pred_19:.4f}       | {reading}")


if __name__ == "__main__":
    main()
