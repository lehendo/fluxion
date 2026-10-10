#!/usr/bin/env python3
"""Figures for Phase 1, drawn from the committed JSON in results/phase1/ (no cluster access needed).
  fig1  attenuation: fraction of ||u||^2 explained by the optimal linear E[delta|z_t] vs t (held-out entries),
        closed-form Gaussian prediction, direct-regression references, uniform-t average.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.theory import f_closed_form, rho_from_f0

R = Path("results/phase1"); OUT = R / "figs"; OUT.mkdir(parents=True, exist_ok=True)
GREY, BLUE, RED = "#8a8f98", "#1f6fd0", "#c0392b"


def fig1():
    h = json.loads((R / "headline.json").read_text())
    t = np.array(h["t_grid"]); m = np.array([h["per_t_mean_ci"][str(x)][0] for x in h["t_grid"]]) * 100
    lo = np.array([h["per_t_mean_ci"][str(x)][1] for x in h["t_grid"]]) * 100
    hi = np.array([h["per_t_mean_ci"][str(x)][2] for x in h["t_grid"]]) * 100
    d0, favg, d01 = h["direct0_mean"] * 100, h["f_avg_mean"] * 100, h["direct01_mean"] * 100
    rho = rho_from_f0(h["direct0_mean"]); tt = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(t, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(t, m, "-o", color=BLUE, ms=3.5, lw=1.8, label="optimal linear E[δ | z_t], held-out entries")
    ax.plot(tt, 100 * f_closed_form(tt, rho), "--", color="black", lw=1.1,
            label=f"Gaussian reversible closed form, ρ = {rho:.2f} from f(0) alone")
    ax.axhline(d01, color=GREY, ls=":", lw=1.4); ax.text(0.5, d01 + 0.6, f"direct regression on (z0, z_prev): {d01:.1f}% (input the model never sees)", ha="center", color=GREY, fontsize=8.5)
    ax.axhline(d0, color=GREY, ls="-", lw=0.8); ax.text(0.5, d0 + 0.6, f"direct regression on z0 only = f(t = 0): {d0:.1f}%", ha="center", color=GREY, fontsize=8.5)
    ax.axhline(favg, color=RED, lw=1.6); ax.text(0.5, favg + 0.6, f"uniform-t average f̄ = {favg:.1f}%  ({h['ratio_direct0_over_favg']:.2f}× below f(0))", ha="center", color=RED, fontsize=8.8)
    ax.set_xlabel("interpolation time t"); ax.set_ylabel("fraction of ‖z1 − z0‖² explained (%)")
    ax.set_title("Most draws of t land where there is little to learn", loc="left", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(-1, 30); ax.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.14), ncol=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig1_attenuation.png", dpi=200); fig.savefig(OUT / "fig1_attenuation.pdf")
    print("wrote", OUT / "fig1_attenuation.png")


if __name__ == "__main__":
    fig1()
