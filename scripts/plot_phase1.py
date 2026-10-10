#!/usr/bin/env python3
"""Figures for Phase 1, drawn from the committed JSON in results/phase1/ (no cluster access needed).
  fig2  lag sweep: % over identity, % over mean, and id/mean error ratio vs lag (50 ns greyed: n_eff ~ 2/protein)
  fig3  Phase 1C: contraction alpha(lag) and full-linear-minus-shrinkage with CI
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


LAGS_PS = [10, 100, 1000, 10000, 50000]
LABELS = ["10 ps", "100 ps", "1 ns", "10 ns", "50 ns"]


def _grey_50ns(ax):
    ax.axvspan(2e4, 1e5, color="#dddddd", alpha=0.7, lw=0)
    ax.text(3.2e4, 0.97, "50 ns:\nn_eff≈2/protein,\nnot interpretable", transform=ax.get_xaxis_transform(), va="top", ha="center", fontsize=7, color="#555555")
    ax.set_xscale("log"); ax.set_xticks(LAGS_PS); ax.set_xticklabels(LABELS); ax.minorticks_off()
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def fig2():
    A = json.loads((R / "sweep" / "across_proteins.json").read_text()); asy = json.loads((R / "sweep" / "asymptote.json").read_text())
    def series(pred, key):
        rows = {r["lag_label"]: r for r in A if r["predictor"] == pred}
        return [rows[l][f"{key}_mean"] for l in LABELS], [rows[l][f"{key}_lo"] for l in LABELS], [rows[l][f"{key}_hi"] for l in LABELS]
    style = {"linear_diag": (BLUE, "-o", "per-channel linear"), "shrink": ("black", "--", "one-parameter shrinkage α·z0"), "linear_full": ("#2e9e5b", "-s", "full linear 384×384"),
             "mlp": ("#b8860b", "-^", "MLP"), "mean": (RED, "-v", "predict-the-mean"), "DiT[in-sample]_raw_n8": (GREY, "-x", "trained DiT (in-sample)")}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title, asy_line in ((axs[0], "id", "improvement over IDENTITY (%)", 50.0), (axs[1], "mean", "improvement over MEAN (%)", 0.0)):
        for pred, (c, fmt, lab) in style.items():
            if key == "mean" and pred == "mean": continue
            m, lo, hi = series(pred, key)
            if key == "mean" and pred.startswith("DiT"):
                lab = "identity = trained DiT (in-sample)"
            ax.errorbar(LAGS_PS, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt=fmt, color=c, ms=4, lw=1.4, capsize=2, label=lab)
        ax.axhline(asy_line, color="black", lw=0.8, ls=":"); ax.axhline(0, color="#999999", lw=0.6)
        ax.text(12, asy_line + 1.5, f"asymptote {asy_line:.0f}%", fontsize=7.5); ax.set_title(title, loc="left", fontsize=10.5); _grey_50ns(ax)
    axs[0].legend(fontsize=7.2, frameon=False, loc="center left"); axs[0].set_ylim(-70, 60); axs[1].set_ylim(-125, 60)
    m = [r["id_over_mean_measured"] for r in asy]; f = [r["id_over_mean_from_rho"] for r in asy]
    ax = axs[2]
    ax.errorbar(LAGS_PS, [x[0] for x in m], yerr=[[x[0] - x[1] for x in m], [x[2] - x[0] for x in m]], fmt="-o", color=BLUE, capsize=2, ms=4, label="measured identity error / mean error")
    ax.plot(LAGS_PS, [x[0] for x in f], "--s", color="black", ms=3.5, lw=1, label="2(1−ρ), ρ from the structure function")
    ax.axhline(2.0, color="black", ls=":", lw=0.8); ax.text(12, 2.04, "asymptote 2", fontsize=7.5); ax.axhline(1.0, color="#999999", lw=0.6)
    ax.set_title("which baseline wins: crossover at ≈1 ns", loc="left", fontsize=10.5); ax.legend(fontsize=7.2, frameon=False, loc="upper left"); _grey_50ns(ax)
    for a in axs: a.set_xlabel("lag")
    fig.tight_layout(); fig.savefig(OUT / "fig2_lag_sweep.png", dpi=200); fig.savefig(OUT / "fig2_lag_sweep.pdf"); print("wrote fig2")


def fig3():
    O = json.loads((R / "sweep" / "onec.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axs[0]; ax.plot(LAGS_PS, [o["alpha_mean"] for o in O], "-o", color=BLUE); ax.set_title("fitted shrinkage α(lag) (≈ autocorrelation)", loc="left", fontsize=10.5)
    ax.set_ylim(0, 1); _grey_50ns(ax)
    ax = axs[1]
    for k, c, lab in (("linear_full", "#2e9e5b", "full linear 384×384"), ("mlp", "#b8860b", "MLP"), ("linear_diag", BLUE, "per-channel linear")):
        v = [o[f"{k}.pct_over_shrink"] for o in O]
        ax.errorbar(LAGS_PS, [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], fmt="-o", color=c, capsize=2, ms=4, label=lab)
    ax.axhline(0, color="black", lw=0.8); ax.set_title("probe minus shrinkage α·z0 (% of error); ≥0 needed to beat it", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False); _grey_50ns(ax)
    for a in axs: a.set_xlabel("lag")
    fig.tight_layout(); fig.savefig(OUT / "fig3_phase1c.png", dpi=200); fig.savefig(OUT / "fig3_phase1c.pdf"); print("wrote fig3")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
