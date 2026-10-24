#!/usr/bin/env python3
"""Figures for the flow-model experiments (paper Section 6, Appendix A.7)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fluxion.analysis.theory import f_closed_form

R = Path("paper_results/phase3")
P1 = Path("paper_results/phase1")
OUT = R / "figs"
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"uniform": "#2a78d6", "beta": "#eb6834", "bs16": "#1baf7a"}
MARK = {"uniform": "o", "beta": "s", "bs16": "^"}
LABEL = {"uniform": "uniform t (batch 256)", "beta": "Beta(0.25, 0.25) t (batch 256)", "bs16": "uniform t, batch 16 (3c)"}
MILESTONES = ["025", "050", "075", "100"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def entry_of(protein: str) -> str:
    return protein.split("_")[0]


def entry_mean_curve(flow: dict, key: str, group: str = "held_out") -> np.ndarray:
    """Mean over PDB entries of the per-protein R(t) curve `key` (proteins of one entry are averaged first)."""
    ents: dict[str, list] = {}
    for k, v in flow.items():
        g, p = k.split(":")
        if g == group and key in v:
            ents.setdefault(entry_of(p), []).append(v[key])
    return np.mean([np.mean(c, axis=0) for c in ents.values()], axis=0)


def entry_values(res: dict, method: str, group: str = "held_out") -> dict[str, float]:
    ents: dict[str, list] = {}
    for k, v in res["per_protein"].items():
        g, p = k.split(":")
        if g == group and method in v["pct_vs_identity"]:
            ents.setdefault(entry_of(p), []).append(v["pct_vs_identity"][method])
    return {e: float(np.mean(x)) for e, x in ents.items()}


def load(name: str) -> dict | None:
    f = R / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def direct_label(ax, x, y, text, color, dx=0.0, dy=0.0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=8, **kw)


def fig6():
    r_u, r_b, r_c = load("eval_100"), load("eval_3c_100"), None
    if r_u is None:
        return
    r_c = load("eval_3c_100")
    T = np.array(r_u["t_grid"])
    h = json.loads((P1 / "headline.json").read_text())
    tg = np.array(h["t_grid"]); fm = np.array([h["per_t_mean_ci"][str(x)][0] for x in h["t_grid"]]) * 100
    rho = 1 - 2 * h["direct0_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tg[1:-1], fm[1:-1], "-", color=INK, lw=2.0)
    for arm, res, key in (("uniform", r_u, "uniform_ema"), ("beta", r_u, "beta_ema"), ("bs16", r_c, "bs16_ema")):
        if res is None:
            continue
        c = 100 * entry_mean_curve(res["flow"], key)
        ax.plot(T, c, "-", color=COL[arm], lw=1.6, marker=MARK[arm], ms=4, markevery=5)
    ax.plot(np.linspace(0.01, 0.99, 400), 100 * f_closed_form(np.linspace(0.01, 0.99, 400), rho), "--", color="#ffffff", lw=0.9, zorder=5)
    ax.set_xlabel("interpolation time t"); ax.set_ylabel("held-out flow-loss reduction (% of ‖z1 − z0‖²)")
    ax.set_title("Held-out R(t): batch-256 models just below the linear-optimal curve; batch 16 lower", loc="left", fontsize=9.5)
    direct_label(ax, 0.02, 16.0, "19-protein linear-optimal f(t) (black; white dashes: closed form)", INK, dx=4, dy=0)
    direct_label(ax, 0.30, 9.4, "uniform and β (batch 256): within about a point of f(t)", COL["uniform"], dx=0, dy=0)
    direct_label(ax, 0.36, 5.6, "3c (batch 16): 1–3 points lower", COL["bs16"], dx=0, dy=0)
    ax.set_xlim(0, 1); ax.set_ylim(-1, 17); style(ax)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "fig6_flow_curves.png", dpi=200); fig.savefig(OUT / "fig6_flow_curves.pdf"); plt.close(fig)
    print("wrote fig6")


def fig7():
    evals = {m: load(f"eval_{m}") for m in MILESTONES}
    c3 = {m: load(f"eval_3c_{m}") for m in MILESTONES}
    if any(v is None for v in evals.values()):
        return
    pred = json.loads((R / "predictions.json").read_text())["arms"]
    xs = [25, 50, 75, 100]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    for col, w in enumerate(("raw", "ema")):
        a1, a2 = axs[0, col], axs[1, col]
        for arm in ("uniform", "beta", "bs16"):
            ys, zs = [], []
            for m in MILESTONES:
                src = c3[m] if arm == "bs16" else evals[m]
                if src is None:
                    ys.append(np.nan); zs.append(np.nan); continue
                ys.append(100 * src["flow_summary"]["held_out"][f"{arm}_{w}"]["attained_reduction"][0])
                zs.append(np.mean(list(entry_values(src, f"dit_{arm}_{w}_n1").values())))
            a1.plot(xs, ys, "-", color=COL[arm], lw=1.8, marker=MARK[arm], ms=5)
            a2.plot(xs, zs, "-", color=COL[arm], lw=1.8, marker=MARK[arm], ms=5)
            direct_label(a1, 100, ys[-1], {"uniform": "uniform", "beta": "beta", "bs16": "3c"}[arm], COL[arm], dx=5, dy=-3)
        a1.axhline(100 * pred["uniform"]["closed_form"], color=COL["uniform"], ls=":", lw=1)
        a1.axhline(100 * pred["beta(0.25,0.25)"]["closed_form"], color=COL["beta"], ls=":", lw=1)
        shr = np.mean(list(entry_values(evals["100"], "shrink").values()))
        a2.axhline(shr, color=INK, ls="--", lw=1.1); direct_label(a2, 26, shr, "α·z0 shrinkage ceiling", INK, dx=0, dy=4)
        a1.set_title(f"{'raw' if w == 'raw' else 'EMA'} weights", loc="left", fontsize=10)
        a2.set_xlabel("training progress (% of the run)")
        a1.set_ylabel("attained flow-loss reduction (%)"); a2.set_ylabel("one-step improvement over identity (%)")
        a1.set_ylim(-12, 13); a2.set_ylim(-8, 20)
        style(a1); style(a2); a2.set_xticks(xs)
    axs[0, 0].text(0.02, 0.97, "dotted: closed-form prediction for each arm", transform=axs[0, 0].transAxes, fontsize=7.5, color=MUTED, va="top")
    fig.suptitle("Held-out milestones: raw weights plateau by mid-run, the EMA lags, and 3c declines after an early plateau", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig7_trajectory.png", dpi=200); fig.savefig(OUT / "fig7_trajectory.pdf"); plt.close(fig)
    print("wrote fig7")


def fig8():
    r_u, r_c = load("eval_100"), load("eval_3c_100")
    if r_u is None:
        return
    ks = [1, 2, 4, 8]
    rho = 1 - 2 * json.loads((P1 / "headline.json").read_text())["direct0_mean"]
    c = lambda t: -(1 - rho) * (1 - 2 * t) / (1 - 2 * (1 - rho) * t * (1 - t))

    def pred(k):
        a = 1.0
        for j in range(k):
            a *= 1 + c(max(j / k, 0.01)) / k
        return 100 * (1 - (1 + a * a - 2 * a * rho) / (2 - 2 * rho))

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, w in zip(axs, ("raw", "ema")):
        for arm, res in (("uniform", r_u), ("beta", r_u), ("bs16", r_c)):
            if res is None:
                continue
            ys = [np.mean(list(entry_values(res, f"dit_{arm}_{w}_n{k}").values())) for k in ks]
            ax.plot(ks, ys, "-", color=COL[arm], lw=1.8, marker=MARK[arm], ms=5)
            off = {"raw": {"uniform": 7, "beta": -10, "bs16": -2}, "ema": {"uniform": -3, "beta": -12, "bs16": 8}}[w]
            direct_label(ax, 8, ys[-1], {"uniform": "uniform", "beta": "beta", "bs16": "3c"}[arm], COL[arm], dx=7, dy=off[arm])
        ax.plot(ks, [pred(k) for k in ks], "--", color=INK, lw=1.3)
        direct_label(ax, 1.6, 1.6, "dashed: closed form for the exact optimal field", INK)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
        ax.set_xlabel("Euler integration steps from t = 0.01"); ax.set_title(f"{w if w == 'raw' else 'EMA'} weights", loc="left", fontsize=10)
        ax.set_xlim(0.9, 12); style(ax)
    axs[0].set_ylabel("held-out improvement over identity (%)")
    fig.suptitle("More ODE steps move the forecast toward the identity map, as the closed form predicts (magnitudes run 1–2 points higher)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "fig8_steps.png", dpi=200); fig.savefig(OUT / "fig8_steps.pdf"); plt.close(fig)
    print("wrote fig8")


def fig9():
    import pandas as pd
    d = R / "diag"
    if not (d / "A_s0_loss_log.csv").exists():
        return
    names = {"A": "A  uniform t, peak LR", "B": "B  Beta, clamp, peak LR", "C": "C  Beta, truncate, peak LR", "D": "D  Beta, clamp, half LR"}
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True)
    for ax, c in zip(axs, "ABCD"):
        for s, ls in ((0, "-"), (1, "--")):
            df = pd.read_csv(d / f"{c}_s{s}_loss_log.csv")
            y = df.flow_reduction_pct.clip(lower=-30)
            ax.plot(df.step, y, ls, color="#2a78d6", lw=1.4, label=f"seed {s}")
            low = df.flow_reduction_pct.min()
            if low < -30:
                i = df.flow_reduction_pct.idxmin()
                ax.annotate(f"seed {s}: {low:.0f}%", (df.step[i], -30), xytext=((8 if df.step[i] < 330 else -78), 14 if s == 0 else 38), textcoords="offset points", fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.axvline(350, color=MUTED, lw=0.8, ls=":")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(names[c], loc="left", fontsize=9.5); ax.set_xlabel("optimizer step"); ax.set_xlim(0, 600); ax.set_ylim(-31, 15)
        ax.legend(frameon=False, fontsize=8, loc="upper left"); style(ax)
    axs[0].set_ylabel("live flow-loss reduction (%, training batches)")
    fig.suptitle("Halving the peak LR removed every excursion (dotted line: end of warmup; y clipped at −30%)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig9_diagnostic.png", dpi=200); fig.savefig(OUT / "fig9_diagnostic.pdf"); plt.close(fig)
    print("wrote fig9")


def fig10():
    r_u, r_c = load("eval_100"), load("eval_3c_100")
    if r_u is None:
        return
    series = [("shrinkage α·z0", entry_values(r_u, "shrink"), INK, None)]
    series += [(LABEL["uniform"], entry_values(r_u, "dit_uniform_ema_n1"), COL["uniform"], "uniform"), (LABEL["beta"], entry_values(r_u, "dit_beta_ema_n1"), COL["beta"], "beta")]
    if r_c is not None:
        series.append((LABEL["bs16"], entry_values(r_c, "dit_bs16_ema_n1"), COL["bs16"], "bs16"))
    ents = sorted(series[0][1])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    w = 0.8 / len(series); x = np.arange(len(ents))
    for i, (lab, vals, col, key) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, [vals[e] for e in ents], width=w * 0.88, color=col, label=lab, hatch=None if key != "bs16" else "////", edgecolor="white" if key != "bs16" else col, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ents); ax.set_ylabel("one-step improvement over identity (%)")
    ax.set_title("Held-out, one step (EMA weights, 100%): within ~1 point of shrinkage except 1bq8_A", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2); style(ax)
    fig.tight_layout(); fig.savefig(OUT / "fig10_entries.png", dpi=200); fig.savefig(OUT / "fig10_entries.pdf"); plt.close(fig)
    print("wrote fig10")


def fig11():
    """Seed replication of the uniform batch-256 arm: held-out R(t) - f(t) (EMA and raw) at 100%, and the held-out attained flow reduction across milestones, for seeds 0, 1, 2."""
    files = {0: ("eval", "#2a78d6"), 1: ("eval_s1", "#1baf7a"), 2: ("eval_s2", "#eb6834")}
    h = json.loads((P1 / "headline.json").read_text()); pm = h["per_t_mean_ci"]
    ht = np.array([float(k) for k in pm]); fv = np.array([pm[k][0] for k in pm]); o = np.argsort(ht); ht, fv = ht[o], fv[o]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for s, (stem, col) in files.items():
        d100 = load(f"{stem}_100")
        if d100 is None:
            continue
        tg = np.array(d100["t_grid"]); fi = np.interp(tg, ht, fv)
        for ax, key, ttl in ((axes[0], "uniform_ema", "EMA weights"), (axes[1], "uniform_raw", "raw weights")):
            ax.plot(tg, 100 * (entry_mean_curve(d100["flow"], key) - fi), color=col, lw=1.6, label=f"seed {s}")
        att = [load(f"{stem}_{m}")["flow_summary"]["held_out"]["uniform_ema"]["attained_reduction"][0] for m in MILESTONES]
        axes[2].plot([25, 50, 75, 100], att, color=col, marker="o", lw=1.6, label=f"seed {s}")
    for ax, ttl in ((axes[0], "held-out R(t) - f(t), EMA, 100%"), (axes[1], "held-out R(t) - f(t), raw, 100%")):
        ax.axhspan(-0.6, 0.6, color="#e6e5e0", alpha=0.7, lw=0); ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlabel("t", color=MUTED); ax.set_ylabel("points of explained |delta|^2", color=MUTED); ax.set_title(ttl, fontsize=9, color=INK, loc="left"); style(ax); ax.legend(frameon=False, fontsize=8)
    axes[2].axhline(0.0541, color=INK, lw=0.6, ls="--"); axes[2].text(99, 0.0605, "registered prediction 0.0541", fontsize=7.5, color=MUTED, ha="right"); axes[2].set_ylim(-0.82, 0.09)
    axes[2].set_xlabel("% of the 6,400-step budget", color=MUTED); axes[2].set_ylabel("held-out attained flow-loss reduction (EMA)", color=MUTED)
    axes[2].set_title("milestones: seed 2 collapses near step 2,560", fontsize=9, color=INK, loc="left"); style(axes[2]); axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig11_seeds.png", dpi=200); fig.savefig(OUT / "fig11_seeds.pdf"); plt.close(fig)
    print("wrote fig11")


def fig12():
    """T3: in-sample one-step improvement over identity (primary metric) across milestones for 3c (neither factor), T3c (uncentered), T3a (both) and T3b (padding counted; two seeds)."""
    arms = [("3c: masked, centered (evaluated at 25% and 100% only)", "eval_3c", "dit_bs16", "#1baf7a", "^", "--"), ("T3c: masked, uncentered", "eval_t3c", "dit_t3c", "#7a5ccf", "D", "-"),
            ("T3a: counted, uncentered", "eval_t3a", "dit_t3a", "#eb6834", "s", "-"), ("T3b: counted, centered, seed 0", "eval_t3b", "dit_t3b", "#2a78d6", "o", "-"), ("T3b: counted, centered, second seed", "eval_t3b_s1", "dit_t3b", "#0b0b0b", "x", "-")]
    pos = {"025": 25, "050": 50, "075": 75, "100": 100}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, wt in zip(axes, ("ema", "raw")):
        for lab, stem, key, col, mk, ls in arms:
            pts = [(pos[m], load(f"{stem}_{m}")["summary"]["in_sample"][f"{key}_{wt}_n1"][0]) for m in MILESTONES if load(f"{stem}_{m}") and "in_sample" in load(f"{stem}_{m}")["summary"]]
            ax.plot(*zip(*pts), color=col, marker=mk, lw=1.5, ls=ls, label=lab)
        ax.axhline(14.09, color=INK, lw=0.6, ls=":"); ax.text(99, 14.5, "shrinkage 14.09", fontsize=7.5, color=MUTED, ha="right")
        ax.axhspan(5, 12, color="#e6e5e0", alpha=0.5, lw=0); ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlabel("% of the matched 1.638M-sample budget", color=MUTED); ax.set_title(f"in-sample one-step improvement over identity, {wt.upper()} weights", fontsize=9, color=INK, loc="left"); style(ax)
    axes[0].set_ylabel("% over identity (grey: registered 5-12% band)", color=MUTED, fontsize=8)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(OUT / "fig12_t3.png", dpi=200); fig.savefig(OUT / "fig12_t3.pdf"); plt.close(fig)
    print("wrote fig12")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig6, fig7, fig8, fig9, fig10, fig11, fig12):
        f()


if __name__ == "__main__":
    main()
