#!/usr/bin/env python3
"""Regenerates the paper's tables from the saved results into paper_results/tables.md: the headline decomposition (Section 4.1), the baseline ratios (3.3), the padding/centering arms (7.1), and the per-protein noise, robustness and curve tables (Appendices A.1, A.3, A.7)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fluxion.paths import PAPER_RESULTS

P1, P3 = PAPER_RESULTS / "phase1", PAPER_RESULTS / "phase3"
J = lambda p: json.loads(Path(p).read_text())
ci = lambda v, d=1: f"{v[0]:.{d}f}% [{v[1]:.{d}f}, {v[2]:.{d}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def fmt(v: float) -> str:
    return f"{v:.2f}%" if abs(v) >= 0.01 else f"{v:.4f}%"


def decomposition() -> str:
    d = J(P1 / "noise_attribution_perprotein.json")["decomposition"]
    rows = [["own mean over all replicas, noisy latents (as first measured)", ci(d["V0_observed"])], ["mean from the other replicas, noisy latents", ci(d["V1_observed"])],
            ["noise removed, measured per protein (headline)", ci(d["V1_noise_free_HEADLINE"])]]
    return "## Mean-reversion headline: shrinkage gain over identity at 10 ps (Section 4.1)\n\n" + table(["step", "improvement over identity"], rows)


def baselines() -> str:
    asy, ind = J(P1 / "sweep/asymptote.json"), J(P1 / "inductive_mean.json")["_summary"]
    lags = ["10ps", "100ps", "1ns", "10ns"]
    rows = [["own mean (all replicas)"] + [f"{asy[i]['id_over_mean_measured'][0]:.3f}" for i in range(4)], ["predicted from the autocorrelation, 2(1 - rho)"] + [f"{asy[i]['id_over_mean_from_rho'][0]:.3f}" for i in range(4)],
            ["mean of the other replicas"] + [f"{ind[l]['V1']['id_over_mean']['mean']:.3f}" for l in lags], ["pooled mean of the training proteins"] + [f"{ind[l]['V2']['id_over_mean']['mean']:.3f}" for l in lags]]
    return "## MSE(identity) / MSE(mean baseline) by lag (Section 3.3)\n\n" + table(["lag", "10 ps", "100 ps", "1 ns", "10 ns"], rows)


def arms() -> str:
    spec = [("3c", "masked", "centered", "eval_3c_100.json", "dit_bs16_ema_n1"), ("T3c", "masked", "uncentered", "eval_t3c_100.json", "dit_t3c_ema_n1"),
            ("T3a", "counted", "uncentered", "eval_t3a_100.json", "dit_t3a_ema_n1"), ("T3b (seed 0)", "counted", "centered", "eval_t3b_100.json", "dit_t3b_ema_n1"), ("T3b (seed 1)", "counted", "centered", "eval_t3b_s1_100.json", "dit_t3b_ema_n1")]
    rows = [[a, pad, cen, fmt(J(P3 / f)['summary']['in_sample'][k][0])] for a, pad, cen, f, k in spec]
    return "## In-sample improvement over identity, end of the matched budget (Section 7.1)\n\n" + table(["arm", "padding", "centering", "in-sample improvement over identity"], rows)


def noise_table() -> str:
    npd, npp = J(P1 / "noise_per_protein.json"), J(P1 / "noise_attribution_perprotein.json")
    rows = []
    for p, v in sorted(npd.items(), key=lambda kv: kv[1]["n_res"]):
        r = npp["per_protein"][p]
        rows.append([p, str(v["n_res"]), f"{v['sigma2_original'] * 1e4:.2f}", f"{npp['noise_vs_step_pct']['per_protein'][p]:.1f}", f"{100 * r['nu_V0']:.1f}"])
    return "## Per-protein extraction noise (Appendix A.1)\n\n" + table(["protein", "residues", "noise variance (1e-4 per element)", "two-extraction difference, % of the 10 ps step", "noise share of the latent variance (%)"], rows)


def robustness() -> str:
    rob = J(P1 / "noise_attribution_perprotein.json")["robustness_multiplier"]
    return "## Headline against a common factor on every noise variance (Appendix A.3)\n\n" + table(["factor", "noise-free gain, 10 ps"], [[("1 (measured)" if float(k) == 1.0 else k), ci(v)] for k, v in rob.items()])


def curves() -> str:
    e100, e3c, mc, h = J(P3 / "eval_100.json"), J(P3 / "eval_3c_100.json"), J(P3 / "matched_linear_curve.json"), J(P1 / "headline.json")
    from plot_phase3 import entry_mean_curve
    T = np.array(e100["t_grid"]); idx = (0, 9, 20, 32, 40)
    tg = np.array(h["t_grid"]); fm = np.array([h["per_t_mean_ci"][str(x)][0] for x in h["t_grid"]]) * 100
    f19 = np.interp(T, tg, fm); fmatch = 100 * np.array(mc["entry_mean"])
    cols = {"uniform EMA": 100 * entry_mean_curve(e100["flow"], "uniform_ema"), "uniform raw": 100 * entry_mean_curve(e100["flow"], "uniform_raw"),
            "Beta EMA": 100 * entry_mean_curve(e100["flow"], "beta_ema"), "Beta raw": 100 * entry_mean_curve(e100["flow"], "beta_raw"), "3c EMA": 100 * entry_mean_curve(e3c["flow"], "bs16_ema")}
    rows = [[f"{T[i]:.2f}", f"{f19[i]:.2f}", f"{fmatch[i]:.2f}"] + [f"{c[i]:.2f}" for c in cols.values()] for i in idx]
    return "## Held-out R(t) against the two reference curves, % of the squared displacement explained (Appendix A.7)\n\n" + table(["t", "f(t), 19 proteins", "f(t), matched"] + list(cols), rows)


def main() -> None:
    out = "# Tables regenerated from paper_results/ (scripts/make_tables.py)\n\n" + "\n".join(f() for f in (decomposition, baselines, arms, noise_table, robustness, curves))
    (PAPER_RESULTS / "tables.md").write_text(out)
    print(f"wrote {PAPER_RESULTS / 'tables.md'}")


if __name__ == "__main__":
    main()
