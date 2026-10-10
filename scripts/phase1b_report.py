#!/usr/bin/env python3
"""
scripts/phase1b_report.py

Paired Chai-1 minus Cartesian report for B1, from results/phase1/b1/<protein>.json (fetched from the cluster).
Statistic (pre-registered, EXPERIMENT_LOG.md 2026-10-10): per protein, difference in % improvement over identity
(Chai-1 PCA minus aligned-Calpha PCA, same k = min(3 n_res, 384), leave-one-replica-out); proteins of one PDB entry
(1bxy_A/B) are averaged to ONE value; 95% CI = percentile bootstrap over entries. With few entries a sign count and the
exact one-sided sign-flip p-value are printed as well. ALL six predictors are reported (no best-of selection).
Evaluated on TWO protein sets and reported side by side: ORIGINAL (8 proteins passing the pre-registered criterion iii) and REVISED
(13 proteins passing the post hoc within-replica criterion iii'); see EXPERIMENT_LOG.md 2026-10-11.
CLAIM LANGUAGE: B1 tests LEARNABILITY, not information (Chai-1 latents are a deterministic function of coordinates + sequence). The statement it can support is
'frozen Chai-1 trunk latents offer no learnability advantage over matched-dimension aligned-C-alpha PCA for linear or MLP probes at lags >= 100 ps'.
The claim rests on the identity-relative rows linear_diag and shrink; linear_full and mlp overfit at ~2000 pairs/protein in both representations.
A Chai-1 advantage would require the CI of the paired difference (linear_diag, shrink) to lie above 0 at lags <= 10 ns.   50 ns: not interpretable (n_eff ~ 2).
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np

B1 = Path("results/phase1/b1")
LAGS = ["100ps", "1ns", "10ns", "50ns"]
PREDS = ["linear_diag", "linear_full", "shrink", "mlp", "mean"]


def entry(p: str) -> str:
    return p.split("_")[0]


def boot_ci(x, n=20000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def signflip_p(x):
    """exact one-sided p that mean(x) >= observed under random sign flips (H0: symmetric about 0)."""
    x = np.asarray(x); obs = x.mean(); tot = 0; ge = 0
    for s in itertools.product([1, -1], repeat=len(x)):
        tot += 1; ge += (np.dot(s, x) / len(x)) >= obs - 1e-12
    return ge / tot


SETS = {"original8": ["1ab1_A", "1ah7_A", "1aol_A", "1b2s_F", "1bq8_A", "1btk_A", "1bxy_A", "1bxy_B"],
        "revised13": ["1ab1_A", "1af7_A", "1ah7_A", "1aol_A", "1b25_D", "1b2s_E", "1b2s_F", "1b8d_K", "1bkp_A", "1bq8_A", "1btk_A", "1bxy_A", "1bxy_B"]}


def main() -> None:
    allres = {json.loads(f.read_text())["protein"]: json.loads(f.read_text()) for f in sorted(B1.glob("*.json")) if f.name != "b1_report.json"}
    final = {}
    for setname, prots in SETS.items():
        missing = [p for p in prots if p not in allres]
        if missing:
            print(f"\n##### set {setname}: SKIPPED, missing results for {missing}"); continue
        print(f"\n##################### set {setname}")
        final[setname] = report({p: allres[p] for p in prots})
    (B1 / "b1_report.json").write_text(json.dumps(final, indent=1))


def report(R):
    ents = sorted({entry(p) for p in R})
    print(f"proteins: {sorted(R)}  ->  {len(ents)} PDB entries")
    out = {}
    for lag in LAGS:
        print(f"\n=== lag {lag}{'  (NOT interpretable, n_eff~2)' if lag == '50ns' else ''} ===")
        print(f"{'predictor':12s} {'Chai % vs id':>13s} {'Cart % vs id':>13s} | {'paired diff Chai-Cart':>34s} | entries Chai>Cart  sign-flip p")
        for pr in PREDS + ["id/mean"]:
            ch, ca = {}, {}
            for p, r in R.items():
                c, g = r["lags"][lag]["chai"], r["lags"][lag]["cart"]
                if pr == "id/mean":
                    vc, vg = c["identity"]["mse"] / c["mean"]["mse"], g["identity"]["mse"] / g["mean"]["mse"]
                else:
                    vc, vg = c[pr]["pct_vs_identity"], g[pr]["pct_vs_identity"]
                ch.setdefault(entry(p), []).append(vc); ca.setdefault(entry(p), []).append(vg)
            c_e = np.array([np.mean(ch[e]) for e in ents]); g_e = np.array([np.mean(ca[e]) for e in ents]); d = c_e - g_e
            m, lo, hi = boot_ci(d)
            star = "*" if lo > 0 else ("v" if hi < 0 else " ")
            print(f"{pr:12s} {c_e.mean():13.2f} {g_e.mean():13.2f} | {m:+8.2f} [{lo:+8.2f},{hi:+8.2f}] {star:1s}            | {int((d > 0).sum())}/{len(d)}              {signflip_p(d):.3f}")
            out.setdefault(lag, {})[pr] = {"chai": float(c_e.mean()), "cart": float(g_e.mean()), "diff": [m, lo, hi], "n_chai_gt": int((d > 0).sum()), "n": len(d), "per_entry_diff": dict(zip(ents, map(float, d)))}
    print("\n* = CI above 0 (Chai-1 better), v = CI below 0 (Cartesian better), blank = CI includes 0")
    return out


if __name__ == "__main__":
    main()
