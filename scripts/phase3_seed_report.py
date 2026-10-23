#!/usr/bin/env python3
"""Registered reading of the seed replication (paper Section 6.1)."""
from __future__ import annotations
from fluxion.paths import PAPER_RESULTS
import json, sys
from pathlib import Path
import numpy as np

R = PAPER_RESULTS
PRED = 0.054117548092600554
D_MAX = 0.6


def f_curve(tg):
    h = json.load(open(R / "phase1/headline.json")); pm = h["per_t_mean_ci"]
    ht = np.array([float(k) for k in pm]); f = np.array([pm[k][0] for k in pm]); o = np.argsort(ht)
    return np.interp(tg, ht[o], f[o])


def entry_mean_curve(d, key):
    by = {}
    for k, v in d["flow"].items():
        if k.startswith("held_out:"):
            by.setdefault(k.split(":")[1].split("_")[0], []).append(np.array(v[key], float))
    return np.mean([np.mean(v, 0) for v in by.values()], 0)


def classify(rows) -> str:
    if any((not np.isfinite(r["att_ema"])) or r["att_ema"] < 0.5 * PRED for r in rows):
        return "CONDITIONAL"
    return "ROBUST" if all(r["D_ema"] <= D_MAX and r["D_raw"] <= D_MAX for r in rows) else "OTHER"


def main() -> None:
    files = {0: R / "phase3/eval_100.json", 1: R / "phase3/eval_s1_100.json", 2: R / "phase3/eval_s2_100.json"}
    rows = []
    print("seed | attained held-out flow reduction EMA (x prediction) / raw | D_EMA, D_raw (points) | one-step held-out EMA / raw | mean signed R - f (EMA, raw)")
    for s, f in files.items():
        d = json.load(open(f)); tg = np.array(d["t_grid"]); fi = f_curve(tg)
        att = d["flow_summary"]["held_out"]
        ae, ar = att["uniform_ema"]["attained_reduction"][0], att["uniform_raw"]["attained_reduction"][0]
        ce, cr = entry_mean_curve(d, "uniform_ema"), entry_mean_curve(d, "uniform_raw")
        n1 = d["summary"]["held_out"]
        row = {"seed": s, "att_ema": ae, "att_raw": ar, "D_ema": 100 * np.abs(ce - fi).max(), "D_raw": 100 * np.abs(cr - fi).max()}
        rows.append(row)
        print(f"  {s} | {ae:.4f} ({ae / PRED:.2f}) / {ar:.4f} | {row['D_ema']:.2f}, {row['D_raw']:.2f} | {n1['dit_uniform_ema_n1'][0]:+.2f} / {n1['dit_uniform_raw_n1'][0]:+.2f} | {100 * (ce - fi).mean():+.2f}, {100 * (cr - fi).mean():+.2f}")
    print("READING:", classify(rows))
    print(f"three-seed spread: attained EMA {min(r['att_ema'] for r in rows):.4f}-{max(r['att_ema'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
