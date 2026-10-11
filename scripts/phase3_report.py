#!/usr/bin/env python3
"""
scripts/phase3_report.py : tables from the pre-registered Phase 3a evaluations (results/phase3/eval_<m>.json).
For each available milestone: (1) PRIMARY, held-out attained flow-loss reduction per arm vs the closed-form prediction, with the treatment/control ratio; (2) the alpha*z0 shrinkage ceiling on the SAME held-out entries, per-entry
delta (model - shrinkage, in percentage points of improvement over identity), mean, bootstrap CI over the held-out entries, and the number of entries with a positive delta; (3) downstream % over identity.
Held-out entries: proteins of one PDB entry are averaged first (1bxy_A/B -> 1bxy).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

D = Path("results/phase3")
PRED = json.loads((D / "predictions.json").read_text())["arms"]
PRED_FLOW = {"uniform": PRED["uniform"]["closed_form"], "beta": PRED["beta(0.25,0.25)"]["closed_form"]}


def entry(p): return p.split("_")[0]


def boot(x, n=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_entry(res, group):
    ents = {}
    for k, v in res["per_protein"].items():
        g, p = k.split(":")
        if g == group:
            ents.setdefault(entry(p), []).append(v["pct_vs_identity"])
    methods = next(iter(ents.values()))[0].keys()
    return {e: {m: float(np.mean([r[m] for r in rs])) for m in methods} for e, rs in ents.items()}


def report(m):
    f = D / f"eval_{m}.json"
    if not f.exists():
        return None
    res = json.loads(f.read_text())
    out = {"milestone": m, "steps": res["arms_steps"], "alpha": res["alpha"]}
    lines = [f"\n######## milestone {m}%  (steps {res['arms_steps']}, alpha = {res['alpha']:.4f}) ########"]
    fs = res["flow_summary"]["held_out"]
    lines.append("PRIMARY (held-out attained flow-loss reduction, fraction of E||delta||^2; each arm integrated against its OWN clamped t distribution):")
    for k, v in fs.items():
        if "attained_reduction" in v:
            arm = k.split("_")[0]
            lines.append(f"  {k:14s} {v['attained_reduction'][0]:.4f} [{v['attained_reduction'][1]:.4f}, {v['attained_reduction'][2]:.4f}]   per-entry {[round(x, 4) for x in v['per_entry']]}   predicted(closed form) {PRED_FLOW.get(arm, float('nan')):.4f}   attained/predicted {v['attained_reduction'][0] / PRED_FLOW.get(arm, float('nan')):.2f}")
        else:
            lines.append(f"  {k:36s} {v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]   (predicted ratio 1.89; pass band [1.5, 2.3])")
    pe = per_entry(res, "held_out"); ents = sorted(pe)
    lines.append(f"DOWNSTREAM held-out, % improvement over identity, per entry {ents}:")
    meths = [k for k in pe[ents[0]] if k in ("shrink", "linear_diag") or k.endswith("_n1") or k.endswith("_n8")]
    for k in meths:
        vals = [pe[e][k] for e in ents]
        mm, lo, hi = boot(vals)
        lines.append(f"  {k:26s} " + "  ".join(f"{v:+7.2f}" for v in vals) + f"   mean {mm:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]")
    lines.append("MODEL MINUS SHRINKAGE (percentage points, same 4 held-out entries; positive = model above shrinkage):")
    deltas = {}
    for k in [x for x in meths if x.startswith("dit_")]:
        d = [pe[e][k] - pe[e]["shrink"] for e in ents]
        mm, lo, hi = boot(d); deltas[k] = {"per_entry": dict(zip(ents, d)), "mean": mm, "ci95": [lo, hi], "n_positive": int(sum(x > 0 for x in d)), "n": len(d)}
        lines.append(f"  {k:26s} " + "  ".join(f"{v:+6.2f}" for v in d) + f"   mean {mm:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]   positive in {sum(x > 0 for x in d)}/{len(d)}")
    out["deltas_vs_shrink"] = deltas
    pi = per_entry(res, "in_sample")
    lines.append("IN-SAMPLE (training proteins; labelled in-sample), mean over entries of % over identity: " + ", ".join(f"{k}={np.mean([pi[e][k] for e in pi]):+.2f}" for k in meths if k.endswith("_ema_n1") or k == "shrink"))
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    ms = sys.argv[1:] or ["025", "050", "075", "100"]
    allout = {m: report(m) for m in ms}
    (D / "phase3_report.json").write_text(json.dumps({m: v for m, v in allout.items() if v}, indent=1))
