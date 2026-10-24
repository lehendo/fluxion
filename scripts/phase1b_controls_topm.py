#!/usr/bin/env python3
"""Post hoc restriction of the 10 ps lead to the top components (paper Appendix A.5)."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from phase1b_report_10ps import SET13, boot_ci, entry

CTRL = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_results/phase1/b1_10ps_controls")
LA = Path(sys.argv[2] if len(sys.argv) > 2 else "paper_results/phase1/atlas10ps/local_alignment")
R = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(CTRL / "*.json")))}
passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
sets = {"SET13": [p for p in SET13 if p in R]}
sets["UNION"] = sorted(set(sets["SET13"]) | set(p for p in passL if p in R))


def pct_m(r, rep, pred, m, view="raw"):
    d = r["lags"]["10ps"][rep][view]["per_dim_mse"]; idn = np.array(d["identity"]); pr = np.array(d[pred])
    m = min(m, len(idn)); return 100 * (1 - pr[:m].sum() / idn[:m].sum())


for sname, ps in sets.items():
    ents = sorted(set(entry(p) for p in ps))
    print(f"\n{sname}: {len(ps)} proteins / {len(ents)} entries   (POST HOC; Chai - Cart in points over identity, restricted to the top-m components)")
    for pred in ("linear_diag", "shrink"):
        for m in (5, 20, 60, 150, 10 ** 6):
            d = np.array([np.mean([pct_m(R[p], "chai", pred, m) - pct_m(R[p], "cart", pred, m) for p in ps if entry(p) == e]) for e in ents])
            ch = np.mean([np.mean([pct_m(R[p], "chai", pred, m) for p in ps if entry(p) == e]) for e in ents]); ca = np.mean([np.mean([pct_m(R[p], "cart", pred, m) for p in ps if entry(p) == e]) for e in ents])
            mu, lo, hi = boot_ci(d)
            print(f"  {pred:12s} m = {'all' if m > 1000 else m:>4}: lead {mu:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]   absolute Chai {ch:+6.2f}  Cart {ca:+6.2f}")
    for rep in ("chai", "cart"):
        vs, es = [], []
        for p in ps:
            sp = np.array(R[p]["spectrum"][rep]); idn = np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["identity"]); m = 60
            if len(sp) > m: vs.append(sp[m:].sum() / sp.sum()); es.append(idn[m:].sum() / idn.sum())
        print(f"  beyond component 60 ({len(vs)} proteins with k > 60), {rep}: variance share {np.mean(vs):.3f}, identity-error share {np.mean(es):.3f}")
