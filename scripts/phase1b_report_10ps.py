#!/usr/bin/env python3
"""
scripts/phase1b_report_10ps.py

Paired Chai-1 minus Cartesian report for B1 at the full 10 ps stride, from results/phase1/b1_10ps/<protein>.json, as pre-registered in
experiments/EXPERIMENT_LOG.md (2026-10-14, 'B1 AT 10 ps'). Statistic: per protein, difference in % improvement over identity (Chai-1 PCA minus aligned-Calpha PCA, same k,
leave-one-replica-out); proteins of one PDB entry (1b2s_E/F, 1bxy_A/B) are averaged to ONE value; 95% CI = percentile bootstrap over entries.
Protein sets: SET13 (13 proteins passing the post hoc within-replica criterion iii') and UNION (SET13 plus the proteins that pass the frame-exact local
test iii-L: results/phase1/atlas10ps/local_alignment). PRIMARY FAMILY: {linear_diag, shrink} x {10 ps} x {SET13, UNION} (Bonferroni m = 4, or 2 if the sets coincide).
Everything else is secondary (m = 16 correction printed). CLAIM LANGUAGE: learnability, never information.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

B1 = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1/b1_10ps")
LA = Path(sys.argv[2] if len(sys.argv) > 2 else "results/phase1/atlas10ps/local_alignment")
LAGS = ["10ps", "100ps", "1ns", "10ns"]
PREDS = ["linear_diag", "shrink", "linear_diag_ridge", "linear_full", "mlp", "mean"]
SET13 = "1ab1_A 1ah7_A 1aol_A 1b2s_F 1bq8_A 1btk_A 1bxy_A 1bxy_B 1af7_A 1b25_D 1b2s_E 1b8d_K 1bkp_A".split()


def entry(p: str) -> str:
    return p.split("_")[0]


def boot_ci(x, level=95.0, n=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1); tail = (100 - level) / 2
    return float(x.mean()), float(np.percentile(m, tail)), float(np.percentile(m, 100 - tail))


def per_entry_diff(R, proteins, lag, pred):
    by = {}
    for p in proteins:
        r = R[p]["lags"][lag]
        by.setdefault(entry(p), []).append(r["chai"][pred]["pct_vs_identity"] - r["cart"][pred]["pct_vs_identity"])
    return np.array([np.mean(v) for v in by.values()]), list(by)


def main() -> None:
    R = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(B1 / "*.json")))}
    passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
    sets = {"SET13": [p for p in SET13 if p in R]}
    union = sorted(set(sets["SET13"]) | set(p for p in passL if p in R))
    if set(union) != set(sets["SET13"]):
        sets["UNION"] = union
    m_primary = 2 * len(sets)
    print(f"proteins with B1-10ps results: {len(R)}; iii-L passes: {passL}\nsets: " + "; ".join(f"{k}: {len(v)} proteins / {len(set(entry(p) for p in v))} entries" for k, v in sets.items()))
    print(f"PRIMARY FAMILY (10 ps; {{linear_diag, shrink}} x sets), Bonferroni m = {m_primary} -> level {100 * (1 - 0.05 / m_primary):.2f}%")
    for sname, ps in sets.items():
        for pred in ("linear_diag", "shrink"):
            d, _ = per_entry_diff(R, ps, "10ps", pred)
            mu, lo, hi = boot_ci(d); _, blo, bhi = boot_ci(d, level=100 * (1 - 0.05 / m_primary))
            print(f"  {sname:6s} {pred:12s} 10ps  Chai - Cart = {mu:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]   Bonferroni [{blo:+6.2f}, {bhi:+6.2f}]   entries>0: {int((d > 0).sum())}/{len(d)}")
    print("\nSECONDARY (all lags and predictors; m = 16 interval for linear_diag and shrink only):")
    for sname, ps in sets.items():
        for lag in LAGS:
            for pred in PREDS:
                if any(pred not in R[p]["lags"][lag]["chai"] for p in ps):
                    continue
                d, _ = per_entry_diff(R, ps, lag, pred); mu, lo, hi = boot_ci(d)
                extra = ""
                if pred in ("linear_diag", "shrink"):
                    _, blo, bhi = boot_ci(d, level=100 * (1 - 0.05 / 16)); extra = f"  m=16 [{blo:+6.2f}, {bhi:+6.2f}]"
                print(f"  {sname:6s} {lag:>5s} {pred:18s} {mu:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]{extra}")
    print("\nABSOLUTE LEVELS (mean over entries, % over identity), SET13:")
    ps = sets["SET13"]
    for lag in LAGS:
        row = []
        for pred in ("shrink", "linear_diag"):
            for rep in ("chai", "cart"):
                v = np.mean([np.mean([R[p]["lags"][lag][rep][pred]["pct_vs_identity"] for p in ps if entry(p) == e]) for e in set(entry(p) for p in ps)])
                row.append(f"{pred}/{rep} {v:6.2f}")
        idm = np.mean([R[p]["lags"][lag]["chai"]["identity"]["mse"] / R[p]["lags"][lag]["chai"]["mean"]["mse"] for p in ps])
        print(f"  {lag:>5s}  " + "  ".join(row) + f"   id/mean (chai) {idm:.3f}")


if __name__ == "__main__":
    main()
