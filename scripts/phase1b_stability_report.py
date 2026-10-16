#!/usr/bin/env python3
"""
scripts/phase1b_stability_report.py

Applies the pre-registered reading for the PCA-axis stability diagnostics (S-a principal angles, S-b held-out variance and capture ratio) to
results/phase1/b1_10ps_stability/<protein>.json (experiments/EXPERIMENT_LOG.md, 2026-10-16 night). Paired Chai - Cart over entries, 95% bootstrap CI.
PRIMARY: P1 mean matched-axis |cos| over indices 61-137, P2 S_100, P3 mean log(held-out/training variance) over indices 61-137, P4 capture ratio at m = 100.
READING: MECHANISM IDENTIFIED iff P2 and P4 are each higher for Chai-1 by >= 0.05 with a CI excluding 0 (Cartesian tail axes less stable / capturing less);
REJECTED iff |diff| < 0.05 with a CI including 0 in both, or Cartesian MORE stable in either; otherwise INTERMEDIATE.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase1b_report_10ps import SET13, boot_ci, entry

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1/b1_10ps_stability")
LA = Path(sys.argv[2] if len(sys.argv) > 2 else "results/phase1/atlas10ps/local_alignment")
THRESH = 0.05


def reading(d2, ci2, d4, ci4) -> str:
    """d = Chai - Cart mean difference, ci = (lo, hi), for P2 (S_100) and P4 (capture ratio at 100)."""
    ident = all(d >= THRESH and ci[0] > 0 for d, ci in ((d2, ci2), (d4, ci4)))
    rejected = all(abs(d) < THRESH and ci[0] <= 0 <= ci[1] for d, ci in ((d2, ci2), (d4, ci4))) or d2 < 0 or d4 < 0
    return "MECHANISM IDENTIFIED" if ident else ("MECHANISM REJECTED" if rejected else "INTERMEDIATE")


def stats(r, space):
    s = r["spaces"][space]
    ac = np.array(s["axis_cos"]); ho = np.array(s["heldout_ratio"])
    out = {"P1": float(ac[:, 60:137].mean()), "P2": float(np.mean([v for v in s["S_m"]["100"] if v is not None])),
           "P3": float(np.log(ho[:, 60:137]).mean()), "P4": float(np.mean([v for v in s["capture_m"]["100"] if v is not None]))}
    for q in ("0.9", "0.95", "0.99"):
        out[f"S_q{q}"] = float(np.mean([v[1] for v in s["S_mq"][q]])); out[f"cap_q{q}"] = float(np.mean(s["capture_mq"][q]))
        out[f"m_q{q}"] = float(np.mean(s["mq"][q]))
    return out


def main() -> None:
    R = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(D / "*.json")))}
    passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
    sets = {"SET13": [p for p in SET13 if p in R]}
    u = sorted(set(sets["SET13"]) | set(p for p in passL if p in R))
    if set(u) != set(sets["SET13"]):
        sets["UNION"] = u
    for sname, ps in sets.items():
        ents = sorted(set(entry(p) for p in ps)); ST = {p: {sp: stats(R[p], sp) for sp in ("chai", "cart")} for p in ps}
        print(f"\n===== {sname}: {len(ps)} proteins / {len(ents)} entries; paired Chai - Cart (positive = Chai-1 more stable), 95% CI =====")
        res = {}
        for key, desc in (("P1", "mean matched-axis |cos|, idx 61-137"), ("P2", "S_100 (principal angles, top-100)"), ("P3", "mean log held-out/train variance, idx 61-137"), ("P4", "capture ratio, m = 100"),
                          ("S_q0.9", "S at m_90%"), ("S_q0.95", "S at m_95%"), ("S_q0.99", "S at m_99%"), ("cap_q0.9", "capture at m_90%"), ("cap_q0.95", "capture at m_95%"), ("cap_q0.99", "capture at m_99%")):
            d = np.array([np.mean([ST[p]["chai"][key] - ST[p]["cart"][key] for p in ps if entry(p) == e]) for e in ents]); mu, lo, hi = boot_ci(d)
            ch = np.mean([ST[p]["chai"][key] for p in ps]); ca = np.mean([ST[p]["cart"][key] for p in ps]); res[key] = (mu, (lo, hi))
            print(f"  {key:9s} {desc:46s} Chai {ch:+.3f}  Cart {ca:+.3f}  diff {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print("  mean m reaching 90/95/99% of variance (chai | cart):", [f"{np.mean([ST[p]['chai'][f'm_q{q}'] for p in ps]):.0f}|{np.mean([ST[p]['cart'][f'm_q{q}'] for p in ps]):.0f}" for q in ("0.9", "0.95", "0.99")])
        print(f"  READING ({sname}): {reading(res['P2'][0], res['P2'][1], res['P4'][0], res['P4'][1])}")
        for sp in ("chai", "cart"):
            ac = np.mean([np.mean(np.array(R[p]["spaces"][sp]["axis_cos"]), 0)[:137] for p in ps], 0); ho = np.mean([np.mean(np.array(R[p]["spaces"][sp]["heldout_ratio"]), 0)[:137] for p in ps], 0)
            bands = [(0, 5), (5, 20), (20, 60), (60, 100), (100, 137)]
            print(f"  {sp}: matched-axis |cos| by band {[round(float(ac[a:b].mean()), 2) for a, b in bands]};  held-out/train variance by band {[round(float(ho[a:b].mean()), 2) for a, b in bands]}")


if __name__ == "__main__":
    main()
