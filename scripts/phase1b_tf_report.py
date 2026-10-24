#!/usr/bin/env python3
"""Report of the template-feature control (paper Appendix A.5)."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from phase1b_report_10ps import SET13, boot_ci, entry

B1, TF, LA = Path("paper_results/phase1/b1_10ps"), Path("paper_results/phase1/b1_10ps_tf"), Path("paper_results/phase1/atlas10ps/local_alignment")


def reading(lead_ct, ci_ct, lead_tc, ci_tc):
    """lead_ct/ci_ct: Chai - TF (per-channel linear) for one set; lead_tc/ci_tc: TF - Cartesian."""
    return lead_ct, ci_ct, lead_tc, ci_tc


def classify(per_set):
    surv = all(v["ct"][0] >= 4.6 and v["ct"][1][0] > 0 for v in per_set.values())
    expl = all((v["ct"][0] < 2.3 or v["ct"][1][0] <= 0) and v["tc"][0] >= 4.6 and v["tc"][1][0] > 0 for v in per_set.values())
    return "LEAD SURVIVES THE MATCHED CONTROL" if surv else ("LEAD EXPLAINED BY INPUT INFORMATION" if expl else "INTERMEDIATE")


def main() -> None:
    R = {p: json.load(open(B1 / f"{p}.json")) for p in SET13}; T = {p: json.load(open(TF / f"{p}.json")) for p in SET13 if (TF / f"{p}.json").exists()}
    passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
    union = sorted(set(SET13) | set(passL)); sets = {"SET13": SET13, "UNION": union}
    for p in union:
        R.setdefault(p, json.load(open(B1 / f"{p}.json"))); T.setdefault(p, json.load(open(TF / f"{p}.json")))
    out = {}
    for sname, ps in sets.items():
        ents = sorted(set(entry(p) for p in ps)); out[sname] = {}
        print(f"\n===== {sname}: {len(ps)} proteins / {len(ents)} entries; points over identity at 10 ps; 95% CI [Bonferroni m = 4 CI] =====")
        for pred in ("linear_diag", "shrink"):
            def v(rep, p):
                return (T[p]["lags"]["10ps"]["tf"] if rep == "tf" else R[p]["lags"]["10ps"][rep])[pred]["pct_vs_identity"]
            ent = lambda rep: np.array([np.mean([v(rep, p) for p in ps if entry(p) == e]) for e in ents])
            ch, tf, ca = ent("chai"), ent("tf"), ent("cart")
            for name, d in (("Chai - Cart", ch - ca), ("Chai - TF", ch - tf), ("TF - Cart", tf - ca)):
                mu, lo, hi = boot_ci(d); _, blo, bhi = boot_ci(d, level=98.75)
                print(f"  {pred:12s} {name:12s} {mu:+7.2f} [{lo:+7.2f}, {hi:+7.2f}] [{blo:+7.2f}, {bhi:+7.2f}]   absolute: Chai {ch.mean():+6.2f}  TF {tf.mean():+6.2f}  Cart {ca.mean():+6.2f}")
                if pred == "linear_diag":
                    out[sname]["ct" if name == "Chai - TF" else ("tc" if name == "TF - Cart" else "cc")] = (mu, (blo, bhi))
    print("\nREADING:", classify(out))


if __name__ == "__main__":
    main()
