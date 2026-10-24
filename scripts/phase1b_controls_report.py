#!/usr/bin/env python3
"""Registered readings of the 10 ps controls (paper Appendix A.5)."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from fluxion.analysis.b1_controls import BANDS
from phase1b_report_10ps import SET13, boot_ci, entry

CTRL = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_results/phase1/b1_10ps_controls")
LA = Path(sys.argv[2] if len(sys.argv) > 2 else "paper_results/phase1/atlas10ps/local_alignment")
M_SETS = 2


def read_c1(lead_unwhitened: float, point: float, lo: float, hi: float) -> str:
    """lo, hi = 97.5% interval of the whitened linear_diag lead."""
    if lo <= 0 <= hi or point < 0.25 * lead_unwhitened:
        return "COLLAPSES"
    return "SURVIVES" if point >= 0.5 * lead_unwhitened else "INTERMEDIATE"


def read_c2(tail_share: float, top_band_leads: list[float]) -> str:
    if tail_share >= 0.5:
        return "TAIL"
    return "HEAD" if (tail_share <= 0.25 and any(x > 0 for x in top_band_leads[:2])) else "MIXED"


def read_c3(lead_linear: float, point: float, lo: float, hi: float) -> str:
    if point < 0.5 * lead_linear and lo <= 0 <= hi:
        return "INTERCEPT-DRIVEN"
    return "SLOPE-DRIVEN" if (point >= 0.75 * lead_linear and not lo <= 0 <= hi) else "INTERMEDIATE"


def band_vec(mid: np.ndarray, mp: np.ndarray) -> np.ndarray:
    """Variance-weighted band contributions (points of % improvement over identity), zero-padded to len(BANDS)."""
    tot = mid.sum(); k = len(mid)
    return np.array([100 * (mid[lo:min(hi, k)] - mp[lo:min(hi, k)]).sum() / tot if lo < k else 0.0 for lo, hi in BANDS])


def tail_contrib(mid: np.ndarray, mp: np.ndarray) -> tuple[float, float]:
    h = len(mid) // 2; tot = mid.sum()
    return float(100 * (mid[:h] - mp[:h]).sum() / tot), float(100 * (mid[h:] - mp[h:]).sum() / tot)


def main() -> None:
    R = {Path(f).stem: json.load(open(f)) for f in sorted(glob.glob(str(CTRL / "*.json")))}
    passL = [Path(f).stem for f in sorted(glob.glob(str(LA / "*.json"))) if next(iter(json.load(open(f))["per_protein"].values()))["pass"]]
    sets = {"SET13": [p for p in SET13 if p in R]}
    union = sorted(set(sets["SET13"]) | set(p for p in passL if p in R))
    if set(union) != set(sets["SET13"]):
        sets["UNION"] = union
    lvl = 100 * (1 - 0.05 / M_SETS)
    verdicts = {}
    for sname, ps in sets.items():
        ents = sorted(set(entry(p) for p in ps))
        def ent_mean(fn):
            return np.array([np.mean([fn(R[p]) for p in ps if entry(p) == e]) for e in ents])
        pct = lambda r, rep, view, pred, lag="10ps": r["lags"][lag][rep][view]["pct_vs_identity"][pred]
        lead = lambda view, pred, lag="10ps": ent_mean(lambda r: pct(r, "chai", view, pred, lag) - pct(r, "cart", view, pred, lag))
        print(f"\n===== {sname}: {len(ps)} proteins / {len(ents)} entries; intervals: 95% and {lvl:.1f}% (m = {M_SETS} sets) =====")
        for view, tag in (("raw", "UNWHITENED"), ("white", "C1 WHITENED")):
            for pred in ("linear_diag", "shrink", "noint"):
                d = lead(view, pred); mu, lo, hi = boot_ci(d); _, blo, bhi = boot_ci(d, level=lvl)
                print(f"  {tag:12s} {pred:12s} 10ps Chai - Cart = {mu:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]  m={M_SETS}: [{blo:+7.2f}, {bhi:+7.2f}]  entries>0: {int((d > 0).sum())}/{len(d)}")
        for view, tag in (("raw", "UNWHITENED"), ("white", "C1 WHITENED")):
            for pred in ("linear_diag", "shrink", "noint"):
                print(f"    absolute {tag:11s} {pred:12s} Chai {ent_mean(lambda r: pct(r, 'chai', view, pred)).mean():+7.2f}  Cart {ent_mean(lambda r: pct(r, 'cart', view, pred)).mean():+7.2f}")
        l_raw = lead("raw", "linear_diag"); l_wh = lead("white", "linear_diag"); l_ni = lead("raw", "noint")
        mu_raw = l_raw.mean(); mu_wh, _, _ = boot_ci(l_wh); _, wlo, whi = boot_ci(l_wh, level=lvl)
        c1 = read_c1(mu_raw, mu_wh, wlo, whi)
        mu_ni, _, _ = boot_ci(l_ni); _, nlo, nhi = boot_ci(l_ni, level=lvl)
        c3 = read_c3(mu_raw, mu_ni, nlo, nhi)
        # C2
        bl = {}
        for rep in ("chai", "cart"):
            bl[rep] = np.mean([[band_vec(np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["identity"]), np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["linear_diag"]))] for p in ps], 0)[0]
        band_lead = bl["chai"] - bl["cart"]
        tails = {rep: np.mean([tail_contrib(np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["identity"]), np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["linear_diag"])) for p in ps], 0) for rep in ("chai", "cart")}
        head_lead = tails["chai"][0] - tails["cart"][0]; tail_lead = tails["chai"][1] - tails["cart"][1]
        share = tail_lead / (head_lead + tail_lead) if (head_lead + tail_lead) != 0 else float("nan")
        c2 = read_c2(share, list(band_lead))
        print("\n  C2 variance-weighted band contributions to the linear_diag improvement over identity (points; bands [1-5],[6-20],[21-60],[61-150],[151-384]):")
        print("    Chai :", np.round(bl["chai"], 2)); print("    Cart :", np.round(bl["cart"], 2)); print("    LEAD :", np.round(band_lead, 2), f"sum {band_lead.sum():+.2f}")
        print(f"    halves: lead from index <= k/2 {head_lead:+.2f}, from index > k/2 {tail_lead:+.2f}; tail share of the lead {share:.1%}")
        for rep in ("chai", "cart"):
            fr = np.mean([np.array([np.sum(np.array(R[p]["spectrum"][rep])[lo:min(hi, len(R[p]["spectrum"][rep]))]) / np.sum(R[p]["spectrum"][rep]) if lo < len(R[p]["spectrum"][rep]) else 0.0 for lo, hi in BANDS]) for p in ps], 0)
            print(f"    variance fraction by band, {rep}: {np.round(fr, 3)}")
        unw = {rep: np.mean([[(100 * (1 - np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["linear_diag"]) / np.maximum(np.array(R[p]["lags"]["10ps"][rep]["raw"]["per_dim_mse"]["identity"]), 1e-30)))[lo:min(hi, len(R[p]["spectrum"][rep]))].mean() if lo < len(R[p]["spectrum"][rep]) else np.nan for lo, hi in BANDS] for p in ps], 0) for rep in ("chai", "cart")}
        print("    UNWEIGHTED per-dimension % improvement over identity by band (linear_diag):  Chai", np.round(unw["chai"], 1), " Cart", np.round(unw["cart"], 1))
        verdicts[sname] = (c1, c2, c3)
        print(f"\n  READINGS ({sname}): C1 {c1}   C2 {c2}   C3 {c3}")
        real = c1 == "SURVIVES" and c3 != "INTERCEPT-DRIVEN"
        print(f"  DECISION RULE: the per-channel-linear lead is {'REAL (write the registered sentence)' if real else 'NOT established as real (estimation / conditioning effect, or intermediate): rewrite S9'}")


if __name__ == "__main__":
    main()
