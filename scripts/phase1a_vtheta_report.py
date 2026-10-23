#!/usr/bin/env python3
"""Summary of the original run's velocity norm (paper Section 7.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json
from pathlib import Path
import numpy as np
import torch

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DATA_ROOT / "phase1_vtheta")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/vtheta_summary.json")
    args = ap.parse_args()
    D = {p: torch.load(args.dir / f"{p}__vtheta.pt", weights_only=False) for p in PILOT}
    t = D[PILOT[0]]["t_grid"]; nT = len(t); d = D[PILOT[0]]["d"]
    step = {D[p]["ckpt_step"] for p in PILOT}; assert len(step) == 1
    out = {"t_grid": t, "ckpt_step": step.pop(), "in_sample": True, "proteins": PILOT, "models": {}, "per_protein": {}}

    for m in ("raw", "ema"):
        S = {k: sum(D[p]["acc"][m][k] for p in PILOT).numpy() for k in ("v2", "u2", "vu", "e2")}
        N = sum(D[p]["n_tokens"] for p in PILOT) * d
        out["models"][m] = {
            "rms_v": np.sqrt(S["v2"] / N).tolist(), "rms_u": np.sqrt(S["u2"] / N).tolist(),
            "ratio": (np.sqrt(S["v2"]) / np.sqrt(S["u2"])).tolist(),
            "corr": (S["vu"] / np.sqrt(S["v2"] * S["u2"] + 1e-300)).tolist(),
            "rel_err": (np.sqrt(S["e2"]) / np.sqrt(S["u2"])).tolist(),
            "overall_ratio": float(np.sqrt(S["v2"].sum() / S["u2"].sum()))}
        pp = {}
        for p in PILOT:
            a = D[p]["acc"][m]
            pp[p] = {"ratio": (a["v2"].sqrt() / a["u2"].sqrt()).tolist(),
                     "overall_ratio": float((a["v2"].sum() / a["u2"].sum()).sqrt())}
        out["per_protein"][m] = pp
        # per-token ratio distribution (fixed token subset), pooled over proteins
        tr = np.concatenate([(D[p]["tok_rms_v"][m] / D[p]["tok_rms_u"].clamp_min(1e-12)).numpy() for p in PILOT], axis=1)
        qs = [5, 25, 50, 75, 95]
        out["models"][m]["token_ratio_quantiles_all_t"] = np.percentile(tr, qs).tolist()
        out["models"][m]["token_ratio_quantiles_t0.5"] = np.percentile(tr[t.index(0.5)], qs).tolist()
        out["models"][m]["token_ratio_quantiles_t0.01"] = np.percentile(tr[0], qs).tolist()
        out["models"][m]["token_ratio_quantiles_t0.99"] = np.percentile(tr[-1], qs).tolist()

    # empirical conditional mean (linear diagonal) from pooled sufficient statistics
    cs = {k: sum(D[p]["condmean_stats"][k] for p in PILOT).numpy() for k in ("sx", "sy", "sxx", "sxy", "syy")}
    n = sum(D[p]["n_tokens"] for p in PILOT)
    k_ = (cs["sxy"] - cs["sx"] * cs["sy"] / n) / np.maximum(cs["sxx"] - cs["sx"] ** 2 / n, 1e-30)
    a_ = (cs["sy"] - k_ * cs["sx"]) / n
    fit2 = n * a_ ** 2 + 2 * a_ * k_ * cs["sx"] + k_ ** 2 * cs["sxx"]                 # sum (a+kx)^2  [nT, d]
    sse = cs["syy"] - 2 * a_ * cs["sy"] - 2 * k_ * cs["sxy"] + fit2
    out["cond_mean"] = {"rms_pred": np.sqrt(fit2.sum(1) / (n * d)).tolist(),
                        "frac_explained_vs_zero": (1 - sse.sum(1) / cs["syy"].sum(1)).tolist(),
                        "mean_slope": k_.mean(1).tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))

    print(f"checkpoint step {out['ckpt_step']} (IN-SAMPLE: trained on all 19 proteins); {len(PILOT)} proteins, real tokens only")
    hdr = f"{'t':>5s} {'RMS|u|':>9s} | {'ratio raw':>9s} {'ratio EMA':>9s} {'corr EMA':>9s} {'relerr EMA':>10s} | {'RMS E[d|zt]':>11s} {'frac expl.':>10s} {'slope':>8s}"
    print(hdr)
    for i, tt in enumerate(t):
        r, e, c = out["models"]["raw"], out["models"]["ema"], out["cond_mean"]
        print(f"{tt:5.2f} {e['rms_u'][i]:9.5f} | {r['ratio'][i]:9.5f} {e['ratio'][i]:9.5f} {e['corr'][i]:9.5f} {e['rel_err'][i]:10.6f} | "
              f"{c['rms_pred'][i]:11.5f} {c['frac_explained_vs_zero'][i]:10.5f} {c['mean_slope'][i]:8.4f}")
    for m in ("raw", "ema"):
        o = out["models"][m]
        print(f"\n{m}: overall ||v||/||u|| = {o['overall_ratio']:.6f}; token ratio quantiles [5,25,50,75,95]%: "
              f"all t {np.round(o['token_ratio_quantiles_all_t'], 5).tolist()}  t=0.5 {np.round(o['token_ratio_quantiles_t0.5'], 5).tolist()}")
        pr = np.array([out["per_protein"][m][p]["overall_ratio"] for p in PILOT])
        print(f"   per-protein overall ratio: min {pr.min():.6f} median {np.median(pr):.6f} max {pr.max():.6f}")
    e = out["models"]["ema"]["ratio"]
    mid = np.mean([e[i] for i, tt in enumerate(t) if 0.4 <= tt <= 0.6]); end = np.mean([e[0], e[1], e[-2], e[-1]])
    print(f"\nEMA ratio: middle (0.4-0.6) {mid:.6f}  vs endpoints (0.01,0.05,0.95,0.99) {end:.6f}  -> mid/end = {mid / max(end, 1e-30):.3f}")
    print(f"empirical E[delta|z_t]: RMS at t=0.5 {out['cond_mean']['rms_pred'][t.index(0.5)]:.5f}, at endpoints "
          f"{np.mean([out['cond_mean']['rms_pred'][i] for i in (0, 1, nT - 2, nT - 1)]):.5f}")


if __name__ == "__main__":
    main()
