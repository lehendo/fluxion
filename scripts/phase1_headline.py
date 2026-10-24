#!/usr/bin/env python3
"""Measured f(t), attainable reduction and attenuation on the 19 proteins (paper Section 5.4)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from fluxion.analysis.probes import fit_linear_diag, predict_linear_diag
from fluxion.analysis.stats import entry_groups

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
_trapz = getattr(np, 'trapezoid', None) or np.trapz
T_GRID = [0.0, 0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99, 1.0]


def stats_for(x, y):
    """per-channel sufficient statistics (float64) of x -> y regression."""
    x, y = x.double(), y.double()
    return {"n": x.shape[0], "sx": x.sum(0), "sy": y.sum(0), "sxx": (x * x).sum(0), "sxy": (x * y).sum(0), "syy": (y * y).sum(0)}


def add(a, b):
    return {k: a[k] + b[k] for k in a}


def fit_from_stats(s):
    k = (s["sxy"] - s["sx"] * s["sy"] / s["n"]) / (s["sxx"] - s["sx"] ** 2 / s["n"]).clamp_min(1e-30)
    a = (s["sy"] - k * s["sx"]) / s["n"]
    return a, k


def sse_from_stats(s, a, k):
    return (s["syy"] - 2 * a * s["sy"] - 2 * k * s["sxy"] + s["n"] * a * a + 2 * a * k * s["sx"] + k * k * s["sxx"]).sum().item()


def boot_mean_ratio(vals_num, vals_den, groups, n_boot, seed):
    """bootstrap over entries of mean(num), mean(den), and ratio of means."""
    ents = list(groups); rng = np.random.default_rng(seed)
    out = {"num": [], "den": [], "ratio": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ents), len(ents))
        ps = [p for i in pick for p in groups[ents[i]]]
        n, d = np.mean([vals_num[p] for p in ps]), np.mean([vals_den[p] for p in ps])
        out["num"].append(n); out["den"].append(d); out["ratio"].append(n / d)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DATA_ROOT / "phase1_cache")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/headline.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cells = {}
    for p in PILOT:
        c = torch.load(args.cache / f"{p}__lag1.pt", weights_only=False)
        m = torch.load(args.cache / f"{p}__meta.pt", weights_only=False)["mean_pos"].float()
        pos = c["pos"].long()
        cells[p] = {"z0": c["z0"] - m[pos], "z1": c["z1"] - m[pos], "zp": c["zprev"] - m[pos]}
    groups = entry_groups(PILOT)
    # per-protein, per-t sufficient statistics of (z_t -> delta)
    S = {p: [] for p in PILOT}
    for p in PILOT:
        d = cells[p]["z1"] - cells[p]["z0"]
        for t in T_GRID:
            S[p].append(stats_for((1 - t) * cells[p]["z0"] + t * cells[p]["z1"], d))
    f = {p: np.zeros(len(T_GRID)) for p in PILOT}
    direct0, direct01 = {}, {}
    for entry, members in groups.items():
        train = [p for p in PILOT if p not in members]
        z0 = torch.cat([cells[p]["z0"] for p in train]); z1 = torch.cat([cells[p]["z1"] for p in train])
        zp = torch.cat([cells[p]["zp"] for p in train])
        a0, k0, _ = fit_linear_diag(z0, z1)
        a1, kk0, kk1 = fit_linear_diag(z0, z1, zp)
        for ti in range(len(T_GRID)):
            tot = None
            for p in train:
                tot = S[p][ti] if tot is None else add(tot, S[p][ti])
            a, k = fit_from_stats(tot)
            for p in members:
                f[p][ti] = 1 - sse_from_stats(S[p][ti], a.double(), k.double()) / S[p][ti]["syy"].sum().item()
        for p in members:
            u2 = ((cells[p]["z1"] - cells[p]["z0"]) ** 2).sum().item()
            direct0[p] = 1 - ((predict_linear_diag(cells[p]["z0"], a0, k0) - cells[p]["z1"]) ** 2).sum().item() / u2
            direct01[p] = 1 - ((predict_linear_diag(cells[p]["z0"], a1, kk0, cells[p]["zp"], kk1) - cells[p]["z1"]) ** 2).sum().item() / u2
        print(f"[headline] fold {entry} done", flush=True)
    tg = np.array(T_GRID)
    favg = {p: float(_trapz(f[p], tg)) for p in PILOT}                      # uniform-t average on [0,1]
    # f_0 must equal the direct z0-only regression (same estimator class, same fit)
    gap = max(abs(f[p][0] - direct0[p]) for p in PILOT)
    assert gap < 1e-4, f"f_0 != direct0 (max gap {gap}); estimator footing mismatch"
    ci_pt = {}
    for ti, t in enumerate(T_GRID):
        v = {p: f[p][ti] for p in PILOT}
        b = boot_mean_ratio(v, v, groups, 2000, ti)["num"]
        ci_pt[str(t)] = [float(np.mean(list(v.values()))), b[0], b[1]]
    b_ratio = boot_mean_ratio(direct0, favg, groups, args.n_boot, 1)
    b_ratio01 = boot_mean_ratio(direct01, favg, groups, args.n_boot, 2)
    res = {"t_grid": T_GRID, "per_t_mean_ci": ci_pt,
           "f_avg_mean": float(np.mean(list(favg.values()))), "f_avg_ci": b_ratio["den"],
           "direct0_mean": float(np.mean(list(direct0.values()))), "direct0_ci": b_ratio["num"],
           "direct01_mean": float(np.mean(list(direct01.values()))), "direct01_ci": b_ratio01["num"],
           "ratio_direct0_over_favg": float(np.mean(list(direct0.values())) / np.mean(list(favg.values()))),
           "ratio_direct0_over_favg_ci": b_ratio["ratio"],
           "ratio_direct01_over_favg": float(np.mean(list(direct01.values())) / np.mean(list(favg.values()))),
           "ratio_direct01_over_favg_ci": b_ratio01["ratio"],
           "f0_equals_direct0_max_gap": float(gap),
           "per_protein": {p: {"f": f[p].tolist(), "f_avg": favg[p], "direct0": direct0[p], "direct01": direct01[p]} for p in PILOT},
           "n_entries": len(groups), "n_proteins": len(PILOT)}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res))
    print("\n=== E1: fraction of ||u||^2 explained (held-out entries, mean over 19 proteins, 95% CI over 17 entries) ===")
    print(f"direct regression z0 only  (== f(t=0), asserted, gap {gap:.1e}): {100*res['direct0_mean']:6.2f}%  [{100*res['direct0_ci'][0]:.2f}, {100*res['direct0_ci'][1]:.2f}]")
    print(f"direct regression (z0, z_prev) (ladder's 27%, info the model never sees): {100*res['direct01_mean']:6.2f}%  [{100*res['direct01_ci'][0]:.2f}, {100*res['direct01_ci'][1]:.2f}]")
    print(f"OT-CFM attainable, uniform-t average f_avg:                    {100*res['f_avg_mean']:6.2f}%  [{100*res['f_avg_ci'][0]:.2f}, {100*res['f_avg_ci'][1]:.2f}]")
    print(f"ratio direct0 / f_avg = {res['ratio_direct0_over_favg']:.2f}  CI [{res['ratio_direct0_over_favg_ci'][0]:.2f}, {res['ratio_direct0_over_favg_ci'][1]:.2f}]")
    print(f"ratio direct(z0,zprev) / f_avg = {res['ratio_direct01_over_favg']:.2f}  CI [{res['ratio_direct01_over_favg_ci'][0]:.2f}, {res['ratio_direct01_over_favg_ci'][1]:.2f}]")
    print("\n  t      f_t (mean, 95% CI)")
    for t in T_GRID:
        m, lo, hi = ci_pt[str(t)]; print(f"{t:5.2f}  {100*m:7.3f}% [{100*lo:7.3f}, {100*hi:7.3f}]")


if __name__ == "__main__":
    main()
