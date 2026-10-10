#!/usr/bin/env python3
"""
scripts/phase1a_ladder.py

Step 0 as an ATTRIBUTION LADDER. One protocol change per rung, old split
{1bxy_A, 1bxy_B, 1btk_A} until the last rung, (z0, z_prev) inputs and the OLD MLP throughout:

  S0.0  old protocol verbatim: padding counted (rows uniform over ALL positions), equal weight per
        protein, MLP early-stops on the TEST proteins.  Old published: mean 0.01944, linear 0.001638,
        MLP 0.001701; identity 0.002079 (DiT-eval number, token-weighted: reported with its own weighting).
        PASS iff each old value lies inside the bootstrap 95% CI of the new estimate. FAIL => stop.
  S0.1  + MLP early-stops on training-side entries only.
  S0.2  + padding masked (real residues only).
  S0.3  + leave-one-PDB-entry-out (17 folds).
For S0.1-S0.3 there is no tolerance: the deltas are the result.

Reports both baselines for every rung (% reduction vs MEAN, the old headline currency, and % vs
IDENTITY). Bootstrap = time blocks of length lag + 2*tau_int per protein (acf.json), resampling with
fitted models held fixed (evaluation uncertainty); MLP training randomness is shown separately as the
spread over seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from analysis.probes import fit_linear_diag, predict_linear_diag, train_old_mlp, row_sqerr
from analysis.stats import (entry_groups, time_blocks, block_len, pct_improvement, entry_bootstrap_mean)

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
OLD_VAL = ["1bxy_A", "1bxy_B", "1btk_A"]
OLD = {"mean": 0.01944, "linear": 0.001638, "mlp": 0.001701, "identity_dit_weighting": 0.002079}


def load(cache: Path, p: str, acf: dict):
    cell = torch.load(cache / f"{p}__lag1.pt", weights_only=False)
    meta = torch.load(cache / f"{p}__meta.pt", weights_only=False)
    mp, pos = meta["mean_pos"].float(), cell["pos"].long()
    d = {"p": p, "L": cell["L_bucket"], "n_res": cell["n_real"],
         "z0": cell["z0"] - mp[pos], "z1": cell["z1"] - mp[pos], "zp": cell["zprev"] - mp[pos],
         "pad": cell["is_pad"].bool()}
    d["blocks"] = time_blocks(cell["replica"].numpy(), cell["frame"].numpy(), block_len(1, acf[p]["tau_int_frames"]))
    return d


def sub(c, keep):
    out = dict(c)
    for k in ("z0", "z1", "zp", "pad"):
        out[k] = c[k][keep]
    out["blocks"] = c["blocks"][keep.numpy()]
    return out


def maybe_mask(cells, mask_pad):
    return [sub(c, ~c["pad"]) for c in cells] if mask_pad else cells


def cat(cells, k):
    return torch.cat([c[k] for c in cells], 0)


def fit_predict(lin_train, mlp_train, stop, test, device, seeds):
    """mean / linear(z0,zp) are fit on `lin_train`; the old MLP is fit on `mlp_train` and early-stopped on
    `stop`. S0.0: stop = the TEST proteins (the old leak). S0.1+: stop = held-in entries that are NOT in
    `mlp_train`. Returns per-row squared errors on each `test` protein."""
    z0, z1, zp = cat(lin_train, "z0"), cat(lin_train, "z1"), cat(lin_train, "zp")
    mean_vec = z1.mean(0, keepdim=True)
    a, k0, k1 = fit_linear_diag(z0, z1, zp)
    mz0, mz1, mzp = cat(mlp_train, "z0"), cat(mlp_train, "z1"), cat(mlp_train, "zp")
    mlps = [train_old_mlp(mzp, mz0, mz1, cat(stop, "zp"), cat(stop, "z0"), cat(stop, "z1"),
                          device=device, seed=s)[0] for s in seeds]
    out = {}
    with torch.no_grad():
        for c in test:
            y = c["z1"]
            e = {"identity": row_sqerr(c["z0"], y), "mean": row_sqerr(mean_vec.expand_as(y), y),
                 "linear": row_sqerr(predict_linear_diag(c["z0"], a, k0, c["zp"], k1), y)}
            per_seed = [row_sqerr(m(c["zp"], c["z0"]), y) for m in mlps]
            e["mlp"] = torch.stack(per_seed).mean(0)
            e["mlp_seeds"] = torch.stack(per_seed)
            out[c["p"]] = {k: v.numpy() for k, v in e.items()}
    return out


def pooled(errs, cells, weighting="probe"):
    """probe: equal weight per protein (old probe pooled ~150k rows per protein).
    dit  : token-weighted by L_bucket (the DiT eval drew frames uniformly, tokens ~ L)."""
    out = {}
    for k in ("identity", "mean", "linear", "mlp"):
        m = np.array([errs[c["p"]][k].mean() for c in cells])
        w = np.ones(len(cells)) if weighting == "probe" else np.array([c["L"] for c in cells], float)
        out[k] = float((m * w).sum() / w.sum())
    out["mlp_seed_mse"] = [float(np.mean([errs[c["p"]]["mlp_seeds"][s].mean() for c in cells]))
                           for s in range(errs[cells[0]["p"]]["mlp_seeds"].shape[0])]
    return out


def bootstrap(errs, cells, weighting, n_boot, seed):
    rng = np.random.default_rng(seed)
    keys = ("identity", "mean", "linear", "mlp")
    pre = {}
    for c in cells:
        b = c["blocks"]; nb = b.max() + 1; cnt = np.bincount(b, minlength=nb).astype(float)
        pre[c["p"]] = (nb, cnt, {k: np.bincount(b, weights=errs[c["p"]][k], minlength=nb) for k in keys})
    w = np.ones(len(cells)) if weighting == "probe" else np.array([c["L"] for c in cells], float)
    draws = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        per = {k: [] for k in keys}
        for c in cells:
            nb, cnt, s = pre[c["p"]]
            pick = rng.integers(0, nb, nb)
            for k in keys:
                per[k].append(s[k][pick].sum() / cnt[pick].sum())
        for k in keys:
            draws[k][i] = (np.array(per[k]) * w).sum() / w.sum()
    return {k: [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))] for k in keys}


def rung_row(name, desc, p):
    return {"rung": name, "desc": desc, **{k: p[k] for k in ("identity", "mean", "linear", "mlp")},
            "id_vs_mean": pct_improvement(p["identity"], p["mean"]),
            "lin_vs_mean": pct_improvement(p["linear"], p["mean"]),
            "mlp_vs_mean": pct_improvement(p["mlp"], p["mean"]),
            "lin_vs_id": pct_improvement(p["linear"], p["identity"]),
            "mlp_vs_id": pct_improvement(p["mlp"], p["identity"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_cache_s0all"))
    ap.add_argument("--acf", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/acf.json"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/ladder.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force-rungs", action="store_true", help="run S0.1-S0.3 even if S0.0 fails (not the default)")
    args = ap.parse_args()
    acf = json.loads(args.acf.read_text())
    print("loading dense lag-1 cache ...", flush=True)
    cells = {p: load(args.cache, p, acf) for p in PILOT}
    for p in OLD_VAL:
        c = cells[p]
        print(f"  {p}: {len(c['z0'])} rows, {100 * c['pad'].float().mean():.1f}% padding rows "
              f"(n_res {c['n_res']} of L {c['L']})", flush=True)
    groups = entry_groups(PILOT)
    train_ids = [p for p in PILOT if p not in OLD_VAL]
    val = [cells[p] for p in OLD_VAL]
    train = [cells[p] for p in train_ids]
    res: dict = {"old_published": OLD, "rungs": []}

    # ------------------------------------------------------------ S0.0
    print("\n=== S0.0 old protocol verbatim (padding counted, MLP early-stops on TEST) ===", flush=True)
    e = fit_predict(train, train, val, val, args.device, args.seeds)
    pp, pd = pooled(e, val, "probe"), pooled(e, val, "dit")
    bp, bd = bootstrap(e, val, "probe", args.n_boot, 0), bootstrap(e, val, "dit", args.n_boot, 0)
    checks = [("mean (probe weighting)", OLD["mean"], pp["mean"], bp["mean"]),
              ("linear (probe weighting)", OLD["linear"], pp["linear"], bp["linear"]),
              ("mlp (probe weighting)", OLD["mlp"], pp["mlp"], bp["mlp"]),
              ("identity (DiT token weighting)", OLD["identity_dit_weighting"], pd["identity"], bd["identity"])]
    ok_all = True
    print(f"{'quantity':34s} {'old published':>14s} {'new estimate':>13s} {'95% CI':>26s}  inside?  delta")
    for name, old, new, ci in checks:
        ok = ci[0] <= old <= ci[1]; ok_all &= ok
        print(f"{name:34s} {old:14.6f} {new:13.6f} [{ci[0]:11.6f},{ci[1]:11.6f}]  {'PASS' if ok else 'FAIL'}  {100 * (new / old - 1):+6.1f}%")
    print(f"identity under PROBE weighting (not comparable to 0.002079): {pp['identity']:.6f}")
    print(f"MLP seed spread (training randomness, CI above is evaluation-only): {np.round(pp['mlp_seed_mse'], 6).tolist()}")
    res["S0.0"] = {"probe_weighting": pp, "dit_weighting": pd, "ci_probe": bp, "ci_dit": bd, "all_pass": ok_all,
                   "checks": [{"name": n, "old": o, "new": v, "ci": c, "pass": c[0] <= o <= c[1]} for n, o, v, c in checks]}
    res["rungs"].append(rung_row("S0.0", "old protocol verbatim", pp))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    if not ok_all and not args.force_rungs:
        print("\nS0.0 FAILS the CI criterion -> STOP (sixth finding). Rungs S0.1-S0.3 NOT run.")
        return
    print("\nS0.0 PASS" if ok_all else "\nS0.0 FAIL but --force-rungs set", flush=True)

    # ------------------------------------------------------------ S0.1
    inner_ids = ["1b8d", "1b0n"]       # two held-in entries (fixed, recorded) for MLP early stopping
    fit_ids = [p for p in train_ids if p.split("_")[0] not in inner_ids]
    inner = [cells[p] for p in train_ids if p.split("_")[0] in inner_ids]
    fit = [cells[p] for p in fit_ids]
    print(f"\n=== S0.1 + MLP early-stops on training-side entries {inner_ids} only ===", flush=True)
    e = fit_predict(fit + inner, fit, inner, val, args.device, args.seeds)   # MLP fit on `fit` only; stop on held-in `inner`
    p1 = pooled(e, val, "probe"); res["rungs"].append(rung_row("S0.1", f"+ MLP early-stop on held-in entries {inner_ids}", p1))

    # ------------------------------------------------------------ S0.2
    print("\n=== S0.2 + padding masked (real residues only) ===", flush=True)
    m_train, m_inner, m_val = maybe_mask(fit + inner, True), maybe_mask(inner, True), maybe_mask(val, True)
    e = fit_predict(m_train, maybe_mask(fit, True), m_inner, m_val, args.device, args.seeds)
    p2 = pooled(e, m_val, "probe"); res["rungs"].append(rung_row("S0.2", "+ padding masked", p2))
    res["S0.2_real_rows"] = {c["p"]: int(len(c["z0"])) for c in m_val}

    # ------------------------------------------------------------ S0.3
    print("\n=== S0.3 + leave-one-PDB-entry-out (17 folds), padding masked ===", flush=True)
    masked = {p: sub(cells[p], ~cells[p]["pad"]) for p in PILOT}
    per_protein = {}
    for entry, members in groups.items():
        others = [e_ for e_ in groups if e_ != entry]
        stop_entries = [others[0], others[len(others) // 2]]
        tr_ids = [p for e_ in others if e_ not in stop_entries for p in groups[e_]]
        st_ids = [p for e_ in stop_entries for p in groups[e_]]
        eo = fit_predict([masked[p] for p in tr_ids + st_ids], [masked[p] for p in tr_ids],
                         [masked[p] for p in st_ids], [masked[p] for p in members], args.device, args.seeds[:1])
        for p in members:
            per_protein[p] = {k: float(eo[p][k].mean()) for k in ("identity", "mean", "linear", "mlp")}
        print(f"  fold {entry}: " + ", ".join(f"{p} mlp/id={per_protein[p]['mlp'] / per_protein[p]['identity']:.3f}" for p in members), flush=True)
    three = {k: float(np.mean([per_protein[p][k] for p in OLD_VAL])) for k in ("identity", "mean", "linear", "mlp")}
    three["mlp_seed_mse"] = []
    res["rungs"].append(rung_row("S0.3", "+ leave-one-entry-out (the 3 old proteins, now held out per entry)", three))
    allp = {k: float(np.mean([per_protein[p][k] for p in PILOT])) for k in ("identity", "mean", "linear", "mlp")}
    res["rungs"].append(rung_row("S0.3-all19", "same, mean over ALL 19 proteins", allp))
    pct = lambda a, b: {p: pct_improvement(per_protein[p][a], per_protein[p][b]) for p in PILOT}
    res["S0.3_per_protein"] = per_protein
    res["S0.3_across"] = {n: dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean(pct(a, b), n_boot=10000)))
                          for n, (a, b) in {"mlp_vs_mean": ("mlp", "mean"), "linear_vs_mean": ("linear", "mean"),
                                            "identity_vs_mean": ("identity", "mean"), "mlp_vs_identity": ("mlp", "identity"),
                                            "linear_vs_identity": ("linear", "identity")}.items()}

    # ------------------------------------------------------------ attribution table
    print("\n=== ATTRIBUTION LADDER (lag 1 = 10 ps; predictors on (z0, z_prev)) ===")
    h = f"{'rung':11s} {'mean':>9s} {'identity':>9s} {'linear':>9s} {'MLP':>9s} | {'id/mean':>8s} {'lin/mean':>9s} {'MLP/mean':>9s} | {'lin/id':>7s} {'MLP/id':>7s}"
    print(h)
    for r in res["rungs"]:
        print(f"{r['rung']:11s} {r['mean']:9.6f} {r['identity']:9.6f} {r['linear']:9.6f} {r['mlp']:9.6f} | "
              f"{r['id_vs_mean']:7.2f}% {r['lin_vs_mean']:8.2f}% {r['mlp_vs_mean']:8.2f}% | {r['lin_vs_id']:6.2f}% {r['mlp_vs_id']:6.2f}%")
    print("old published: mean 0.01944, linear 0.001638, MLP 0.001701 (MLP/mean 91.25%); identity 0.002079 (DiT weighting)")
    print("\nS0.3 across all 19 proteins, mean % with 95% CI over PDB entries:")
    for n, v in res["S0.3_across"].items():
        print(f"  {n:20s} {v['mean']:+7.2f}%  [{v['lo']:+7.2f}, {v['hi']:+7.2f}]")
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
