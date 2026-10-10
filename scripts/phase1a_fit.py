#!/usr/bin/env python3
"""
scripts/phase1a_fit.py

The lag sweep (Phase 1A) with Phase 1C and the asymptote check.

For each lag in the sweep and each of the 17 leave-one-PDB-entry-out folds, fits on the other entries and scores
the held-out entry's rows (real tokens only; centered by the protein's own per-position mean, so mu = 0):

  identity, mean, linear_diag (per-channel OLS), linear_full (A z0 + b), mlp (early-stops on two held-in
  entries only), shrink (the ONE-parameter model zhat1 = alpha * z0, alpha by least squares), and the existing
  DiT checkpoint, IN-SAMPLE, keyed "DiT[in-sample]_*" everywhere.

Reported against BOTH baselines for every lag (identity is the right baseline at short lags and the wrong one at
long lags; the mean is the reverse), with cross-protein CIs bootstrapped over the 17 PDB ENTRIES (never rows).

Phase 1C (is the gain dynamics or variance collapse?):
  C1 contraction ratio ||zhat1 - mu|| / ||z1 - mu||; C2 eigenvalue spectrum of A (linear_full); C3 improvement of
  each probe over the shrinkage model, with entry-bootstrap CI; C4 cosine(predicted displacement, true displacement)
  and the cosine of the part BEYOND shrinkage. Pre-registered kill: if full linear does not beat shrinkage by more
  than the CI, or cosine ~ 0 while MSE improves, the gain is mean reversion with no directional information.

Asymptote check (E4): at lags >> tau_int, E||z1-z0||^2 -> 2 sigma^2 and E||z1-mu||^2 -> sigma^2, so identity/mean -> 2,
probe-vs-identity -> 50%, probe-vs-mean -> 0. Also compares each protein's measured identity/mean to the parameter-free
2(1 - rho(lag)) built from the independently measured autocorrelation (acf.json).
50 ns is computed but is NOT interpretable (n_eff ~ 2 per protein); every table marks it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from analysis.probes import (fit_linear_diag, predict_linear_diag, fit_linear_full, predict_linear_full,
                             train_mlp, row_sqerr)
from analysis.stats import (entry_groups, lag_frames_to_label, time_blocks, block_len,
                            block_bootstrap_improvement, entry_bootstrap_mean, ci_excludes_zero)

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
UNINTERPRETABLE_LAG = 5000          # 50 ns: n_eff ~ 2 per protein
PROBES = ["linear_diag", "linear_full", "mlp", "shrink"]


def load_cell(cache: Path, protein: str, lag: int):
    cell = torch.load(cache / f"{protein}__lag{lag}.pt", weights_only=False)
    meta = torch.load(cache / f"{protein}__meta.pt", weights_only=False)
    mp, pos = meta["mean_pos"].float(), cell["pos"].long()
    cell["z0c"], cell["z1c"] = cell["z0"] - mp[pos], cell["z1"] - mp[pos]
    return cell


def cat(cells, key):
    return torch.cat([c[key] for c in cells], dim=0)


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def run_fold(cells, held_entry, groups, lag, device, seed):
    held = groups[held_entry]
    train_entries = [e for e in groups if e != held_entry]
    rng = np.random.default_rng(seed + lag)
    inner = set(rng.choice(train_entries, size=2, replace=False).tolist())
    train_cells = [cells[p] for e in train_entries for p in groups[e]]
    fit_cells = [cells[p] for e in train_entries if e not in inner for p in groups[e]]
    inner_cells = [cells[p] for e in train_entries if e in inner for p in groups[e]]

    z0, z1 = cat(train_cells, "z0c"), cat(train_cells, "z1c")
    mean_vec = z1.mean(0, keepdim=True)
    a, k, _ = fit_linear_diag(z0, z1)
    A, b = fit_linear_full(z0, z1)
    alpha = float((z0 * z1).sum() / (z0 * z0).sum())               # one-parameter shrinkage toward mu = 0
    eig = torch.linalg.eigvals(A.double())
    mlp, v, ep = train_mlp(cat(fit_cells, "z0c"), cat(fit_cells, "z1c"),
                           cat(inner_cells, "z0c"), cat(inner_cells, "z1c"), device=device, seed=seed)
    out, onec = {}, {}
    for p in held:
        c = cells[p]
        x, y = c["z0c"], c["z1c"]
        with torch.no_grad():
            preds = {"identity": x, "mean": mean_vec.expand_as(y),
                     "linear_diag": predict_linear_diag(x, a, k), "linear_full": predict_linear_full(x, A, b),
                     "mlp": mlp(x), "shrink": alpha * x}
        out[p] = {n: row_sqerr(pr, y).numpy() for n, pr in preds.items()}
        if c["dit_sqerr"]:
            for n, e in c["dit_sqerr"].items():
                out[p][f"DiT[in-sample]_{n}"] = e.numpy()
        u = y - x
        sse = {n: float(((preds[n] - y) ** 2).sum()) for n in PROBES}
        o = {"alpha": alpha}
        for n in PROBES:
            pr = preds[n]
            o[n] = {"C1_contraction": float(pr.norm() / (y.norm() + 1e-30)),
                    "C4_cos_displacement": cos(pr - x, u),
                    "cos_beyond_shrink": (cos(pr - preds["shrink"], y - preds["shrink"]) if n != "shrink" else None),
                    "pct_over_shrink": (100.0 * (1.0 - sse[n] / sse["shrink"]) if n != "shrink" else 0.0)}
        onec[p] = o
    return out, onec, {"mlp_epochs": ep, "inner_entries": sorted(inner), "alpha": alpha,
                       "eig_real": eig.real.numpy().tolist(), "eig_imag": eig.imag.numpy().tolist()}


def rho_at(acf_p, lag):
    taus, rho = acf_p["taus"], acf_p["rho"]
    return float(np.interp(lag, taus, rho))


def sweep(cache: Path, lags, device, out_dir: Path, seed: int, n_boot: int, acf_path: Path) -> None:
    groups = entry_groups(PILOT)
    acf = json.loads(acf_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row, per_protein, onec_all, eig_all = {}, [], {}, {}
    for lag in lags:
        cells = {p: load_cell(cache, p, lag) for p in PILOT}
        for entry in groups:
            fold, onec, info = run_fold(cells, entry, groups, lag, device, seed)
            eig_all[(lag, entry)] = (info["eig_real"], info["eig_imag"])
            for p, errs in fold.items():
                per_row[(p, lag)] = errs
                onec_all[(p, lag)] = onec[p]
                c = cells[p]
                blocks = time_blocks(c["replica"].numpy(), c["frame"].numpy(), block_len(lag, acf[p]["tau_int_frames"]))
                ident = errs["identity"]
                for name, e in errs.items():
                    if name == "identity":
                        continue
                    est, lo, hi = block_bootstrap_improvement(e, ident, blocks, n_boot=min(n_boot, 1000), seed=seed)
                    per_protein.append({"protein": p, "entry": entry, "lag": lag, "predictor": name,
                                        "mse": float(e.mean()), "mse_identity": float(ident.mean()),
                                        "mse_mean": float(errs["mean"].mean()),
                                        "pct_vs_identity": est, "ci_lo": lo, "ci_hi": hi,
                                        "pct_vs_mean": 100.0 * (1.0 - float(e.mean()) / float(errs["mean"].mean())),
                                        "n_rows": int(len(e)), "n_blocks": int(blocks.max() + 1)})
            print(f"[sweep] lag {lag_frames_to_label(lag):>6s} held-out {entry} done (mlp epochs {info['mlp_epochs']}, alpha {info['alpha']:.3f})", flush=True)
    torch.save({"per_row": per_row, "eig": eig_all}, out_dir / "per_row_sqerr.pt")
    (out_dir / "per_protein.json").write_text(json.dumps(per_protein))

    predictors = sorted({s["predictor"] for s in per_protein})
    across = []
    for lag in lags:
        for pred in predictors:
            rows = [s for s in per_protein if s["lag"] == lag and s["predictor"] == pred]
            rec = {"lag": lag, "lag_label": lag_frames_to_label(lag), "predictor": pred, "interpretable": lag != UNINTERPRETABLE_LAG}
            for key, nm in (("pct_vs_identity", "id"), ("pct_vs_mean", "mean")):
                vals = {s["protein"]: s[key] for s in rows}
                mu, lo, hi = entry_bootstrap_mean(vals, n_boot=n_boot, seed=seed)
                rec[f"{nm}_mean"], rec[f"{nm}_lo"], rec[f"{nm}_hi"] = mu, lo, hi
                rec[f"{nm}_excl0"] = ci_excludes_zero(lo, hi)
                rec[f"{nm}_n_pos"] = int(sum(v > 0 for v in vals.values()))
            across.append(rec)
    (out_dir / "across_proteins.json").write_text(json.dumps(across, indent=1))

    # -------- asymptote check (E4)
    asym = []
    for lag in lags:
        meas, pred_, id_vs_mean = {}, {}, {}
        for p in PILOT:
            r = next(s for s in per_protein if s["protein"] == p and s["lag"] == lag and s["predictor"] == "linear_diag")
            meas[p] = r["mse_identity"] / r["mse_mean"]
            pred_[p] = 2.0 * (1.0 - rho_at(acf[p], lag))
        diff = {p: meas[p] - pred_[p] for p in PILOT}
        m, lo, hi = entry_bootstrap_mean(meas, n_boot=n_boot, seed=seed)
        pm, plo, phi = entry_bootstrap_mean(pred_, n_boot=n_boot, seed=seed)
        dm, dlo, dhi = entry_bootstrap_mean(diff, n_boot=n_boot, seed=seed)
        lin = next(a for a in across if a["lag"] == lag and a["predictor"] == "linear_diag")
        asym.append({"lag": lag, "lag_label": lag_frames_to_label(lag), "interpretable": lag != UNINTERPRETABLE_LAG,
                     "id_over_mean_measured": [m, lo, hi], "id_over_mean_from_rho": [pm, plo, phi],
                     "measured_minus_from_rho": [dm, dlo, dhi], "asymptote_id_over_mean": 2.0,
                     "linear_diag_vs_identity": [lin["id_mean"], lin["id_lo"], lin["id_hi"]], "asymptote_vs_identity": 50.0,
                     "linear_diag_vs_mean": [lin["mean_mean"], lin["mean_lo"], lin["mean_hi"]], "asymptote_vs_mean": 0.0})
    (out_dir / "asymptote.json").write_text(json.dumps(asym, indent=1))

    # -------- Phase 1C summary
    onec_sum = []
    for lag in lags:
        rec = {"lag": lag, "lag_label": lag_frames_to_label(lag), "interpretable": lag != UNINTERPRETABLE_LAG,
               "alpha_mean": float(np.mean([onec_all[(p, lag)]["alpha"] for p in PILOT]))}
        for n in PROBES:
            for key in ("C1_contraction", "C4_cos_displacement", "cos_beyond_shrink", "pct_over_shrink"):
                vals = {p: onec_all[(p, lag)][n][key] for p in PILOT}
                if any(v is None for v in vals.values()):
                    continue
                mu, lo, hi = entry_bootstrap_mean(vals, n_boot=n_boot, seed=seed)
                rec[f"{n}.{key}"] = [mu, lo, hi]
        er = np.concatenate([np.array(eig_all[(lag, e)][0]) for e in groups]); ei = np.concatenate([np.array(eig_all[(lag, e)][1]) for e in groups])
        mod = np.hypot(er, ei)
        rec["eig"] = {"max_abs": float(mod.max()), "median_real": float(np.median(er)), "p10_real": float(np.percentile(er, 10)),
                      "p90_real": float(np.percentile(er, 90)), "frac_abs_gt_0.9": float((mod > 0.9).mean()),
                      "frac_complex": float((np.abs(ei) > 0.05).mean()), "mean_real": float(er.mean())}
        onec_sum.append(rec)
    (out_dir / "onec.json").write_text(json.dumps(onec_sum, indent=1))

    # ---------------- printing
    tag = lambda lag: " (50 ns: n_eff~2/protein, NOT interpretable)" if lag == UNINTERPRETABLE_LAG else ""
    for nm, title in (("id", "IDENTITY"), ("mean", "MEAN")):
        print(f"\n=== improvement over {title} (%), mean over 19 proteins, 95% CI over 17 PDB entries; * = CI excludes 0 ===")
        print(f"{'predictor':28s}" + "".join(f"{lag_frames_to_label(l):>26s}" for l in lags))
        for pred in predictors:
            row = f"{pred:28s}"
            for lag in lags:
                r = next(a for a in across if a["lag"] == lag and a["predictor"] == pred)
                row += f"{r[nm + '_mean']:+8.2f} [{r[nm + '_lo']:+7.2f},{r[nm + '_hi']:+7.2f}]{'*' if r[nm + '_excl0'] else ' '}"
            print(row)
        if UNINTERPRETABLE_LAG in lags:
            print("(50 ns column: n_eff ~ 2 per protein, NOT interpretable)")
    print("\n=== E4 ASYMPTOTE CHECK ===")
    print(f"{'lag':>8s} {'id/mean measured':>22s} {'2(1-rho) from ACF':>22s} {'diff [CI]':>24s} {'lin vs id (->50)':>20s} {'lin vs mean (->0)':>20s}")
    for r in asym:
        a, b, c = r["id_over_mean_measured"], r["id_over_mean_from_rho"], r["measured_minus_from_rho"]
        print(f"{r['lag_label']:>8s} {a[0]:7.3f} [{a[1]:5.3f},{a[2]:5.3f}] {b[0]:7.3f} [{b[1]:5.3f},{b[2]:5.3f}] {c[0]:+7.3f} [{c[1]:+6.3f},{c[2]:+6.3f}] "
              f"{r['linear_diag_vs_identity'][0]:+7.2f}% {r['linear_diag_vs_mean'][0]:+8.2f}%{'' if r['interpretable'] else '   <-- not interpretable'}")
    print("\n=== PHASE 1C (probes only; the DiT learned nothing and is not analyzed for contraction) ===")
    for rec in onec_sum:
        print(f"-- {rec['lag_label']}{tag(rec['lag'])}: alpha={rec['alpha_mean']:.3f}  eig: max|l|={rec['eig']['max_abs']:.3f} "
              f"median Re={rec['eig']['median_real']:.3f} frac|l|>0.9={rec['eig']['frac_abs_gt_0.9']:.3f} frac complex={rec['eig']['frac_complex']:.3f}")
        for n in ("linear_diag", "linear_full", "mlp", "shrink"):
            g = lambda k: rec.get(f"{n}.{k}")
            s = f"   {n:12s} C1={g('C1_contraction')[0]:.3f}  C4 cos(disp)={g('C4_cos_displacement')[0]:+.3f}"
            if n != "shrink":
                b = g("cos_beyond_shrink"); o = g("pct_over_shrink")
                s += f"  cos beyond shrink={b[0]:+.3f}  over shrinkage {o[0]:+6.2f}% [{o[1]:+6.2f},{o[2]:+6.2f}]{'*' if o[1] > 0 or o[2] < 0 else ' '}"
            print(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_cache"))
    ap.add_argument("--acf", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/acf.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/sweep"))
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 10, 100, 1000, 5000])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    sweep(args.cache, args.lags, args.device, args.out_dir, args.seed, args.n_boot, args.acf)


if __name__ == "__main__":
    main()
