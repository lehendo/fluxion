#!/usr/bin/env python3
"""
scripts/phase1a_fit.py

Leave-one-PDB-entry-out scoring of mean / identity / linear / MLP (and the in-sample
DiT) on the Phase 1A row cache. Everything is reported as % improvement over IDENTITY.

Modes
-----
  --step0    Reproduce the OLD split (train 16 / val 1bxy_A,1bxy_B,1btk_A, lag 1) and
             the old numbers (mean 0.01944, linear 0.001638, MLP 0.001701) AND the DiT's
             identity baseline (0.002079), under both token weightings (real-only vs.
             counting bucket padding as zero-error tokens like the DiT/old probes did).
  (default)  Full sweep: all lags, 17 entry-folds, bootstrap CIs, tables, figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from analysis.probes import (
    fit_linear_diag, predict_linear_diag, fit_linear_full, predict_linear_full,
    train_mlp, row_sqerr,
)
from analysis.stats import (
    entry_groups, pct_improvement, lag_frames_to_label, time_blocks,
    block_bootstrap_improvement, entry_bootstrap_mean, ci_excludes_zero,
)

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
OLD_VAL = ["1bxy_A", "1bxy_B", "1btk_A"]
OLD_REPORTED = {"mean": 0.01944, "linear(z0,zprev)": 0.001638, "mlp(z0,zprev)": 0.001701,
                "identity (DiT flow baseline)": 0.002079}


def load_cell(cache: Path, protein: str, lag: int):
    cell = torch.load(cache / f"{protein}__lag{lag}.pt", weights_only=False)
    meta = torch.load(cache / f"{protein}__meta.pt", weights_only=False)
    mp = meta["mean_pos"].float()
    pos = cell["pos"].long()
    cell["z0c"] = cell["z0"] - mp[pos]
    cell["z1c"] = cell["z1"] - mp[pos]
    cell["zpc"] = (cell["zprev"] - mp[pos]) if cell["zprev"] is not None else None
    return cell


def cat(cells, key):
    return torch.cat([c[key] for c in cells], dim=0)


# ---------------------------------------------------------------------------
# Step 0
# ---------------------------------------------------------------------------

def step0(cache: Path, device: str, out: Path) -> None:
    train_p = [p for p in PILOT if p not in OLD_VAL]
    tr = [load_cell(cache, p, 1) for p in train_p]
    va = [load_cell(cache, p, 1) for p in OLD_VAL]

    def pooled(per_protein_mse, weights):
        return float(sum(w * m for w, m in zip(weights, per_protein_mse)) / sum(weights))

    # per-protein MSEs on REAL tokens
    mean_vec = cat(tr, "z1c").mean(0, keepdim=True)
    a, k0, k1 = fit_linear_diag(cat(tr, "z0c"), cat(tr, "z1c"), cat(tr, "zpc"))
    a2, k02, _ = fit_linear_diag(cat(tr, "z0c"), cat(tr, "z1c"))
    res = {"identity": [], "mean": [], "linear(z0,zprev)": [], "linear(z0)": []}
    for c in va:
        res["identity"].append(row_sqerr(c["z0c"], c["z1c"]).mean().item())
        res["mean"].append(row_sqerr(mean_vec.expand_as(c["z1c"]), c["z1c"]).mean().item())
        res["linear(z0,zprev)"].append(row_sqerr(
            predict_linear_diag(c["z0c"], a, k0, c["zpc"], k1), c["z1c"]).mean().item())
        res["linear(z0)"].append(row_sqerr(predict_linear_diag(c["z0c"], a2, k02), c["z1c"]).mean().item())

    n_real = [c["n_real"] for c in va]
    L_b = [c["L_bucket"] for c in va]
    table = {}
    for name, per in res.items():
        real_only = pooled(per, n_real)                           # real tokens only
        with_pad = float(sum(nr * m for nr, m in zip(n_real, per)) / sum(L_b))  # pad = zero-error tokens
        table[name] = {"real_tokens_only": real_only, "padding_counted_as_zero_error": with_pad,
                       "per_protein_real": dict(zip(OLD_VAL, per))}
    dit = {}
    if va[0]["dit_sqerr"]:
        for k in va[0]["dit_sqerr"]:
            per = [c["dit_sqerr"][k].mean().item() for c in va]
            dit[k] = {"real_tokens_only": pooled(per, n_real),
                      "padding_counted_as_zero_error": float(sum(nr * m for nr, m in zip(n_real, per)) / sum(L_b))}
    out.write_text(json.dumps({"old_reported": OLD_REPORTED, "new_pipeline": table, "dit_in_sample": dit,
                               "val_proteins": OLD_VAL, "n_real": n_real, "L_bucket": L_b}, indent=2))

    print("\n=== STEP 0: old split, lag 1 (10 ps) ===")
    print(f"{'quantity':34s} {'old reported':>13s} {'new (pad counted)':>18s} {'new (real only)':>16s}")
    pairs = [("mean", "mean"), ("linear(z0,zprev)", "linear(z0,zprev)"),
             ("identity (DiT flow baseline)", "identity")]
    for old_k, new_k in pairs:
        t = table[new_k]
        print(f"{old_k:34s} {OLD_REPORTED[old_k]:13.6f} {t['padding_counted_as_zero_error']:18.6f} "
              f"{t['real_tokens_only']:16.6f}")
    t = table["linear(z0)"]
    print(f"{'linear(z0) only [new]':34s} {'-':>13s} {t['padding_counted_as_zero_error']:18.6f} {t['real_tokens_only']:16.6f}")
    for k, v in dit.items():
        print(f"{'DiT in-sample ' + k:34s} {'0.002080 (raw)':>13s} {v['padding_counted_as_zero_error']:18.6f} {v['real_tokens_only']:16.6f}")
    ident_pad = table["identity"]["padding_counted_as_zero_error"]
    print(f"\nidentity MSE reproduced under the DiT's weighting: {ident_pad:.6f} vs reported 0.002079 "
          f"-> ratio {ident_pad / 0.002079:.3f}")
    print(f"improvement over IDENTITY, real tokens, linear(z0,zprev): "
          f"{pct_improvement(table['linear(z0,zprev)']['real_tokens_only'], table['identity']['real_tokens_only']):.2f}%")
    print(f"improvement over IDENTITY, real tokens, linear(z0)      : "
          f"{pct_improvement(table['linear(z0)']['real_tokens_only'], table['identity']['real_tokens_only']):.2f}%")
    print(f"identity vs MEAN baseline, real tokens                  : "
          f"{pct_improvement(table['identity']['real_tokens_only'], table['mean']['real_tokens_only']):.2f}%")


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

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
    mlp, v, ep = train_mlp(cat(fit_cells, "z0c"), cat(fit_cells, "z1c"),
                           cat(inner_cells, "z0c"), cat(inner_cells, "z1c"),
                           device=device, seed=seed)
    out = {}
    for p in held:
        c = cells[p]
        x, y = c["z0c"], c["z1c"]
        with torch.no_grad():
            preds = {
                "identity": x, "mean": mean_vec.expand_as(y),
                "linear_diag": predict_linear_diag(x, a, k),
                "linear_full": predict_linear_full(x, A, b),
                "mlp": mlp(x),
            }
        out[p] = {n: row_sqerr(pr, y).numpy() for n, pr in preds.items()}
        if c["dit_sqerr"]:
            for n, e in c["dit_sqerr"].items():
                out[p][f"dit_{n}"] = e.numpy()
    return out, {"mlp_epochs": ep, "mlp_inner_val": v, "inner_entries": sorted(inner)}


def sweep(cache: Path, lags, device, out_dir: Path, seed: int, n_boot: int) -> None:
    groups = entry_groups(PILOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row: dict = {}
    summary = []
    for lag in lags:
        cells = {p: load_cell(cache, p, lag) for p in PILOT}
        for entry in groups:
            fold, info = run_fold(cells, entry, groups, lag, device, seed)
            for p, errs in fold.items():
                per_row[(p, lag)] = errs
                c = cells[p]
                blocks = time_blocks(c["replica"].numpy(), c["frame"].numpy(), max(500, lag))
                ident = errs["identity"]
                for name, e in errs.items():
                    if name == "identity":
                        continue
                    est, lo, hi = block_bootstrap_improvement(e, ident, blocks, n_boot=min(n_boot, 2000), seed=seed)
                    summary.append({"protein": p, "entry": entry, "lag": lag, "predictor": name,
                                    "mse": float(e.mean()), "mse_identity": float(ident.mean()),
                                    "pct_vs_identity": est, "ci_lo": lo, "ci_hi": hi,
                                    "n_rows": int(len(e)), "n_blocks": int(blocks.max() + 1)})
            print(f"[fit] lag {lag_frames_to_label(lag):>6s} held-out {entry}: "
                  f"mlp epochs={info['mlp_epochs']}", flush=True)
    torch.save(per_row, out_dir / "per_row_sqerr.pt")
    (out_dir / "per_protein.json").write_text(json.dumps(summary, indent=1))

    # across-protein summary with entry-level bootstrap
    across = []
    predictors = sorted({s["predictor"] for s in summary})
    for lag in lags:
        for pred in predictors:
            vals = {s["protein"]: s["pct_vs_identity"] for s in summary
                    if s["lag"] == lag and s["predictor"] == pred}
            if not vals:
                continue
            mu, lo, hi = entry_bootstrap_mean(vals, n_boot=n_boot, seed=seed)
            med, mlo, mhi = entry_bootstrap_mean(vals, n_boot=n_boot, seed=seed, stat="median")
            across.append({"lag": lag, "lag_label": lag_frames_to_label(lag), "predictor": pred,
                           "mean_pct": mu, "mean_lo": lo, "mean_hi": hi, "mean_ci_excludes_0": ci_excludes_zero(lo, hi),
                           "median_pct": med, "median_lo": mlo, "median_hi": mhi,
                           "n_proteins": len(vals), "n_entries": len(groups),
                           "n_proteins_positive": int(sum(v > 0 for v in vals.values()))})
    (out_dir / "across_proteins.json").write_text(json.dumps(across, indent=1))

    print("\n=== improvement over IDENTITY (%), mean across proteins, 95% CI over PDB entries ===")
    print(f"{'predictor':16s}" + "".join(f"{lag_frames_to_label(l):>26s}" for l in lags))
    for pred in predictors:
        row = f"{pred:16s}"
        for lag in lags:
            r = next((x for x in across if x["lag"] == lag and x["predictor"] == pred), None)
            row += (f"{r['mean_pct']:+8.2f} [{r['mean_lo']:+7.2f},{r['mean_hi']:+7.2f}]{'*' if r['mean_ci_excludes_0'] else ' '}"
                    if r else f"{'-':>26s}")
        print(row)
    print("(* = 95% CI excludes zero)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 10, 100, 1000, 5000])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.step0:
        step0(args.cache, args.device, args.out_dir / "step0.json")
    else:
        sweep(args.cache, args.lags, args.device, args.out_dir, args.seed, args.n_boot)


if __name__ == "__main__":
    main()
