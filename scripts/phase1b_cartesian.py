#!/usr/bin/env python3
"""
scripts/phase1b_cartesian.py   (B1: is the pretrained trunk contributing anything beyond coordinates?)

Per protein, leave-one-REPLICA-out (3 folds), frames at 100 ps stride (ATLAS analysis archives; analysis frame k <-> latent
frame 10k). Two representations of the SAME frames, reduced to the SAME dimension k = min(3*n_res, 384):
  chai : normalized Chai-1 single latents, real tokens flattened, PCA to k
  cart : C-alpha coordinates superposed on replica-1 frame 0, flattened, PCA to k
PCA is fit on the two TRAINING replicas of each fold only (kernel/Gram form: no giant covariance matrices), then all frames
are projected. Identical probe suite on the k-vectors at lags {100 ps, 1 ns, 10 ns, 50 ns(uninterpretable)}:
identity, mean, linear_diag, linear_full, shrink, mlp (early-stops on one held-in replica). Output per protein/lag/representation:
% improvement over identity and over the mean (mean over the 3 folds). The cross-protein comparison (paired Chai - Cartesian
difference, CI over PDB entries) is in scripts/phase1b_report.py.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MDAnalysis as mda
import numpy as np
import torch
from analysis.geometry import kabsch_align, pca_scores
from analysis.mask import n_residues
from analysis.probes import (fit_ridge_diag, select_ridge_strength, fit_linear_diag, predict_linear_diag, fit_linear_full, predict_linear_full, train_mlp, row_sqerr)
from data_pipeline.dataset import LatentDataset, _REPLICA_RE

LAGS = {"100ps": 1, "1ns": 10, "10ns": 100, "50ns": 500}          # in analysis steps


def probe_suite(tr_pairs, va_pairs, te_pairs, device="cpu", seed=0):
    (a0, a1), (v0, v1), (t0, t1) = tr_pairs, va_pairs, te_pairs
    z0, z1 = torch.cat([a0, v0]), torch.cat([a1, v1])
    mean_vec = z1.mean(0, keepdim=True)
    a, k, _ = fit_linear_diag(z0, z1)
    A, b = fit_linear_full(z0, z1)
    alpha = float((z0 * z1).sum() / (z0 * z0).sum())
    c_ridge = select_ridge_strength(a0, a1, v0, v1)                      # POST HOC diagnostic (log 2026-10-12): ridge per-channel linear
    ar, kr = fit_ridge_diag(z0, z1, c_ridge)
    mlp, _, _ = train_mlp(a0, a1, v0, v1, hidden=128, batch_size=128, max_epochs=40, patience=5, device=device, seed=seed)
    with torch.no_grad():
        preds = {"identity": t0, "mean": mean_vec.expand_as(t1), "linear_diag": predict_linear_diag(t0, a, k),
                 "linear_full": predict_linear_full(t0, A, b), "shrink": alpha * t0, "mlp": mlp(t0),
                 "linear_diag_ridge": ar + kr * t0}
    out = {n: float(((p - t1) ** 2).mean()) for n, p in preds.items()}
    out["_ridge_c"] = c_ridge
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--analysis-root", type=Path, default=Path("/scratch/arjunc4/atlas_analysis"))
    ap.add_argument("--metadata-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/b1"))
    a = ap.parse_args()
    p, d = a.protein, a.analysis_root / a.protein
    n_res = n_residues(a.metadata_json, p)
    k = min(3 * n_res, 384)
    pdb = next(x for x in sorted(d.rglob("*.pdb")) if "_R" not in x.stem)
    # ---- coordinates, aligned to replica-1 frame 0
    cart, ref = {}, None
    for rep in (1, 2, 3):
        u = mda.Universe(str(pdb), str(d / f"{p}_R{rep}.xtc")); ca = u.select_atoms("name CA")
        X = np.stack([ca.positions.copy() for _ in u.trajectory])             # [1001, n_res, 3]
        assert X.shape[1] == n_res, f"{p}: {X.shape[1]} C-alpha vs {n_res} residues"
        ref = X[0] if ref is None else ref
        cart[rep] = torch.from_numpy(kabsch_align(X, ref).reshape(len(X), -1)).float()
    # ---- latents at frame 10k
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
    sl = {}
    for i, (path, _) in enumerate(ds.index_map):
        c = int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))); sl.setdefault(c, [i, i + 1])[1] = i + 1
    chai = {rep: torch.stack([ds._load_frame(sl[rep][0] + 10 * kk)[:n_res].flatten() for kk in range(1001)]) for rep in (1, 2, 3)}
    print(f"[b1] {p}: n_res={n_res} k={k} chai dim {chai[1].shape[1]} cart dim {cart[1].shape[1]}", flush=True)

    res = {"protein": p, "n_res": n_res, "k": k, "lags": {}}
    for name, X in (("chai", chai), ("cart", cart)):
        folds = []
        for held in (1, 2, 3):
            tr = [r for r in (1, 2, 3) if r != held]
            Xtr = torch.cat([X[r] for r in tr]); Xall = {r: X[r] for r in (1, 2, 3)}
            allx = torch.cat([Xall[r] for r in (1, 2, 3)])
            S = pca_scores(Xtr, allx, k)
            sc = {r: S[(i * 1001):((i + 1) * 1001)] for i, r in enumerate((1, 2, 3))}
            per_lag = {}
            for lname, L in LAGS.items():
                pairs = lambda r: (sc[r][:-L], sc[r][L:])
                per_lag[lname] = probe_suite(pairs(tr[0]), pairs(tr[1]), pairs(held))
            folds.append(per_lag)
        for lname in LAGS:
            res["lags"].setdefault(lname, {})[name] = {
                pred: {"mse": float(np.mean([f[lname][pred] for f in folds])),
                       "pct_vs_identity": float(100 * (1 - np.mean([f[lname][pred] for f in folds]) / np.mean([f[lname]["identity"] for f in folds]))),
                       "pct_vs_mean": float(100 * (1 - np.mean([f[lname][pred] for f in folds]) / np.mean([f[lname]["mean"] for f in folds])))}
                for pred in folds[0][lname] if not pred.startswith("_")}
            res["lags"][lname][name]["_ridge_c_per_fold"] = [f[lname]["_ridge_c"] for f in folds]
        print(f"[b1] {p}: {name} done", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{p}.json").write_text(json.dumps(res))
    for lname in LAGS:
        c, g = res["lags"][lname]["chai"], res["lags"][lname]["cart"]
        print(f"[b1] {p} {lname:>5s}: linear_full vs identity  chai {c['linear_full']['pct_vs_identity']:+7.2f}%  cart {g['linear_full']['pct_vs_identity']:+7.2f}%  | "
              f"id/mean chai {c['identity']['mse'] / c['mean']['mse']:.3f} cart {g['identity']['mse'] / g['mean']['mse']:.3f}", flush=True)


if __name__ == "__main__":
    main()
