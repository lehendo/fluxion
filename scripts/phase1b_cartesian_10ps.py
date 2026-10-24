#!/usr/bin/env python3
"""Chai-1 versus C-alpha PCA probes at 10 ps stride (paper Section 4.2, Appendix A.5)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import torch
from fluxion.analysis.geometry import kabsch_align, pca_fit, pca_project
from fluxion.analysis.mask import n_residues
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE
from phase1b_cartesian import probe_suite

LAGS = {"10ps": 1, "100ps": 10, "1ns": 100, "10ns": 1000}          # in 10 ps steps
N_FRAMES, FIT_STRIDE, CHUNK = 10001, 10, 400


def project_all(frames_fn, fit_idx, folds, k):
    """frames_fn(lo, hi, rep) -> [hi-lo, D] float tensor. Fits one PCA per fold on the strided TRAINING frames, projects every frame of every replica
    onto every fold's axes in chunks. Returns scores[fold][rep] -> [N_FRAMES, k]."""
    fit = {r: torch.cat([frames_fn(j, j + 1, r) for j in fit_idx]) for r in (1, 2, 3)}
    axes = {h: pca_fit(torch.cat([fit[r] for r in (1, 2, 3) if r != h]), k) for h in folds}
    del fit
    out = {h: {r: [] for r in (1, 2, 3)} for h in folds}
    for r in (1, 2, 3):
        for lo in range(0, N_FRAMES, CHUNK):
            X = frames_fn(lo, min(lo + CHUNK, N_FRAMES), r)
            for h in folds:
                out[h][r].append(pca_project(X, *axes[h]))
    return {h: {r: torch.cat(out[h][r]) for r in (1, 2, 3)} for h in folds}


def load_frames(p: str, full_root: Path, metadata_json: Path, latents_dir: Path):
    """Returns (n_res, k, chai_frames, cart_frames): functions (lo, hi, rep) -> [hi-lo, D] float tensors (latent frame j <-> 10 ps frame j)."""
    d = full_root / p
    n_res = n_residues(metadata_json, p); k = min(3 * n_res, 384)
    pdb = sorted(x for x in d.rglob("*.pdb"))[0]
    cart, ref = {}, None
    for rep in (1, 2, 3):
        u = mda.Universe(str(pdb), str(sorted(d.rglob(f"*R{rep}*.xtc"))[0])); ca = u.select_atoms("name CA")
        X = np.stack([ca.positions.copy() for _ in u.trajectory])
        assert X.shape[0] == N_FRAMES and X.shape[1] == n_res, (p, X.shape)
        ref = X[0] if ref is None else ref
        cart[rep] = torch.from_numpy(kabsch_align(X, ref).reshape(N_FRAMES, -1)).float()
    ns = torch.load(latents_dir / "norm_stats.pt", weights_only=True)
    ds = LatentDataset(latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
    sl = {}
    for i, (path, _) in enumerate(ds.index_map):
        c = int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))); sl.setdefault(c, [i, i + 1])[1] = i + 1
    chai_frames = lambda lo, hi, rep: torch.stack([ds._load_frame(sl[rep][0] + j)[:n_res].flatten() for j in range(lo, hi)])
    cart_frames = lambda lo, hi, rep: cart[rep][lo:hi]
    return n_res, k, chai_frames, cart_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--full-root", type=Path, default=ATLAS_10PS)
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--verification", type=Path, default=DATA_ROOT / "phase1_results/atlas_10ps_verification.json")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "phase1_results/b1_10ps")
    a = ap.parse_args()
    p, d = a.protein, a.full_root / a.protein
    v = json.loads(a.verification.read_text())["per_protein"].get(p, {})
    if not v.get("pass"):
        print(f"[b1-10ps] {p}: SKIPPED (did not pass (i)-(iii) of the 10 ps verification: {v})", flush=True); return
    n_res, k, chai_frames, cart_frames = load_frames(p, a.full_root, a.metadata_json, a.latents_dir)
    fit_idx = list(range(0, N_FRAMES, FIT_STRIDE))
    print(f"[b1-10ps] {p}: n_res={n_res} k={k}", flush=True)
    res = {"protein": p, "n_res": n_res, "k": k, "lags": {}}
    for name, fn in (("chai", chai_frames), ("cart", cart_frames)):
        S = project_all(fn, fit_idx, (1, 2, 3), k)
        folds = []
        for held in (1, 2, 3):
            tr = [r for r in (1, 2, 3) if r != held]; sc = S[held]
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
        print(f"[b1-10ps] {p}: {name} done", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{p}.json").write_text(json.dumps(res))
    for lname in LAGS:
        c, g = res["lags"][lname]["chai"], res["lags"][lname]["cart"]
        print(f"[b1-10ps] {p} {lname:>5s}: shrink vs identity chai {c['shrink']['pct_vs_identity']:+7.2f}%  cart {g['shrink']['pct_vs_identity']:+7.2f}%  | "
              f"id/mean chai {c['identity']['mse'] / c['mean']['mse']:.3f} cart {g['identity']['mse'] / g['mean']['mse']:.3f}", flush=True)


if __name__ == "__main__":
    main()
