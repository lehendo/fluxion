#!/usr/bin/env python3
"""Template-feature PCA control for the 10 ps comparison (paper Appendix A.5)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, sys
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import torch
from fluxion.analysis.mask import n_residues
from fluxion.analysis.template_features import template_features
from phase1b_cartesian import probe_suite
from phase1b_cartesian_10ps import FIT_STRIDE, LAGS, N_FRAMES, project_all


def backbone_arrays(universe, n_res):
    sel = {}
    for name in ("N", "CA", "C"):
        sel[name] = universe.select_atoms(f"protein and name {name}")
        assert len(sel[name]) == n_res, (name, len(sel[name]), n_res)
    cb = [r.atoms.select_atoms("name CB")[0] if len(r.atoms.select_atoms("name CB")) else r.atoms.select_atoms("name CA")[0] for r in universe.select_atoms("protein").residues]
    assert len(cb) == n_res
    return sel["N"], sel["CA"], sel["C"], mda.AtomGroup(cb)


def frames_to_arrays(universe, groups):
    out = [[] for _ in groups]
    for ts in universe.trajectory:
        for o, g in zip(out, groups):
            o.append(g.positions.copy())
    return [np.stack(o).astype(np.float32) for o in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--full-root", type=Path, default=ATLAS_10PS); ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--verification", type=Path, default=DATA_ROOT / "phase1_results/atlas_10ps_verification.json")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "phase1_results/b1_10ps_tf")
    a = ap.parse_args(); p = a.protein; d = a.full_root / p
    if not json.loads(a.verification.read_text())["per_protein"].get(p, {}).get("pass"):
        print(f"[tf] {p}: SKIPPED (did not pass the 10 ps verification)", flush=True); return
    n_res = n_residues(a.metadata_json, p); k = min(3 * n_res, 384)
    pdb = sorted(d.rglob("*.pdb"))[0]; arrays = {}
    for rep in (1, 2, 3):
        u = mda.Universe(str(pdb), str(sorted(d.rglob(f"*R{rep}*.xtc"))[0])); arrays[rep] = frames_to_arrays(u, backbone_arrays(u, n_res))
        assert arrays[rep][0].shape[0] == N_FRAMES
    def raw_frames(lo, hi, rep):
        return template_features(*[x[lo:hi] for x in arrays[rep]])
    fit_idx = list(range(0, N_FRAMES, FIT_STRIDE))
    # block scales from the fit frames of all replicas (a fixed scale; the PCA itself is fit per fold on the training replicas)
    dd, uu = zip(*[raw_frames(j, j + 1, r) for r in (1, 2, 3) for j in fit_idx[::5]]); dd, uu = torch.cat(dd), torch.cat(uu)
    sd, su = float(dd.var(0).sum().sqrt()), float(uu.var(0).sum().sqrt()); print(f"[tf] {p}: n_res={n_res} k={k} features {dd.shape[1]} + {uu.shape[1]} block scales {sd:.3f}/{su:.3f}", flush=True); del dd, uu
    def tf_frames(lo, hi, rep):
        dist, unit = raw_frames(lo, hi, rep)
        return torch.cat([dist / sd, unit / su], 1)
    S = project_all(tf_frames, fit_idx, (1, 2, 3), k)
    res = {"protein": p, "n_res": n_res, "k": k, "lags": {}}
    folds = []
    for held in (1, 2, 3):
        tr = [r for r in (1, 2, 3) if r != held]; sc = S[held]; per_lag = {}
        for lname, L in LAGS.items():
            pairs = lambda r: (sc[r][:-L], sc[r][L:]); per_lag[lname] = probe_suite(pairs(tr[0]), pairs(tr[1]), pairs(held))
        folds.append(per_lag)
    for lname in LAGS:
        res["lags"].setdefault(lname, {})["tf"] = {
            pred: {"mse": float(np.mean([f[lname][pred] for f in folds])), "pct_vs_identity": float(100 * (1 - np.mean([f[lname][pred] for f in folds]) / np.mean([f[lname]["identity"] for f in folds]))),
                   "pct_vs_mean": float(100 * (1 - np.mean([f[lname][pred] for f in folds]) / np.mean([f[lname]["mean"] for f in folds])))} for pred in folds[0][lname] if not pred.startswith("_")}
    a.out_dir.mkdir(parents=True, exist_ok=True); (a.out_dir / f"{p}.json").write_text(json.dumps(res))
    for lname in LAGS:
        t = res["lags"][lname]["tf"]; print(f"[tf] {p} {lname:>5s}: shrink {t['shrink']['pct_vs_identity']:+7.2f}%  linear_diag {t['linear_diag']['pct_vs_identity']:+7.2f}%", flush=True)


if __name__ == "__main__":
    main()
