#!/usr/bin/env python3
"""PCA-axis stability diagnostics across replicas (paper Appendix A.5)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from fluxion.analysis.axis_stability import axis_cos, capture_ratio, fit_axes, heldout_variance, principal_cos2
from phase1b_cartesian_10ps import FIT_STRIDE, N_FRAMES, load_frames

M_LIST = [5, 20, 60, 100, 150]
QS = [0.90, 0.95, 0.99]


def m_for_fraction(var_k: torch.Tensor, total_var: float, q: float) -> int:
    c = torch.cumsum(var_k.double(), 0) / total_var
    hit = (c >= q).nonzero()
    return int(hit[0]) + 1 if len(hit) else len(var_k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--full-root", type=Path, default=ATLAS_10PS)
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "phase1_results/b1_10ps_stability")
    a = ap.parse_args()
    p = a.protein
    n_res, k, chai_frames, cart_frames = load_frames(p, a.full_root, a.metadata_json, a.latents_dir)
    idx = list(range(0, N_FRAMES, FIT_STRIDE))
    out = {"protein": p, "n_res": n_res, "k": k, "m_list": M_LIST, "spaces": {}}
    for name, fn in (("chai", chai_frames), ("cart", cart_frames)):
        X = {r: torch.cat([fn(j, j + 1, r) for j in idx]) for r in (1, 2, 3)}
        tot = {r: float(((X[r] - X[r].mean(0, keepdim=True)) ** 2).sum() / (X[r].shape[0] - 1)) for r in X}
        sep = {r: fit_axes(X[r], k) for r in X}
        d = {"axis_cos": [], "S_m": {str(m): [] for m in M_LIST}, "S_mq": {str(q): [] for q in QS}, "heldout_ratio": [], "capture_m": {str(m): [] for m in M_LIST}, "capture_mq": {str(q): [] for q in QS}, "mq": {str(q): [] for q in QS}}
        for (i, j) in ((1, 2), (1, 3), (2, 3)):
            d["axis_cos"].append(axis_cos(sep[i][1], sep[j][1]).tolist())
            for m in M_LIST:
                d["S_m"][str(m)].append(principal_cos2(sep[i][1], sep[j][1], m) if m <= k else None)
            for q in QS:
                mq = m_for_fraction(sep[i][2], tot[i], q); d["S_mq"][str(q)].append([mq, principal_cos2(sep[i][1], sep[j][1], mq)])
        for held in (1, 2, 3):
            tr = [r for r in (1, 2, 3) if r != held]
            Xtr = torch.cat([X[r] for r in tr]); mu, U, var = fit_axes(Xtr, k)
            d["heldout_ratio"].append((heldout_variance(X[held], mu, U) / var).tolist())
            for m in M_LIST:
                d["capture_m"][str(m)].append(capture_ratio(X[held], U, m) if m <= k else None)
            tot_tr = float(((Xtr - mu) ** 2).sum() / (Xtr.shape[0] - 1))
            for q in QS:
                mq = m_for_fraction(var, tot_tr, q); d["mq"][str(q)].append(mq); d["capture_mq"][str(q)].append(capture_ratio(X[held], U, mq))
        out["spaces"][name] = d
        s100 = np.mean([v for v in d["S_m"]["100"] if v is not None]); c100 = np.mean([v for v in d["capture_m"]["100"] if v is not None])
        print(f"[stability] {p} {name}: S_100 {s100:.3f}  capture_100 {c100:.3f}  mean axis |cos| (61-137) {np.mean(np.array(d['axis_cos'])[:, 60:137]):.3f}  mean log held-out/train variance (61-137) {np.mean(np.log(np.array(d['heldout_ratio'])[:, 60:137])):+.3f}", flush=True)
        del X, sep
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{p}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
