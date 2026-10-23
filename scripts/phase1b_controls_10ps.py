#!/usr/bin/env python3
"""Whitening, no-intercept and component-band controls for the 10 ps comparison (paper Appendix A.5)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from fluxion.analysis.b1_controls import pct_vs_identity, per_dim_mse
from phase1b_cartesian_10ps import FIT_STRIDE, N_FRAMES, load_frames, project_all

LAGS = {"10ps": 1, "100ps": 10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--full-root", type=Path, default=ATLAS_10PS)
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--b1-dir", type=Path, default=DATA_ROOT / "phase1_results/b1_10ps")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "phase1_results/b1_10ps_controls")
    a = ap.parse_args()
    p = a.protein
    n_res, k, chai_frames, cart_frames = load_frames(p, a.full_root, a.metadata_json, a.latents_dir)
    fit_idx = list(range(0, N_FRAMES, FIT_STRIDE))
    ref = json.loads((a.b1_dir / f"{p}.json").read_text())
    out = {"protein": p, "n_res": n_res, "k": k, "lags": {}, "spectrum": {}}
    for name, fn in (("chai", chai_frames), ("cart", cart_frames)):
        S = project_all(fn, fit_idx, (1, 2, 3), k)
        stats = {lname: {"raw": [], "white": []} for lname in LAGS}; var_by_fold = []
        for held in (1, 2, 3):
            tr = [r for r in (1, 2, 3) if r != held]; sc = S[held]
            std = torch.cat([sc[r] for r in tr]).std(0)
            var_by_fold.append((std ** 2).numpy())
            for lname, L in LAGS.items():
                pair = lambda r: (sc[r][:-L], sc[r][L:])
                train = (torch.cat([pair(tr[0])[0], pair(tr[1])[0]]), torch.cat([pair(tr[0])[1], pair(tr[1])[1]]))
                stats[lname]["raw"].append(per_dim_mse(train, pair(held)))
                stats[lname]["white"].append(per_dim_mse(train, pair(held), whiten_std=std))
        out["spectrum"][name] = np.mean(var_by_fold, 0).tolist()
        for lname in LAGS:
            d = out["lags"].setdefault(lname, {}).setdefault(name, {})
            for view in ("raw", "white"):
                folds = stats[lname][view]
                d[view] = {"per_dim_mse": {pred: np.mean([f[pred] for f in folds], 0).tolist() for pred in folds[0]},
                           "pct_vs_identity": {pred: pct_vs_identity(folds, pred) for pred in folds[0]}}
            for pred, key in (("linear_diag", "linear_diag"), ("shrink", "shrink")):         # consistency with the main B1-10ps run
                mine, theirs = d["raw"]["pct_vs_identity"][pred], ref["lags"][lname][name][key]["pct_vs_identity"]
                assert abs(mine - theirs) < 0.05, f"{p} {name} {lname} {pred}: controls {mine:.3f} vs b1_10ps {theirs:.3f}"
        print(f"[b1-controls] {p} {name}: consistency with b1_10ps OK; raw linear_diag/noint/shrink at 10 ps = "
              f"{out['lags']['10ps'][name]['raw']['pct_vs_identity']['linear_diag']:+.2f}/{out['lags']['10ps'][name]['raw']['pct_vs_identity']['noint']:+.2f}/{out['lags']['10ps'][name]['raw']['pct_vs_identity']['shrink']:+.2f}; "
              f"whitened {out['lags']['10ps'][name]['white']['pct_vs_identity']['linear_diag']:+.2f}/{out['lags']['10ps'][name]['white']['pct_vs_identity']['noint']:+.2f}/{out['lags']['10ps'][name]['white']['pct_vs_identity']['shrink']:+.2f}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{p}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
