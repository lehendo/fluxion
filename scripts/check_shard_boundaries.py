#!/usr/bin/env python3
"""Continuity of latents across shard boundaries, the input to the padding signal-to-noise estimate (paper Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    ap.add_argument("--n-inside", type=int, default=40)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    meta = json.loads(a.metadata_json.read_text())
    rng = np.random.default_rng(0)
    rows = {}
    print(f"{'protein':8s} {'n_res':>5s} {'L':>4s} {'bounds':>6s} {'inside step':>12s} {'boundary real':>14s} {'ratio':>7s} "
          f"{'boundary pad':>13s} {'#pad const changes':>19s}")
    for p in a.proteins:
        ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
        n_res = meta[p]["n_residues"]
        L = ds._load_frame(0).shape[0]
        reps: dict[int, list[int]] = {}
        for i, (path, _) in enumerate(ds.index_map):
            reps.setdefault(int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))), []).append(i)
        inside, brd_real, brd_pad = [], [], []
        for c, idx in sorted(reps.items()):
            starts = [i for i in idx if ds.index_map[i][1] == 0 and i != idx[0]]    # first frame of each later shard
            for s in starts:
                za, zb = ds._load_frame(s - 1), ds._load_frame(s)
                dlt = (zb - za) ** 2
                brd_real.append(dlt[:n_res].mean().item())
                brd_pad.append(dlt[n_res:].mean().item() if L > n_res else 0.0)
            pick = rng.choice(len(idx) - 1, size=min(a.n_inside, len(idx) - 1), replace=False)
            for k in pick:
                i = idx[k]
                if ds.index_map[i + 1][1] == 0:       # skip pairs that straddle a boundary
                    continue
                inside.append(((ds._load_frame(i + 1) - ds._load_frame(i)) ** 2)[:n_res].mean().item())
        ins, br = float(np.mean(inside)), float(np.mean(brd_real))
        n_pad_changes = int(sum(b > 1e-9 for b in brd_pad))
        rows[p] = {"n_res": n_res, "L": L, "n_boundaries": len(brd_real), "inside_step_mse": ins,
                   "boundary_real_mse": br, "ratio": br / ins, "boundary_pad_mse_mean": float(np.mean(brd_pad)),
                   "n_boundaries_with_pad_change": n_pad_changes,
                   "boundary_real_mse_all": brd_real, "boundary_pad_mse_all": brd_pad}
        print(f"{p:8s} {n_res:5d} {L:4d} {len(brd_real):6d} {ins:12.6f} {br:14.6f} {br / ins:7.2f} "
              f"{np.mean(brd_pad):13.3e} {n_pad_changes:10d}/{len(brd_pad)}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
