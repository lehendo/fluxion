#!/usr/bin/env python3
"""
scripts/b3_prefix_variance.py   (partial Phase 1B / B3, needs NO raw trajectories)

B3 asks for the 'sequence-only, no template' floor, which 'should show zero frame variance'. The PRE-FIX latents
(latents_prefix_broken_20260813) ARE that condition: the per-frame coordinate injection was inert, so the trunk
saw only the sequence. Compares, per protein, frame variance (variance across 150 frames drawn evenly from the
whole trajectory, summed over channels, averaged over REAL tokens; raw float16 values, no normalization) of the
pre-fix latents against the corrected latents. Padding is masked from extraction-time residue counts.
Also reports the mean per-token ||z_t+1 - z_t||^2 between adjacent frames in both conditions.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from analysis.mask import n_residues
from data_pipeline.dataset import LatentDataset

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def frame_stats(root: Path, p: str, n_res: int, n_frames: int):
    ds = LatentDataset(root, p, load_pair=False, norm_stats=None, lag=1)
    idx = np.linspace(0, len(ds) - 1, n_frames).astype(int)
    z = torch.stack([ds._load_frame(int(i))[:n_res] for i in idx]).double()      # [n, n_res, d]
    var = z.var(dim=0).sum(-1).mean().item()                                      # across-frame variance, real tokens
    # adjacent frames inside one shard (first 40 contiguous frames)
    c = torch.stack([ds._load_frame(i)[:n_res] for i in range(40)]).double()
    step = ((c[1:] - c[:-1]) ** 2).sum(-1).mean().item()
    return var, step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--prefix", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents_prefix_broken_20260813"))
    ap.add_argument("--metadata-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/b3_prefix_variance.json"))
    ap.add_argument("--n-frames", type=int, default=150)
    a = ap.parse_args()
    res = {}
    print(f"{'protein':8s} {'n_res':>5s} | {'frame var FIXED':>16s} {'frame var PRE-FIX':>18s} {'ratio':>8s} | {'adj step FIXED':>15s} {'adj step PRE-FIX':>17s}")
    for p in PILOT:
        n = n_residues(a.metadata_json, p)
        vf, sf = frame_stats(a.fixed, p, n, a.n_frames)
        vb, sb = frame_stats(a.prefix, p, n, a.n_frames)
        res[p] = {"n_res": n, "var_fixed": vf, "var_prefix": vb, "ratio": vf / max(vb, 1e-30), "adjstep_fixed": sf, "adjstep_prefix": sb}
        print(f"{p:8s} {n:5d} | {vf:16.4f} {vb:18.6f} {vf / max(vb, 1e-30):8.1f} | {sf:15.5f} {sb:17.8f}", flush=True)
    r = np.array([v["ratio"] for v in res.values()])
    print(f"\nframe-variance ratio (corrected / pre-fix): median {np.median(r):.1f}, min {r.min():.1f}, max {r.max():.1f}")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
