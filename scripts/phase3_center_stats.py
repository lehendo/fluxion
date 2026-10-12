#!/usr/bin/env python3
"""
scripts/phase3_center_stats.py

Per-protein per-position mean of the NORMALIZED latents (every 10th frame, all replicas), the centering the paper's probes use
(F6 in the log: this is the protein's own mean, i.e. transductive for held-out proteins, exactly as in the probe comparisons).
Writes <out-dir>/<protein>.pt = float32 [L_bucket, d]. Also records the 3a train / held-out split.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from data_pipeline.dataset import LatentDataset
from training.phase3_data import per_position_mean, split_proteins

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase3/mean_pos"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    train, held = split_proteins(PILOT)
    for p in PILOT:
        ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
        m = per_position_mean(ds)
        torch.save(m, a.out_dir / f"{p}.pt")
        print(f"[center] {p}: {tuple(m.shape)} from {len(range(0, len(ds), 10))} frames", flush=True)
    (a.out_dir.parent / "split.json").write_text(json.dumps({"train": train, "held_out": held, "excluded": ["1b25_D"], "seed": 0}, indent=1))
    print("train :", train, "\nheld  :", held)


if __name__ == "__main__":
    main()
