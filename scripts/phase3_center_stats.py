#!/usr/bin/env python3
"""Per-protein per-position means that center the latents for the probes and flow models (paper Section 2)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
import torch
from fluxion.data_pipeline.dataset import LatentDataset
from fluxion.training.phase3_data import per_position_mean, split_proteins

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "phase3/mean_pos")
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
