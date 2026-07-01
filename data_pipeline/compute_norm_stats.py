#!/usr/bin/env python3
"""
data_pipeline/compute_norm_stats.py

Computes per-channel mean/std of the Chai-1 single-token latents across the
extracted training set, and writes them to <latents-dir>/norm_stats.pt.

Why this matters
----------------
FlowMatcher regresses an MSE loss directly against raw single-token latents.
Those latents are unnormalized transformer trunk activations, which are well
known to have wildly uneven per-channel variance (a handful of "massive
activation" channels can dominate the rest by orders of magnitude). An MSE
loss computed in that raw space is dominated by the few high-variance
channels — gradient signal for everything else is drowned out, so training
can plateau early after only matching the dominant channels' behavior.
Normalizing every channel to zero mean / unit variance before computing the
loss balances the objective across channels. This is standard practice in
essentially every latent-space diffusion/flow paper (e.g. Stable Diffusion's
latent scaling factor).

Usage
-----
Run once per latents_dir before training (covers every protein present, or
an explicit subset):

    python data_pipeline/compute_norm_stats.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS"      # optional; default = all proteins found
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch


def compute_stats(
    latents_dir: Path,
    protein_ids: list[str] | None = None,
) -> dict[str, torch.Tensor | int]:
    """
    Streams every "single" HDF5 dataset under latents_dir and accumulates
    per-channel sum / sum-of-squares in float64 for numerical stability.

    Returns:
        {"mean": float32 [d_single], "std": float32 [d_single],
         "n_frames": int, "d_single": int}
    """
    latents_dir = Path(latents_dir)
    protein_dirs = sorted(
        d for d in latents_dir.iterdir()
        if d.is_dir() and (protein_ids is None or d.name in protein_ids)
    )
    if not protein_dirs:
        raise RuntimeError(
            f"No protein directories found in {latents_dir} "
            f"(protein_ids filter: {protein_ids})"
        )

    d_single: int | None = None
    total_sum: torch.Tensor | None = None
    total_sumsq: torch.Tensor | None = None
    n_tokens = 0

    for protein_dir in protein_dirs:
        manifest_path = protein_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if protein_dir.name not in manifest:
            continue

        seen: set[str] = set()
        shards = [
            s for s in manifest[protein_dir.name]["shards"]
            if not (s["path"] in seen or seen.add(s["path"]))  # type: ignore[func-returns-value]
        ]

        for shard_info in shards:
            shard_path = protein_dir / shard_info["path"]
            if not shard_path.exists():
                continue
            with h5py.File(shard_path, "r") as hf:
                single = torch.from_numpy(hf["single"][:]).double()  # [n, L, d]

            n, L, d = single.shape
            if d_single is None:
                d_single = d
                total_sum = torch.zeros(d, dtype=torch.float64)
                total_sumsq = torch.zeros(d, dtype=torch.float64)
            elif d != d_single:
                raise ValueError(
                    f"{shard_path} has d_single={d}, expected {d_single} "
                    "(mismatched latent dimension across shards)"
                )

            flat = single.reshape(-1, d)  # [n*L, d]
            total_sum += flat.sum(dim=0)
            total_sumsq += (flat ** 2).sum(dim=0)
            n_tokens += flat.shape[0]

        print(f"[norm_stats] {protein_dir.name}: running total {n_tokens:,} tokens", flush=True)

    if n_tokens == 0:
        raise RuntimeError(f"No frames found under {latents_dir} to compute stats from.")

    mean = total_sum / n_tokens
    var = total_sumsq / n_tokens - mean ** 2
    var = var.clamp(min=1e-12)  # guard against tiny negative variance from float roundoff
    std = var.sqrt()

    return {
        "mean": mean.float(),
        "std": std.float(),
        "n_frames": n_tokens,
        "d_single": d_single,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--proteins", type=str, default=None,
                         help="Comma-separated protein IDs (default: all proteins found)")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output path (default: <latents-dir>/norm_stats.pt)")
    args = parser.parse_args()

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None
    stats = compute_stats(args.latents_dir, protein_ids)

    out_path = args.out or (args.latents_dir / "norm_stats.pt")
    torch.save(stats, out_path)

    print(f"\n[norm_stats] d_single={stats['d_single']}  n_frames={stats['n_frames']:,}")
    print(f"[norm_stats] mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"[norm_stats] std  range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
    print(f"[norm_stats] Saved → {out_path}")


if __name__ == "__main__":
    main()
