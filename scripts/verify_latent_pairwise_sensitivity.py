#!/usr/bin/env python3
"""
scripts/verify_latent_pairwise_sensitivity.py

Pairwise (Mantel-style) counterpart to verify_latent_frame_sensitivity.py.

Why this exists
----------------
verify_latent_frame_sensitivity.py compares every frame's latent deviation
to its Cα RMSD *relative to frame 0 specifically* -- a "star" design. Real MD
trajectories diffuse rather than moving monotonically away from one anchor
frame, so a protein can revisit a similar RMSD-to-frame-0 from a genuinely
different conformation. That structural blind spot can produce a weak
star-correlation even when the latent perfectly tracks conformation.

This script instead compares the FULL pairwise Cα-RMSD matrix across a
subsample of frames to the full pairwise latent-distance matrix -- the
standard way (a Mantel test) to check whether two representations of the
same objects preserve the same relative structure, without privileging one
reference frame. Run on the 2026-08-16 sweep's "strong" vs "weak" proteins
to see whether the star-frame-0 metric explains the split, before treating
it as evidence of a real per-protein gap in the coordinate-injection fix.

Usage
-----
python scripts/verify_latent_pairwise_sensitivity.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents_fixed \
    --atlas-raw   /scratch/arjunc4/atlas_raw \
    --protein-id  1a62_A \
    --n-load      1500 \
    --n-sample    50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_latents(latents_dir: Path, protein_id: str, n_frames: int):
    """Identical loader to verify_latent_frame_sensitivity.py."""
    import h5py  # local import: keeps --help working without h5py installed

    protein_dir = latents_dir / protein_id
    with open(protein_dir / "manifest.json") as f:
        manifest = json.load(f)

    seen: set[str] = set()
    shards = [
        s for s in manifest[protein_id]["shards"]
        if not (s["path"] in seen or seen.add(s["path"]))  # type: ignore[func-returns-value]
    ]
    shards.sort(key=lambda s: s["path"])

    singles: list[np.ndarray] = []
    frame_idxs: list[int] = []
    replica_ids: list[int] = []

    for s in shards:
        path = protein_dir / s["path"]
        if not path.exists():
            continue
        m = re.search(r"_R(\d+)_", Path(path).name)
        replica = int(m.group(1)) if m else 1
        with h5py.File(path, "r") as h5:
            arr = np.asarray(h5["single"])
            fidx = np.asarray(h5["frame_idx"])
        take = min(len(arr), n_frames - len(singles))
        singles.append(arr[:take].astype(np.float32))
        frame_idxs.extend(int(x) for x in fidx[:take])
        replica_ids.extend([replica] * take)
        if len(frame_idxs) >= n_frames:
            break

    if not singles:
        raise RuntimeError(f"No shards found for {protein_id} under {latents_dir}")
    return np.concatenate(singles, axis=0), frame_idxs, replica_ids


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt(((a @ R - b) ** 2).sum(-1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--atlas-raw", type=Path, required=True)
    ap.add_argument("--protein-id", type=str, required=True)
    ap.add_argument("--n-load", type=int, default=1500,
                     help="How many leading frames to load from disk before subsampling.")
    ap.add_argument("--n-sample", type=int, default=50,
                     help="How many frames to evenly subsample for the pairwise matrices.")
    args = ap.parse_args()

    z, frame_idxs, replica_ids = load_latents(
        args.latents_dir, args.protein_id, args.n_load)
    n_total = len(frame_idxs)
    if n_total <= args.n_sample:
        sample_idx = np.arange(n_total)
    else:
        sample_idx = np.unique(
            np.linspace(0, n_total - 1, args.n_sample).round().astype(int))

    z_s = z[sample_idx]
    fidx_s = [frame_idxs[i] for i in sample_idx]
    rep_s = [replica_ids[i] for i in sample_idx]
    n = len(sample_idx)
    print(f"[verify-pairwise] {args.protein_id}: sampled {n} frames "
          f"(replicas present: {sorted(set(rep_s))}, "
          f"frame idx range [{min(fidx_s)}, {max(fidx_s)}])")

    import MDAnalysis as mda

    pdb = args.atlas_raw / args.protein_id / f"{args.protein_id}.pdb"

    # Group by replica so each trajectory is opened only once.
    coords_by_pos: dict[int, np.ndarray] = {}
    for rep in sorted(set(rep_s)):
        xtc = args.atlas_raw / args.protein_id / f"{args.protein_id}_prod_R{rep}_fit.xtc"
        u = mda.Universe(str(pdb), str(xtc))
        ca = u.select_atoms("name CA")
        for pos, (f, r) in enumerate(zip(fidx_s, rep_s)):
            if r != rep:
                continue
            u.trajectory[f]
            coords_by_pos[pos] = ca.positions.copy()

    coords = [coords_by_pos[i] for i in range(n)]

    # Pairwise Cα RMSD matrix (Kabsch-aligned) and pairwise latent L2 distance.
    rmsd_pairs: list[float] = []
    latent_pairs: list[float] = []
    zf = z_s.reshape(n, -1)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd_pairs.append(kabsch_rmsd(coords[i], coords[j]))
            latent_pairs.append(float(np.linalg.norm(zf[i] - zf[j])))

    rmsd_arr = np.asarray(rmsd_pairs)
    latent_arr = np.asarray(latent_pairs)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(rmsd_arr, latent_arr)

    print(f"[verify-pairwise] {len(rmsd_arr)} frame pairs")
    print(f"[verify-pairwise] pairwise Ca RMSD   : min={rmsd_arr.min():.3f}  "
          f"max={rmsd_arr.max():.3f}  mean={rmsd_arr.mean():.3f} A")
    print(f"[verify-pairwise] pairwise latent L2  : min={latent_arr.min():.3f}  "
          f"max={latent_arr.max():.3f}  mean={latent_arr.mean():.3f}")
    print(f"[verify-pairwise] Spearman(pairwise RMSD, pairwise latent dist) = "
          f"{rho:.4f}  (p={pval:.3g})")

    print()
    print("-" * 72)
    if not np.isfinite(rho):
        verdict = "UNDEFINED (degenerate pairwise data -- check sampling)."
    elif abs(rho) >= 0.3 and pval < 0.05:
        verdict = ("Latents PRESERVE pairwise conformational structure -- real signal, "
                   "consistent across the sampled frames regardless of any single "
                   "reference frame.")
    else:
        verdict = ("Latents do NOT track pairwise conformational structure even under "
                   "the pairwise test -- this is NOT explained by the star-frame-0 "
                   "metric's blind spot; treat as a real gap for this protein.")
    print(f"[verdict] {verdict}")
    print("-" * 72)


if __name__ == "__main__":
    main()
