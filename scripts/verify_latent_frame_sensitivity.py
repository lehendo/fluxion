#!/usr/bin/env python3
"""Empirical check that latents respond to per-frame coordinates, the check behind the frame-variance gate (paper Section 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_latents(latents_dir: Path, protein_id: str, n_frames: int):
    """Read up to n_frames single-token latents + their (frame, replica) ids."""
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

    import re
    for s in shards:
        # Match data_pipeline/dataset.py's LatentDataset path resolution
        # exactly (protein_dir / manifest_path, NOT latents_dir / manifest_path)
        # — manifest paths are stored relative to latents_dir but shards can
        # sit one directory level deeper (a known "double-nesting" artifact,
        # see clean_latents.py's docstring). dataset.py's join happens to
        # compensate for it; this script must match dataset.py's resolution
        # exactly, since the whole point is to check what the model actually
        # trains on, not what the manifest's path field nominally implies.
        path = protein_dir / s["path"]
        if not path.exists():
            continue
        m = re.search(r"_R(\d+)_", Path(path).name)
        replica = int(m.group(1)) if m else 1
        with h5py.File(path, "r") as h5:
            arr = np.asarray(h5["single"])       # [n, L, d] float16
            fidx = np.asarray(h5["frame_idx"])   # [n]
        take = min(len(arr), n_frames - len(singles))
        singles.append(arr[:take].astype(np.float32))
        frame_idxs.extend(int(x) for x in fidx[:take])
        replica_ids.extend([replica] * take)
        if len(frame_idxs) >= n_frames:
            break

    if not singles:
        raise RuntimeError(f"No shards found for {protein_id} under {latents_dir}")
    return np.concatenate(singles, axis=0), frame_idxs, replica_ids


def coord_rmsd_to_first(atlas_raw: Path, protein_id: str,
                        frame_idxs: list[int], replica_ids: list[int]) -> np.ndarray | None:
    """Cα RMSD of each frame to the first frame, for correlation testing."""
    try:
        import MDAnalysis as mda
    except ImportError:
        return None

    pdb = atlas_raw / protein_id / f"{protein_id}.pdb"
    if not pdb.exists():
        return None

    coords: list[np.ndarray] = []
    for fidx, rep in zip(frame_idxs, replica_ids):
        xtc = atlas_raw / protein_id / f"{protein_id}_prod_R{rep}_fit.xtc"
        if not xtc.exists():
            return None
        u = mda.Universe(str(pdb), str(xtc))
        u.trajectory[fidx]
        coords.append(u.select_atoms("name CA").positions.copy())

    ref = coords[0]
    out = []
    for c in coords:
        # Kabsch-aligned RMSD
        a = c - c.mean(0)
        b = ref - ref.mean(0)
        U, _, Vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        out.append(float(np.sqrt(((a @ R - b) ** 2).sum(-1).mean())))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--atlas-raw", type=Path, default=None,
                    help="Optional: enables the latent-vs-coordinate-RMSD "
                         "correlation test, which distinguishes real "
                         "conformational signal from numerical noise.")
    ap.add_argument("--protein-id", type=str, required=True)
    ap.add_argument("--n-frames", type=int, default=64)
    args = ap.parse_args()

    z, frame_idxs, replica_ids = load_latents(
        args.latents_dir, args.protein_id, args.n_frames)
    n, L, d = z.shape
    print(f"[verify] {args.protein_id}: loaded {n} frames, L={L}, d={d}")
    print(f"[verify] frame indices: {frame_idxs[:8]}{' …' if n > 8 else ''}")

    ref = z[0]
    diffs = np.abs(z - ref).reshape(n, -1).max(axis=1)
    max_abs_diff = float(diffs.max())
    per_pos_std = z.std(axis=0)          # [L, d] across frames

    print()
    print(f"[verify] max |z_i - z_0| over all frames/positions/channels : {max_abs_diff:.3e}")
    print(f"[verify] mean across-frame std (per position/channel)       : {float(per_pos_std.mean()):.3e}")
    print(f"[verify] max  across-frame std                              : {float(per_pos_std.max()):.3e}")
    print(f"[verify] fraction of (position, channel) with std == 0      : "
          f"{float((per_pos_std == 0).mean()):.4f}")
    print(f"[verify] number of pairwise-identical frames to frame 0     : "
          f"{int((diffs == 0).sum())} / {n}")

    # Scale reference: how big is the variation relative to the signal itself?
    print(f"[verify] |z| mean magnitude                                 : {float(np.abs(z).mean()):.3e}")
    if float(np.abs(z).mean()) > 0:
        print(f"[verify] across-frame std / signal magnitude               : "
              f"{float(per_pos_std.mean()) / float(np.abs(z).mean()):.3e}")

    verdict_lines: list[str] = []
    if max_abs_diff == 0.0:
        verdict_lines.append(
            "FRAME-INVARIANT (bit-identical). The frozen trunk did NOT consume the "
            "injected per-frame coordinates. The latent store holds N copies of one "
            "sequence-only prediction; there is no conformational information in the "
            "dataset, and no model trained on it can represent an MD ensemble.")
    else:
        verdict_lines.append(
            f"Latents DO vary across frames (max abs diff {max_abs_diff:.3e}).")

    # Correlation test: is the variation conformational or numerical?
    if args.atlas_raw is not None and max_abs_diff > 0.0:
        rmsd = coord_rmsd_to_first(
            args.atlas_raw, args.protein_id, frame_idxs, replica_ids)
        if rmsd is None:
            print("\n[verify] (skipped RMSD correlation: MDAnalysis or ATLAS files unavailable)")
        else:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(rmsd[1:], diffs[1:])
            print()
            print(f"[verify] Cα RMSD to frame 0: min={rmsd[1:].min():.3f} Å  "
                  f"max={rmsd[1:].max():.3f} Å")
            print(f"[verify] Spearman(coord RMSD, latent deviation) = {rho:.4f}  (p={pval:.3g})")
            if not np.isfinite(rho) or abs(rho) < 0.3 or pval > 0.05:
                verdict_lines.append(
                    "…but that variation is NOT significantly rank-correlated with actual "
                    "coordinate RMSD, which is the signature of numerical nondeterminism "
                    "(bf16/TF32 reductions, nondeterministic kernels) rather than genuine "
                    "conformational dependence. Treat the latents as effectively "
                    "frame-invariant.")
            else:
                verdict_lines.append(
                    "…and that variation IS rank-correlated with coordinate RMSD, which "
                    "indicates the latents genuinely track conformation. The chai-lab "
                    "source audit in this script's docstring is then incomplete — find "
                    "the path that carries coordinates before trusting either result.")

    print()
    print("─" * 72)
    print("[verdict] " + ("\n          ".join(verdict_lines)))
    print("─" * 72)


if __name__ == "__main__":
    main()
