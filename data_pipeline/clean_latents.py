"""
data_pipeline/clean_latents.py

Remove corrupted HDF5 shards and manifests from a latents directory,
leaving pair.pt, FASTA files, and ESM caches intact.

Use this to reset after a bad swarm run before re-extracting.

Usage:
    # Dry-run — see what would be deleted:
    python data_pipeline/clean_latents.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --dry-run

    # Actually delete:
    python data_pipeline/clean_latents.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_KEEP_SUFFIXES = {".pt", ".fasta"}   # pair.pt, *.fasta
_KEEP_NAMES    = {"esm_cache"}       # ESM embedding cache dir


def clean_protein_dir(protein_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Delete shards and manifests inside a single protein directory.

    Returns (n_deleted, bytes_freed).
    """
    n_deleted   = 0
    bytes_freed = 0

    for item in sorted(protein_dir.iterdir()):
        # Keep pair.pt, *.fasta, esm_cache/
        if item.name in _KEEP_NAMES:
            continue
        if item.suffix in _KEEP_SUFFIXES:
            continue

        # Delete HDF5 shards, manifest.json, manifest.json.lock, and any
        # nested protein subdirectory (the double-nesting artifact).
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {'[dry]' if dry_run else 'DEL '} dir  {item}  ({size/1e6:.1f} MB)")
            if not dry_run:
                import shutil
                shutil.rmtree(item)
        else:
            size = item.stat().st_size
            print(f"  {'[dry]' if dry_run else 'DEL '} file {item}  ({size/1e6:.2f} MB)")
            if not dry_run:
                item.unlink()

        n_deleted   += 1
        bytes_freed += size

    return n_deleted, bytes_freed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting.")
    parser.add_argument("--protein-id", default=None,
                        help="Clean only this protein (default: all).")
    args = parser.parse_args()

    latents_dir = args.latents_dir
    if not latents_dir.exists():
        sys.exit(f"ERROR: {latents_dir} does not exist.")

    # Also clean stray top-level files (manifest.json, esm_cache at root)
    top_level_stray = [
        latents_dir / "manifest.json",
        latents_dir / "manifest.json.lock",
        latents_dir / "esm_cache",
        latents_dir / "1AKE",   # uppercase stray from early test run
    ]
    for stray in top_level_stray:
        if stray.exists():
            if stray.is_dir():
                size = sum(f.stat().st_size for f in stray.rglob("*") if f.is_file())
                print(f"{'[dry]' if args.dry_run else 'DEL '} stray dir  {stray}  ({size/1e6:.1f} MB)")
                if not args.dry_run:
                    import shutil
                    shutil.rmtree(stray)
            else:
                size = stray.stat().st_size
                print(f"{'[dry]' if args.dry_run else 'DEL '} stray file {stray}  ({size/1e6:.2f} MB)")
                if not args.dry_run:
                    stray.unlink()

    # Per-protein cleanup
    if args.protein_id:
        protein_dirs = [latents_dir / args.protein_id]
    else:
        protein_dirs = sorted(
            p for p in latents_dir.iterdir()
            if p.is_dir() and p.name not in {"esm_cache", "1AKE"}
        )

    total_deleted = 0
    total_freed   = 0

    for protein_dir in protein_dirs:
        print(f"\n[{protein_dir.name}]")
        n, b = clean_protein_dir(protein_dir, args.dry_run)
        total_deleted += n
        total_freed   += b

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {total_deleted} items, {total_freed/1e9:.2f} GB freed.")
    if args.dry_run:
        print("Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
