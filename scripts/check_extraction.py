#!/usr/bin/env python3
"""
scripts/check_extraction.py

Verify swarm extraction completeness.

Run on the cluster:
    python scripts/check_extraction.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --atlas-dir   /scratch/arjunc4/atlas_raw

Reports per-(protein, replica): expected frames, extracted frames, status.
"""

import argparse
import json
import sys
from pathlib import Path

import MDAnalysis as mda


def n_trajectory_frames(atlas_dir: Path, protein_id: str, replica: int) -> int | None:
    xtc = atlas_dir / protein_id / f"{protein_id}_prod_R{replica}_fit.xtc"
    pdb = atlas_dir / protein_id / f"{protein_id}.pdb"
    if not xtc.exists():
        return None
    u = mda.Universe(str(pdb), str(xtc))
    return len(u.trajectory)


def read_manifest(latents_dir: Path, protein_id: str) -> dict:
    mf = latents_dir / protein_id / "manifest.json"
    if not mf.exists():
        return {}
    with open(mf) as f:
        return json.load(f)


def frames_done_per_replica(manifest: dict, protein_id: str) -> dict[str, int]:
    """Returns {replica_str: n_frames} keyed by shard_prefix substring."""
    shards = manifest.get(protein_id, {}).get("shards", [])
    counts: dict[str, int] = {}
    for s in shards:
        path = s["path"]
        # path like: protein_id/protein_id_R1_shard_000000.h5
        name = Path(path).name
        # extract R\d from name
        import re
        m = re.search(r"_R(\d+)_", name)
        if not m:
            continue
        rep = m.group(1)
        counts[rep] = counts.get(rep, 0) + s["n_frames"]
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--atlas-dir", required=True)
    p.add_argument("--skip-traj-count", action="store_true",
                   help="Skip opening XTC files (faster but no expected-frame comparison)")
    args = p.parse_args()

    latents_dir = Path(args.latents_dir)
    atlas_dir   = Path(args.atlas_dir)

    protein_ids = sorted(d.name for d in atlas_dir.iterdir() if d.is_dir())

    COMPLETE = []
    PARTIAL  = []
    MISSING  = []
    total_extracted = 0
    total_expected  = 0

    header = f"{'Protein':<14} {'R':<3} {'Done':>8} {'Expected':>10} {'Status'}"
    print(header)
    print("-" * len(header))

    for protein_id in protein_ids:
        manifest = read_manifest(latents_dir, protein_id)
        done_map = frames_done_per_replica(manifest, protein_id)

        for replica in [1, 2, 3]:
            xtc = atlas_dir / protein_id / f"{protein_id}_prod_R{replica}_fit.xtc"
            if not xtc.exists():
                continue  # this replica doesn't exist in ATLAS

            done = done_map.get(str(replica), 0)
            total_extracted += done

            if args.skip_traj_count:
                expected_str = "?"
                if done == 0:
                    status = "MISSING"
                    MISSING.append((protein_id, replica))
                else:
                    status = f"OK? ({done} frames)"
                    COMPLETE.append((protein_id, replica))
            else:
                expected = n_trajectory_frames(atlas_dir, protein_id, replica)
                total_expected += expected
                expected_str = str(expected)
                if done == 0:
                    status = "MISSING"
                    MISSING.append((protein_id, replica))
                elif done < expected:
                    status = f"PARTIAL ({done/expected*100:.0f}%)"
                    PARTIAL.append((protein_id, replica, done, expected))
                else:
                    status = "COMPLETE"
                    COMPLETE.append((protein_id, replica))

            print(f"{protein_id:<14} {replica:<3} {done:>8} {expected_str:>10}  {status}")

    print()
    print(f"Summary: {len(COMPLETE)} complete, {len(PARTIAL)} partial, {len(MISSING)} missing")
    print(f"Total frames extracted: {total_extracted:,}")
    if not args.skip_traj_count:
        print(f"Total frames expected:  {total_expected:,}")
        pct = total_extracted / total_expected * 100 if total_expected else 0
        print(f"Overall completion:     {pct:.1f}%")

    if PARTIAL:
        print("\nPartial jobs (need resubmission):")
        for item in PARTIAL:
            protein_id, replica, done, expected = item
            print(f"  {protein_id} R{replica}: {done}/{expected} frames")

    if MISSING:
        print("\nMissing jobs (need resubmission):")
        for protein_id, replica in MISSING:
            print(f"  {protein_id} R{replica}")

    if PARTIAL or MISSING:
        print("\nResubmit with: sbatch slurm/extraction/swarm_extract.slurm")
        sys.exit(1)
    else:
        print("\nAll jobs complete.")


if __name__ == "__main__":
    main()
