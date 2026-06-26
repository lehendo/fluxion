"""
data_pipeline/extract_pairs.py

Extracts one static pair tensor per protein (frame 0 of replica 1) and saves
it as <out_dir>/<protein_id>/pair.pt.

Because pair representations encode sequence-level co-evolution and topology
rather than per-frame conformation, one tensor per protein is sufficient for
conditioning the DiT during training.

Usage (single protein, interactive debug):
    python data_pipeline/extract_pairs.py \
        --atlas-raw /scratch/arjunc4/atlas_raw \
        --out-dir   /scratch/arjunc4/fluxion_data/latents \
        --protein-id 16pk_A

Usage (all proteins, run inside a SLURM array via extract_pairs.slurm):
    SLURM_ARRAY_TASK_ID=0 python data_pipeline/extract_pairs.py \
        --atlas-raw /scratch/arjunc4/atlas_raw \
        --out-dir   /scratch/arjunc4/fluxion_data/latents
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import MDAnalysis as mda

# Reuse the Chai-1 pipeline helpers from extract_latents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline.extract_latents import (
    _get_sequence,
    _write_fasta,
    _build_feature_context,
    _patch_atom_coords,
    run_trunk_pipeline,
)


def extract_pair_for_protein(
    protein_id: str,
    atlas_raw: Path,
    out_dir: Path,
    device: torch.device,
) -> None:
    protein_dir = atlas_raw / protein_id
    out_protein_dir = out_dir / protein_id
    pair_path = out_protein_dir / "pair.pt"

    if pair_path.exists():
        print(f"[{protein_id}] pair.pt already exists, skipping.", flush=True)
        return

    # Use replica 1 as the canonical structure source
    pdb_path = protein_dir / f"{protein_id}.pdb"
    xtc_path = protein_dir / f"{protein_id}_prod_R1_fit.xtc"

    if not pdb_path.exists() or not xtc_path.exists():
        print(f"[{protein_id}] PDB or XTC not found, skipping.", flush=True)
        return

    print(f"[{protein_id}] Loading frame 0...", flush=True)
    u = mda.Universe(str(pdb_path), str(xtc_path))
    u.trajectory[0]  # seek to frame 0
    positions = u.atoms.positions.copy()  # [N_atoms, 3]

    # Build Chai-1 feature context from sequence
    esm_cache = out_protein_dir / "esm_cache"
    esm_cache.mkdir(parents=True, exist_ok=True)

    sequence = _get_sequence(u)
    fasta_path = out_protein_dir / f"{protein_id}.fasta"
    _write_fasta(sequence, protein_id, fasta_path)

    print(f"[{protein_id}] Building feature context (ESM on {device})...", flush=True)
    # Load ESM on the GPU to avoid OOM-killing CPU RAM on memory-limited nodes.
    from chai_lab.chai1 import make_all_atom_feature_context
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=esm_cache,
        use_esm_embeddings=True,
        esm_device=device,
    )
    feature_context = _patch_atom_coords(feature_context, positions)

    print(f"[{protein_id}] Running Chai-1 trunk...", flush=True)
    with torch.no_grad():
        _single, pair = run_trunk_pipeline(feature_context, device=device)

    # pair: [L, L, d_pair] — save as float16 to halve disk use
    pair_cpu = pair.squeeze(0).cpu().to(torch.float16)
    out_protein_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pair_cpu, pair_path)

    print(f"[{protein_id}] Saved pair.pt  shape={tuple(pair_cpu.shape)}  "
          f"size={pair_path.stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-raw",   required=True, type=Path)
    parser.add_argument("--out-dir",     required=True, type=Path)
    parser.add_argument("--protein-id",  default=None,
                        help="Single protein to process.")
    parser.add_argument("--all",  action="store_true",
                        help="Process all proteins in --atlas-raw sequentially "
                             "(one Python process, ESM loads once).")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.all:
        proteins = sorted(p.name for p in args.atlas_raw.iterdir() if p.is_dir())
        print(f"Processing all {len(proteins)} proteins in one pass.", flush=True)
    elif args.protein_id:
        proteins = [args.protein_id]
    else:
        # SLURM array mode: sorted list, indexed by SLURM_ARRAY_TASK_ID
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        proteins = sorted(p.name for p in args.atlas_raw.iterdir() if p.is_dir())
        if task_id >= len(proteins):
            print(f"Task {task_id} exceeds protein count {len(proteins)}, nothing to do.")
            return
        proteins = [proteins[task_id]]

    for protein_id in proteins:
        extract_pair_for_protein(protein_id, args.atlas_raw, args.out_dir, device)


if __name__ == "__main__":
    main()
