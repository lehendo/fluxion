"""
scripts/verify_chai1_decode.py

Validates the Chai-1 diffusion decoder (inference/visualize.py's
chai1_decode_to_cif) independently of whether FluxionDiT's generative model
works at all: decode a REAL, already-extracted (single, pair) latent for a
known ATLAS frame, and check the resulting structure is close (low RMSD) to
that frame's real, known coordinates from the original MD trajectory.

This is a ground-truth correctness check for the DECODER itself — if this
fails, the decoder has a bug (wrong shapes, wrong dtype, wrong substitution
point), regardless of anything about whether Fluxion's own training worked.
A passing result at a LOOSE threshold (a few Å) confirms the pipeline isn't
completely broken; a genuinely correct implementation should track much
closer than that, since the exact ground-truth trunk representation that
produced this real structure is what's being fed back in.

Usage
-----
python scripts/verify_chai1_decode.py \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --atlas-raw   /scratch/arjunc4/atlas_raw \\
    --protein-id  1bxy_A \\
    --out-dir     results/verify_chai1_decode \\
    --device      cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.visualize import (
    load_all_latents,
    chai1_decode_to_cif,
    extract_ca_from_cif,
    extract_ca_trajectory,
)
from evaluate_ensemble import kabsch_align, _kabsch_rmsd  # noqa: E402 — sibling module


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[verify_chai1_decode] Loading real latent store for {args.protein_id!r} …",
          flush=True)
    latents, frame_idxs, replica_ids = load_all_latents(args.latents_dir, args.protein_id)

    idx = args.test_index
    single = latents[idx]                        # [L, d_single], raw scale
    true_frame_idx = frame_idxs[idx]
    true_replica = replica_ids[idx]
    print(f"  Test frame: index={idx} -> MD frame_idx={true_frame_idx}, "
          f"replica={true_replica}", flush=True)

    pair_path = args.latents_dir / args.protein_id / "pair.pt"
    pair = torch.load(pair_path, weights_only=True).float()

    fasta_path = args.latents_dir / args.protein_id / f"{args.protein_id}.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    print("[verify_chai1_decode] Loading real ground-truth Cα coordinates …", flush=True)
    real_coords = extract_ca_trajectory(
        args.atlas_raw, args.protein_id,
        np.array([true_frame_idx]), np.array([true_replica]),
    )[0]  # [n_ca, 3]

    print(f"[verify_chai1_decode] Decoding via Chai-1 diffusion module "
          f"({args.num_diffn_timesteps} steps) …", flush=True)
    candidates = chai1_decode_to_cif(
        single, pair, fasta_path, args.out_dir / "cif",
        device, num_diffn_timesteps=args.num_diffn_timesteps,
        num_diffn_samples=1, seed=args.seed,
    )
    decoded_coords = extract_ca_from_cif(candidates.cif_paths[0])  # [n_ca, 3]

    print(f"  real Cα count: {real_coords.shape[0]}, decoded Cα count: "
          f"{decoded_coords.shape[0]}", flush=True)
    if real_coords.shape[0] != decoded_coords.shape[0]:
        raise ValueError(
            f"Cα atom count mismatch: real={real_coords.shape[0]} vs "
            f"decoded={decoded_coords.shape[0]} — likely an atom "
            f"ordering/selection bug, not just imprecise coordinates."
        )

    rmsd = _kabsch_rmsd(decoded_coords, real_coords)

    result = {
        "protein_id": args.protein_id,
        "test_index": idx,
        "md_frame_idx": int(true_frame_idx),
        "md_replica": int(true_replica),
        "n_ca_real": int(real_coords.shape[0]),
        "n_ca_decoded": int(decoded_coords.shape[0]),
        "rmsd_A": float(rmsd),
        "pass_threshold_A": args.pass_threshold,
        "passed": bool(rmsd < args.pass_threshold),
    }

    with open(args.out_dir / "verify_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the Chai-1 diffusion decoder against real ground-truth coordinates"
    )
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--atlas-raw",   required=True, type=Path)
    parser.add_argument("--protein-id",  required=True)
    parser.add_argument("--out-dir",     required=True, type=Path)
    parser.add_argument("--test-index",  default=0, type=int,
                        help="Index into the flat real-latent store to test (default 0)")
    parser.add_argument("--num-diffn-timesteps", default=200, type=int)
    parser.add_argument("--pass-threshold", default=5.0, type=float,
                        help="RMSD (Å) below which the decoder is considered "
                             "correct (default 5.0 — a coarse not-completely-"
                             "broken gate, not a claim of tight agreement)")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run(args)

    print("\n── Verification complete ───────────────────────────────", flush=True)
    for k, v in result.items():
        print(f"  {k}: {v}", flush=True)
    if result["passed"]:
        print(f"\nPASSED — decoder reproduces real structure within "
              f"{args.pass_threshold} Å", flush=True)
    else:
        print(f"\nFAILED — decoder output diverges from real structure "
              f"(RMSD {result['rmsd_A']:.2f} Å >= threshold {args.pass_threshold} Å). "
              "Do not trust this decoder on generated ensembles yet.", flush=True)


if __name__ == "__main__":
    main()
