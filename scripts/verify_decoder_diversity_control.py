#!/usr/bin/env python3
"""
scripts/verify_decoder_diversity_control.py

THE MISSING CONTROL for every diversity number this project reports.

The confound
------------
scripts/evaluate_ensemble.py decodes generated sample i with
`chai1_decode_to_cif(..., seed=args.seed + i)` — a DIFFERENT diffusion seed per
sample. So the reported "generated ensemble diversity" (pairwise Cα RMSD spread)
is the sum of two sources:

    (a) genuine variation between the generated latents  — what we want to measure
    (b) Chai-1's own stochastic diffusion decoding        — pure nuisance

Nothing in the repo separates them. This matters in both directions:

  * If measured diversity is LOW, one cannot conclude the generative model
    mode-collapsed — the decoder might be contracting distinct latents onto one
    structure (the trunk representation is a fold-level descriptor; the diffusion
    module may simply be insensitive to small latent perturbations).
  * If measured diversity is HIGH, one cannot claim the model learned the
    conformational ensemble — decoder seed noise alone can produce Å-scale
    spread, so the number may contain no generative signal at all.

This script measures (b) alone: decode ONE FIXED latent N times, varying only
the decoder seed. That is the noise floor. Any claim about generated diversity
must be stated relative to it.

How to read the output
----------------------
Compare `decoder_only_diversity` here against `gen_diversity` from
evaluate_ensemble.py on the same protein:

  gen_diversity ≈ decoder_only_diversity
      -> the reported diversity is decoder noise. No evidence the generative
         model produces diverse conformations. (And if gen_diversity is also
         ≈ real_diversity, that agreement is coincidental, not success.)

  gen_diversity >> decoder_only_diversity
      -> genuine generative diversity above the noise floor; the excess is the
         real, defensible signal to report.

  gen_diversity << decoder_only_diversity
      -> generated latents are collapsing to a single point AND the decoder is
         suppressing its own noise: two problems, not one.

Also reported: the same measurement on a REAL extracted latent. If real and
generated latents give the same decoder-only spread, the decoder is not
distinguishing them at all.

Usage
-----
python scripts/verify_decoder_diversity_control.py \
    --latents-dir  /scratch/arjunc4/fluxion_data/latents \
    --ensemble     results/ensemble_1bxy_A.pt \
    --protein-id   1bxy_A \
    --n-decodes    16 \
    --out-dir      results/decoder_control \
    --device       cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.visualize import (  # noqa: E402
    load_all_latents,
    chai1_decode_to_cif,
    extract_ca_from_cif,
    build_chai1_feature_context,
)
from evaluate_ensemble import (  # noqa: E402
    align_ensemble,
    filter_majority_shape,
    pairwise_rmsd_distribution,
    compute_rmsf,
)


def decode_repeatedly(single: torch.Tensor, pair: torch.Tensor, fasta_path: Path,
                      out_dir: Path, device: torch.device, n_decodes: int,
                      n_timesteps: int, base_seed: int, tag: str,
                      feature_context=None) -> np.ndarray:
    """Decode ONE latent n_decodes times, varying only the decoder seed."""
    coords: list[np.ndarray] = []
    for i in range(n_decodes):
        cands = chai1_decode_to_cif(
            single, pair, fasta_path, out_dir / f"{tag}_{i:03d}",
            device, num_diffn_timesteps=n_timesteps,
            num_diffn_samples=1, seed=base_seed + i,
            feature_context=feature_context,
        )
        if not getattr(cands, "cif_paths", None):
            print(f"  [{tag}] decode {i}: no CIF produced, skipping", flush=True)
            continue
        coords.append(extract_ca_from_cif(cands.cif_paths[0]))
        print(f"  [{tag}] decode {i+1}/{n_decodes} done", flush=True)
    stacked, n_excl = filter_majority_shape(coords)
    if n_excl:
        print(f"  [{tag}] excluded {n_excl} minority-atom-count decodes", flush=True)
    return stacked


def summarize(coords: np.ndarray, label: str) -> dict:
    aligned = align_ensemble(coords, coords[0])
    rmsds = pairwise_rmsd_distribution(aligned)
    rmsf = compute_rmsf(aligned)
    stats = {
        f"{label}_n_decodes":      int(coords.shape[0]),
        f"{label}_mean_pair_rmsd": float(rmsds.mean()),
        f"{label}_std_pair_rmsd":  float(rmsds.std()),
        f"{label}_max_pair_rmsd":  float(rmsds.max()),
        f"{label}_mean_rmsf":      float(rmsf.mean()),
        f"{label}_max_rmsf":       float(rmsf.max()),
    }
    print(f"\n[{label}] mean pairwise Cα RMSD = {stats[f'{label}_mean_pair_rmsd']:.3f} Å "
          f"(max {stats[f'{label}_max_pair_rmsd']:.3f} Å), "
          f"mean RMSF = {stats[f'{label}_mean_rmsf']:.3f} Å", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", type=Path, required=True)
    ap.add_argument("--ensemble", type=Path, default=None,
                    help="Optional: a generated ensemble .pt from "
                         "inference/sample_ensemble.py. Its first latent is "
                         "decoded repeatedly too, so the decoder noise floor is "
                         "measured on a GENERATED latent as well as a real one.")
    ap.add_argument("--protein-id", type=str, required=True)
    ap.add_argument("--n-decodes", type=int, default=16)
    ap.add_argument("--num-diffn-timesteps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    protein_dir = args.latents_dir / args.protein_id
    pair = torch.load(protein_dir / "pair.pt", weights_only=True).float()
    fasta_path = protein_dir / f"{args.protein_id}.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(fasta_path)

    metrics: dict = {"protein_id": args.protein_id, "n_decodes": args.n_decodes}

    # Build once and reuse: same sequence -> same context (skips re-running ESM
    # on every decode), exactly as evaluate_ensemble.py does.
    feature_context = build_chai1_feature_context(fasta_path)

    # ---- Noise floor on a REAL latent -------------------------------------
    latents, _, _ = load_all_latents(args.latents_dir, args.protein_id)
    print(f"[control] Decoding ONE REAL latent {args.n_decodes}× "
          f"(varying decoder seed only) …", flush=True)
    real_repeat = decode_repeatedly(
        latents[0], pair, fasta_path, args.out_dir, device,
        args.n_decodes, args.num_diffn_timesteps, args.seed, "real_fixed",
        feature_context=feature_context)
    metrics.update(summarize(real_repeat, "decoder_noise_floor_real_latent"))

    # ---- Noise floor on a GENERATED latent --------------------------------
    if args.ensemble is not None and args.ensemble.exists():
        blob = torch.load(args.ensemble, weights_only=True)
        # sample_ensemble.py saves {"ensemble": [n, L, d_single], ...}
        gen = blob["ensemble"] if isinstance(blob, dict) and "ensemble" in blob else blob
        gen = gen.float()
        print(f"\n[control] Decoding ONE GENERATED latent {args.n_decodes}× "
              f"(varying decoder seed only) …", flush=True)
        gen_repeat = decode_repeatedly(
            gen[0], pair, fasta_path, args.out_dir, device,
            args.n_decodes, args.num_diffn_timesteps, args.seed, "gen_fixed",
            feature_context=feature_context)
        metrics.update(summarize(gen_repeat, "decoder_noise_floor_gen_latent"))

        # ---- Latent spread, measured in latent space (decoder-free) -------
        flat = gen.reshape(gen.shape[0], -1)
        pdist = torch.cdist(flat, flat)
        iu = torch.triu_indices(len(flat), len(flat), offset=1)
        vals = pdist[iu[0], iu[1]]
        metrics["gen_latent_mean_pairwise_l2"] = float(vals.mean())
        metrics["gen_latent_max_pairwise_l2"]  = float(vals.max())
        metrics["gen_latent_min_pairwise_l2"]  = float(vals.min())
        print(f"\n[control] Generated LATENT pairwise L2: "
              f"mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f}")
        print("[control] (If min≈0 the generative model is producing duplicate "
              "latents — mode collapse in latent space, independent of the decoder.)")

    out_json = args.out_dir / "decoder_diversity_control.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("─" * 72)
    print("[control] Decoder-only diversity is the NOISE FLOOR. Any diversity")
    print("[control] reported by evaluate_ensemble.py must exceed it before it can")
    print("[control] be attributed to the generative model. Compare:")
    print("[control]   evaluate_ensemble  gen_diversity_mean_rmsd")
    print("[control]   vs. this script's  decoder_noise_floor_*_mean_pair_rmsd")
    print(f"[control] Saved → {out_json}")
    print("─" * 72)


if __name__ == "__main__":
    main()
