"""
inference/infer.py

End-to-end inference: trained checkpoint → latent rollout → 3D PDB trajectory.

Combines rollout.py and visualize.py into a single command so you never have
to manage intermediate files manually.  All outputs land in --out-dir:

    <out-dir>/
        rollout.pt                    raw latent trajectory [T+1, L, d_single]
        <protein>_nn_frames.npz       NN frame/replica/distance mapping
        <protein>_rollout_nn.pdb      decoded Cα trajectory (multi-frame PDB)
        <protein>_rollout_nn.xtc      decoded Cα trajectory (XTC, for VMD/PyMOL)
        <protein>_rollout_summary.png NN frame, replica, distance, RMSD plot
        metrics.json                  RMSD stats, NN distance stats, timing

Usage
-----
python inference/infer.py \\
    --checkpoint  /scratch/arjunc4/fluxion_checkpoints/checkpoint.pt \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --atlas-raw   /scratch/arjunc4/atlas_raw \\
    --protein-id  16pk_A \\
    --out-dir     results/16pk_A \\
    --n-steps     200 \\
    --method      euler \\
    --device      cuda

Skip the ODE rollout if rollout.pt already exists (useful for re-decoding):
    --skip-rollout

Use Chai-1 diffusion module instead of NN decoder:
    --mode chai1
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.rollout import load_model, load_z0_from_latents, rollout
from inference.visualize import (
    load_all_latents,
    nn_decode,
    extract_ca_trajectory,
    chai1_decode_to_cif,
    extract_ca_from_cif,
    compute_pairwise_rmsds,
    write_pdb_trajectory,
    write_xtc_trajectory,
    plot_nn_summary,
)


def run(args: argparse.Namespace) -> dict:
    """Execute the full inference pipeline. Returns a metrics dict."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics: dict = {"protein_id": args.protein_id, "device": str(device)}

    # ── Stage 1: ODE rollout ────────────────────────────────────────────────
    rollout_path = args.out_dir / "rollout.pt"

    if args.skip_rollout and rollout_path.exists():
        print(f"[infer] Reusing existing rollout from {rollout_path}", flush=True)
        rollout_data = torch.load(rollout_path, weights_only=False, map_location="cpu")
        trajectory   = rollout_data["trajectory"]
        t_eval       = rollout_data["t_eval"]
    else:
        print(f"[infer] Loading model from {args.checkpoint}", flush=True)
        model = load_model(args.checkpoint, device, use_ema=not args.no_ema)

        norm_stats_path = args.latents_dir / "norm_stats.pt"
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_path}. "
                "Run data_pipeline/compute_norm_stats.py first — the model "
                "was trained on normalized latents, so rollout must "
                "integrate in that same space."
            )
        norm_stats = torch.load(norm_stats_path, weights_only=True)

        print(f"[infer] Loading z0: protein={args.protein_id!r} frame={args.frame}",
              flush=True)
        z0, pair = load_z0_from_latents(args.latents_dir, args.protein_id, args.frame)
        print(f"  z0 {tuple(z0.shape)}  pair {tuple(pair.shape)}", flush=True)

        print(f"[infer] Rolling out: method={args.method}  n_steps={args.n_steps}",
              flush=True)
        t0 = time.perf_counter()
        trajectory, t_eval = rollout(
            model, z0, pair,
            n_steps=args.n_steps,
            t_end=args.t_end,
            method=args.method,
            device=device,
            norm_stats=norm_stats,
        )
        rollout_elapsed = time.perf_counter() - t0
        print(f"  trajectory {tuple(trajectory.shape)}  "
              f"({rollout_elapsed:.1f}s, "
              f"{rollout_elapsed / args.n_steps * 1000:.1f} ms/step)",
              flush=True)
        metrics["rollout_seconds"] = round(rollout_elapsed, 2)
        metrics["ms_per_step"]     = round(rollout_elapsed / args.n_steps * 1000, 1)

        torch.save(
            {
                "trajectory": trajectory,
                "t_eval":     t_eval,
                "protein_id": args.protein_id,
                "frame_idx":  args.frame,
                "method":     args.method,
                "n_steps":    args.n_steps,
                "checkpoint": str(args.checkpoint),
            },
            rollout_path,
        )
        print(f"[infer] Saved rollout → {rollout_path}", flush=True)

    T = trajectory.shape[0]
    metrics["n_steps"] = T - 1

    # ── Stage 2: Decode to 3D ───────────────────────────────────────────────
    pdb_template = args.atlas_raw / args.protein_id / f"{args.protein_id}.pdb"

    if args.mode == "nn":
        print("[infer] Loading latent store for NN decode …", flush=True)
        latents, frame_idxs, replica_ids = load_all_latents(
            args.latents_dir, args.protein_id
        )
        print(f"  {len(frame_idxs)} frames, replicas {sorted(set(replica_ids))}",
              flush=True)

        print("[infer] Nearest-neighbor search …", flush=True)
        t0 = time.perf_counter()
        nn_frames, nn_replicas, nn_distances = nn_decode(
            trajectory, latents, frame_idxs, replica_ids,
            batch_size=args.nn_batch_size,
        )
        nn_elapsed = time.perf_counter() - t0
        print(f"  frame range [{nn_frames.min()}, {nn_frames.max()}]  "
              f"mean dist={nn_distances.mean():.4f}  ({nn_elapsed:.1f}s)",
              flush=True)
        metrics["nn_mean_distance"] = float(nn_distances.mean())
        metrics["nn_max_distance"]  = float(nn_distances.max())

        npz_path = args.out_dir / f"{args.protein_id}_nn_frames.npz"
        np.savez(
            npz_path,
            nn_frames=nn_frames,
            nn_replicas=nn_replicas,
            nn_distances=nn_distances,
            t_eval=t_eval.numpy(),
        )
        print(f"[infer] Saved NN data → {npz_path}", flush=True)

        print("[infer] Extracting Cα coordinates …", flush=True)
        ca_coords = extract_ca_trajectory(
            args.atlas_raw, args.protein_id, nn_frames, nn_replicas
        )
        print(f"  Cα shape: {ca_coords.shape}", flush=True)

        rmsds = compute_pairwise_rmsds(ca_coords, ref_idx=0)
        print(f"  Cα RMSD: min={rmsds.min():.2f}  "
              f"max={rmsds.max():.2f}  mean={rmsds.mean():.2f} Å", flush=True)
        metrics["rmsd_min_A"]  = round(float(rmsds.min()), 3)
        metrics["rmsd_max_A"]  = round(float(rmsds.max()), 3)
        metrics["rmsd_mean_A"] = round(float(rmsds.mean()), 3)

        write_pdb_trajectory(
            ca_coords, pdb_template,
            args.out_dir / f"{args.protein_id}_rollout_nn.pdb",
        )
        write_xtc_trajectory(
            ca_coords, pdb_template,
            args.out_dir / f"{args.protein_id}_rollout_nn.xtc",
        )
        plot_nn_summary(
            nn_frames, nn_replicas, nn_distances, rmsds,
            args.out_dir, args.protein_id,
        )

    elif args.mode == "chai1":
        pair_path  = args.latents_dir / args.protein_id / "pair.pt"
        pair       = torch.load(pair_path, weights_only=True).float()
        fasta_path = args.latents_dir / args.protein_id / f"{args.protein_id}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(
                f"FASTA not found at {fasta_path}. "
                "Run extract_pairs.py to generate it."
            )

        # Real diffusion sampling (num_diffn_timesteps steps) per frame — a
        # genuine multi-step generative decode, not a cheap single forward
        # pass. Expensive by construction; this is a legacy per-frame loop
        # kept working for the temporal trajectory path. For the current
        # static-ensemble pivot, decode ensemble members individually via
        # chai1_decode_to_cif directly instead of looping a "trajectory".
        print(f"[infer] Chai-1 diffusion decode ({T} steps, real multi-step "
              f"sampling per step — this is slow) …", flush=True)
        t0 = time.perf_counter()
        chai1_out_dir = args.out_dir / f"{args.protein_id}_chai1_cif"
        all_coords = []
        for step_idx in range(T):
            print(f"  Chai-1 decode: {step_idx}/{T}", flush=True)
            candidates = chai1_decode_to_cif(
                trajectory[step_idx], pair, fasta_path,
                chai1_out_dir / f"step_{step_idx:04d}",
                device, num_diffn_samples=1,
            )
            all_coords.append(extract_ca_from_cif(candidates.cif_paths[0]))
        coords = np.stack(all_coords, axis=0).astype(np.float32)
        chai_elapsed = time.perf_counter() - t0
        print(f"  Decoded {T} frames in {chai_elapsed:.1f}s "
              f"({chai_elapsed / T:.1f} s/step)", flush=True)
        metrics["chai1_decode_seconds"] = round(chai_elapsed, 2)

        out_pt = args.out_dir / f"{args.protein_id}_rollout_chai1.pt"
        torch.save({"coords": torch.from_numpy(coords), "t_eval": t_eval}, out_pt)
        print(f"[infer] Saved Chai-1 coords → {out_pt}", flush=True)

    # ── Stage 3: Save metrics ───────────────────────────────────────────────
    metrics_path = args.out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[infer] Saved metrics → {metrics_path}", flush=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fluxion end-to-end inference: checkpoint → rollout → PDB"
    )
    # Required
    parser.add_argument("--checkpoint",    required=True, type=Path)
    parser.add_argument("--latents-dir",   required=True, type=Path)
    parser.add_argument("--atlas-raw",     required=True, type=Path)
    parser.add_argument("--protein-id",    required=True)
    parser.add_argument("--out-dir",       required=True, type=Path)

    # Rollout options
    parser.add_argument("--frame",     default=0,       type=int,
                        help="Initial MD frame index to use as z_0 (default 0)")
    parser.add_argument("--n-steps",   default=200,     type=int,
                        help="ODE integration steps (default 200)")
    parser.add_argument("--t-end",     default=1.0,     type=float)
    parser.add_argument("--method",    default="euler",
                        choices=["euler", "heun", "rk4", "dopri5", "bosh3"],
                        help="ODE integrator (default: euler)")
    parser.add_argument("--no-ema",    action="store_true",
                        help="Use raw model weights instead of EMA")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="Reuse existing rollout.pt in --out-dir")

    # Decode options
    parser.add_argument("--mode",         default="nn", choices=["nn", "chai1"],
                        help="Decoding mode: nn (default) or chai1")
    parser.add_argument("--nn-batch-size", default=64,  type=int,
                        help="Batch size for NN distance computation (default 64)")

    # Hardware
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    metrics = run(args)

    print("\n── Inference complete ──────────────────────────────────", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
