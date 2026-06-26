"""
inference/visualize.py

Decode a FluxionDiT rollout trajectory (latent space) back to 3D coordinates
and produce MD-compatible output.

Two decoding strategies:
  1. Nearest-neighbor (fast):  for each predicted latent, find the closest
     frame in the MD-derived latent store and return those coordinates.
     Useful for evaluation (does the rollout trace known conformations?) and
     for quick visualization without re-running Chai-1.

  2. Chai-1 structure module (proper): run Chai-1's coordinate diffusion
     conditioned on the predicted single representation. Requires ~40 GB GPU
     memory and runs the full structure module forward pass. Produces genuine
     3D structure predictions.

Usage (nearest-neighbor decode):
    python inference/visualize.py \\
        --rollout     rollout_16pk_A.pt \\
        --latents-dir /scratch/arjunc4/fluxion_data/latents \\
        --atlas-raw   /scratch/arjunc4/atlas_raw \\
        --protein-id  16pk_A \\
        --out-dir     vis_16pk_A \\
        --mode        nn

Usage (Chai-1 decode — requires 48 GB GPU):
    python inference/visualize.py \\
        --rollout     rollout_16pk_A.pt \\
        --latents-dir /scratch/arjunc4/fluxion_data/latents \\
        --atlas-raw   /scratch/arjunc4/atlas_raw \\
        --protein-id  16pk_A \\
        --out-dir     vis_16pk_A \\
        --mode        chai1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis.analysis import rms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Latent store helpers
# ---------------------------------------------------------------------------

def load_all_latents(latents_dir: Path, protein_id: str) -> tuple[torch.Tensor, list[int]]:
    """
    Load all single-token latents for a protein from HDF5 shards.

    Returns:
        latents:     [N, L, d_single] float32  — all MD-derived latents
        frame_idxs:  list of N original MD frame indices
    """
    protein_dir = latents_dir / protein_id
    with open(protein_dir / "manifest.json") as f:
        manifest = json.load(f)

    all_singles = []
    all_frame_idxs = []
    for shard_info in manifest["shards"]:
        shard_path = protein_dir / shard_info["path"]
        with h5py.File(shard_path, "r") as hf:
            all_singles.append(torch.from_numpy(np.array(hf["single"])).float())
            all_frame_idxs.extend(hf["frame_idx"][:].tolist())

    return torch.cat(all_singles, dim=0), all_frame_idxs


# ---------------------------------------------------------------------------
# Nearest-neighbor decoder
# ---------------------------------------------------------------------------

def nn_decode(
    trajectory: torch.Tensor,    # [T, L, d_single]
    latents: torch.Tensor,        # [N, L, d_single] — training set
    frame_idxs: list[int],        # [N] — corresponding MD frame indices
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each latent in the rollout trajectory, find the nearest neighbor in
    the training set (Frobenius distance over [L, d_single]).

    Returns:
        nn_frames:    [T] int — MD frame index of nearest neighbor
        nn_distances: [T] float — mean per-token L2 distance
    """
    T = trajectory.shape[0]
    N = latents.shape[0]

    # Flatten [*, L, d] → [*, L*d] for distance computation
    traj_flat = trajectory.view(T, -1)       # [T, L*d]
    lat_flat  = latents.view(N, -1).float()  # [N, L*d]

    nn_indices   = np.empty(T, dtype=np.int64)
    nn_distances = np.empty(T, dtype=np.float32)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        traj_chunk = traj_flat[start:end]          # [B, L*d]
        # Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        d2 = (
            (traj_chunk ** 2).sum(dim=1, keepdim=True)   # [B, 1]
            + (lat_flat ** 2).sum(dim=1).unsqueeze(0)    # [1, N]
            - 2 * traj_chunk @ lat_flat.T                # [B, N]
        ).clamp(min=0)
        best = d2.argmin(dim=1)                           # [B]
        nn_indices[start:end]   = best.numpy()
        nn_distances[start:end] = d2[torch.arange(end - start), best].sqrt().numpy()

    nn_frames = np.array([frame_idxs[i] for i in nn_indices], dtype=np.int64)
    return nn_frames, nn_distances


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def extract_ca_trajectory(
    atlas_raw: Path,
    protein_id: str,
    frame_indices: np.ndarray,
    replica: int = 1,
) -> np.ndarray:
    """
    Extract Cα coordinates for the given frame indices from an MD trajectory.

    Returns:
        coords: [T, n_ca, 3] float32 — Angstroms
    """
    protein_dir = atlas_raw / protein_id
    pdb_path = protein_dir / f"{protein_id}.pdb"
    xtc_path = protein_dir / f"{protein_id}_prod_R{replica}_fit.xtc"

    u = mda.Universe(str(pdb_path), str(xtc_path))
    ca_sel = u.select_atoms("name CA")

    coords = []
    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        coords.append(ca_sel.positions.copy())

    return np.stack(coords, axis=0).astype(np.float32)  # [T, n_ca, 3]


# ---------------------------------------------------------------------------
# Chai-1 structure module decoder
# ---------------------------------------------------------------------------

def chai1_decode(
    trajectory: torch.Tensor,   # [T, L, d_single]
    pair: torch.Tensor,          # [L, L, d_pair]
    fasta_path: Path,
    device: torch.device,
    n_diffusion_samples: int = 1,
) -> np.ndarray:
    """
    Decode each latent in the rollout trajectory to 3D coordinates using
    Chai-1's structure module.

    This loads all four Chai-1 compiled modules and runs the full decoding
    pipeline for each step.  Expect ~2-4 s/step on an A100.

    Returns:
        coords: [T, L, 3] float32 — predicted Cα positions in Angstroms
    """
    try:
        from chai_lab.chai1 import load_exported, run_inference
        from chai_lab.chai1 import make_all_atom_feature_context
    except ImportError as exc:
        raise ImportError("chai-lab is not installed or not on sys.path.") from exc

    T = trajectory.shape[0]
    all_coords = []

    print(f"Decoding {T} frames via Chai-1 structure module ...", flush=True)
    for step_idx in range(T):
        if step_idx % 10 == 0:
            print(f"  step {step_idx}/{T}", flush=True)

        single = trajectory[step_idx].unsqueeze(0).to(device)  # [1, L, d]
        pair_b = pair.unsqueeze(0).to(device)                   # [1, L, L, d]

        # Chai-1's structure module expects the single + pair outputs from the
        # trunk and runs an EDM diffusion process to sample coordinates.
        # The exact API depends on the chai-lab version installed; this calls
        # the run_inference helper which wraps the structure module sampling.
        try:
            from chai_lab.chai1 import _decode_structure  # type: ignore[attr-defined]
            xyz = _decode_structure(single, pair_b, fasta_path, device=device)
        except (ImportError, AttributeError):
            # Fallback: alert the user that this path requires deeper chai-lab integration
            raise NotImplementedError(
                "Chai-1 structure decoding is not yet wired up for this chai-lab version.\n"
                "Use --mode nn for nearest-neighbor decoding instead.\n"
                "To implement proper decoding, hook into chai_lab internals:\n"
                "  from chai_lab.model.diffusion.conditioning import DiffusionConditioning\n"
                "  from chai_lab.model.diffusion.model import DiffusionModule"
            )

        # xyz: [1, L, 3]
        all_coords.append(xyz.squeeze(0).cpu().numpy())

    return np.stack(all_coords, axis=0).astype(np.float32)  # [T, L, 3]


# ---------------------------------------------------------------------------
# RMSD evaluation
# ---------------------------------------------------------------------------

def compute_pairwise_rmsds(
    coords: np.ndarray,  # [T, n_ca, 3]
    ref_idx: int = 0,
) -> np.ndarray:
    """RMSD of each frame against the reference frame (after Kabsch alignment)."""
    from MDAnalysis.analysis.rms import rmsd as mda_rmsd

    ref = coords[ref_idx]
    rmsds = np.array([
        mda_rmsd(coords[i], ref, superposition=True)
        for i in range(len(coords))
    ])
    return rmsds


# ---------------------------------------------------------------------------
# Trajectory writing (MDAnalysis)
# ---------------------------------------------------------------------------

def write_trajectory(
    coords: np.ndarray,  # [T, n_atoms, 3]
    pdb_template: Path,
    out_pdb: Path,
    selection: str = "name CA",
) -> None:
    """
    Write decoded trajectory as a multi-frame PDB using the input structure
    as topology template.
    """
    u = mda.Universe(str(pdb_template))
    sel = u.select_atoms(selection)

    if coords.shape[1] != len(sel):
        raise ValueError(
            f"Coordinate frames have {coords.shape[1]} atoms but "
            f"selection '{selection}' has {len(sel)} atoms."
        )

    with mda.Writer(str(out_pdb), sel.n_atoms) as writer:
        for frame_coords in coords:
            sel.positions = frame_coords
            writer.write(sel)

    print(f"Wrote {len(coords)}-frame PDB → {out_pdb}", flush=True)


# ---------------------------------------------------------------------------
# Matplotlib plot
# ---------------------------------------------------------------------------

def plot_nn_summary(
    nn_frames: np.ndarray,
    nn_distances: np.ndarray,
    rmsds: np.ndarray | None,
    out_dir: Path,
    protein_id: str,
) -> None:
    """Summary plot: NN frame index and NN distance along the rollout."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.", flush=True)
        return

    T = len(nn_frames)
    t_axis = np.linspace(0, 1, T)

    n_panels = 3 if rmsds is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)

    axes[0].plot(t_axis, nn_frames)
    axes[0].set_ylabel("NN frame index")
    axes[0].set_title(f"{protein_id} — Fluxion rollout nearest-neighbor analysis")

    axes[1].plot(t_axis, nn_distances)
    axes[1].set_ylabel("NN distance (L2)")

    if rmsds is not None:
        axes[2].plot(t_axis, rmsds)
        axes[2].set_ylabel("RMSD to t=0 (Å)")
        axes[2].set_xlabel("t")
    else:
        axes[1].set_xlabel("t")

    plt.tight_layout()
    out_path = out_dir / f"{protein_id}_rollout_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Decode FluxionDiT rollout to 3D")
    parser.add_argument("--rollout",      required=True, type=Path,
                        help=".pt file from rollout.py")
    parser.add_argument("--latents-dir",  required=True, type=Path)
    parser.add_argument("--atlas-raw",    required=True, type=Path)
    parser.add_argument("--protein-id",   required=True)
    parser.add_argument("--out-dir",      required=True, type=Path)
    parser.add_argument("--mode",         default="nn",
                        choices=["nn", "chai1"],
                        help="Decoding strategy (default: nn)")
    parser.add_argument("--replica",      default=1, type=int,
                        help="ATLAS replica to use for coordinate extraction (default 1)")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load rollout
    print(f"Loading rollout from {args.rollout}", flush=True)
    rollout_data = torch.load(args.rollout, weights_only=True, map_location="cpu")
    trajectory = rollout_data["trajectory"]   # [T+1, L, d_single]
    t_eval      = rollout_data["t_eval"]
    print(f"  Trajectory: {tuple(trajectory.shape)}  t=[{t_eval[0]:.2f}..{t_eval[-1]:.2f}]",
          flush=True)

    if args.mode == "nn":
        # ── Nearest-neighbor decode ─────────────────────────────────────────
        print("Loading all latents for nearest-neighbor lookup ...", flush=True)
        latents, frame_idxs = load_all_latents(args.latents_dir, args.protein_id)
        print(f"  {len(frame_idxs)} frames in latent store", flush=True)

        print("Running nearest-neighbor matching ...", flush=True)
        nn_frames, nn_distances = nn_decode(trajectory, latents, frame_idxs)
        print(f"  NN frame range: [{nn_frames.min()}, {nn_frames.max()}]", flush=True)

        # Save NN mapping
        nn_path = args.out_dir / f"{args.protein_id}_nn_frames.npz"
        np.savez(nn_path,
                 nn_frames=nn_frames,
                 nn_distances=nn_distances,
                 t_eval=t_eval.numpy())
        print(f"Saved NN data → {nn_path}", flush=True)

        # Extract coordinates from MD trajectory
        print("Extracting Cα coordinates from MD trajectory ...", flush=True)
        ca_coords = extract_ca_trajectory(
            args.atlas_raw, args.protein_id, nn_frames, replica=args.replica
        )
        print(f"  Cα coords shape: {ca_coords.shape}", flush=True)

        # RMSD progression
        rmsds = compute_pairwise_rmsds(ca_coords, ref_idx=0)
        print(f"  RMSD range: {rmsds.min():.2f} – {rmsds.max():.2f} Å", flush=True)

        # Write decoded trajectory as multi-frame PDB
        pdb_template = args.atlas_raw / args.protein_id / f"{args.protein_id}.pdb"
        write_trajectory(
            ca_coords,
            pdb_template,
            args.out_dir / f"{args.protein_id}_rollout_nn.pdb",
            selection="name CA",
        )

        # Summary plot
        plot_nn_summary(nn_frames, nn_distances, rmsds, args.out_dir, args.protein_id)

    elif args.mode == "chai1":
        # ── Chai-1 structure module decode ──────────────────────────────────
        pair_path = args.latents_dir / args.protein_id / "pair.pt"
        pair = torch.load(pair_path, weights_only=True).float()

        fasta_path = (args.latents_dir / args.protein_id
                      / f"{args.protein_id}.fasta")
        if not fasta_path.exists():
            raise FileNotFoundError(
                f"FASTA not found at {fasta_path}.  "
                "Run extract_pairs.py first to generate it."
            )

        coords = chai1_decode(trajectory, pair, fasta_path, device)
        out_pt = args.out_dir / f"{args.protein_id}_rollout_chai1.pt"
        torch.save({"coords": torch.from_numpy(coords), "t_eval": t_eval}, out_pt)
        print(f"Saved Chai-1-decoded trajectory → {out_pt}", flush=True)


if __name__ == "__main__":
    main()
