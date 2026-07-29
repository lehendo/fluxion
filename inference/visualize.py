"""
inference/visualize.py

Decode a FluxionDiT rollout trajectory (latent space) back to 3D coordinates
and produce MD-compatible output.

Two decoding strategies
-----------------------
1. Nearest-neighbor (nn):  for each predicted latent, find the closest frame
   in the MD-derived latent store and return those Cα coordinates.  No GPU
   required for decoding; runs in seconds.  Useful for evaluation (does the
   rollout trace known conformations?) and quick visualization.

2. Chai-1 structure module (chai1):  inject the predicted single representation
   into Chai-1's diffusion decoder to produce genuine 3D coordinates.
   Requires ~40 GB GPU memory and ~2–4 s/step.

Usage
-----
# Nearest-neighbor:
python inference/visualize.py \\
    --rollout     rollout_16pk_A.pt \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --atlas-raw   /scratch/arjunc4/atlas_raw \\
    --protein-id  16pk_A \\
    --out-dir     vis_16pk_A

# Chai-1 decode (needs 48 GB GPU):
python inference/visualize.py \\
    --rollout     rollout_16pk_A.pt \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --atlas-raw   /scratch/arjunc4/atlas_raw \\
    --protein-id  16pk_A \\
    --out-dir     vis_16pk_A \\
    --mode        chai1 \\
    --device      cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Latent store helpers
# ---------------------------------------------------------------------------

def load_all_latents(
    latents_dir: Path,
    protein_id: str,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """
    Load all single-token latents from every shard for a protein.

    Deduplicates shard entries (defensive against old concurrency bug) and
    parses the replica number from shard filenames of the form
    ``<protein>_R<N>_shard_<idx>.h5``.

    Returns:
        latents:     float32 [N, L, d_single]
        frame_idxs:  [N] — original MD frame index within the replica's trajectory
        replica_ids: [N] — which ATLAS replica (1/2/3) each frame came from
    """
    protein_dir = latents_dir / protein_id
    with open(protein_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Deduplicate by path (same logic as dataset.py)
    seen: set[str] = set()
    unique_shards = [
        s for s in manifest[protein_id]["shards"]
        if not (s["path"] in seen or seen.add(s["path"]))  # type: ignore[func-returns-value]
    ]

    all_singles: list[torch.Tensor] = []
    all_frame_idxs: list[int] = []
    all_replica_ids: list[int] = []

    for shard_info in unique_shards:
        shard_path = protein_dir / shard_info["path"]
        # Parse replica: "16pk_A_R2_shard_000003.h5" → 2
        m = re.search(r"_R(\d+)_", shard_path.name)
        replica = int(m.group(1)) if m else 1

        with h5py.File(shard_path, "r") as hf:
            all_singles.append(torch.from_numpy(np.array(hf["single"])).float())
            frames = hf["frame_idx"][:].tolist()
            all_frame_idxs.extend(frames)
            all_replica_ids.extend([replica] * len(frames))

    latents = torch.cat(all_singles, dim=0)  # [N, L, d_single]
    return latents, all_frame_idxs, all_replica_ids


# ---------------------------------------------------------------------------
# Nearest-neighbor decoder
# ---------------------------------------------------------------------------

def nn_decode(
    trajectory: torch.Tensor,   # [T, L, d_single]
    latents: torch.Tensor,       # [N, L, d_single]
    frame_idxs: list[int],
    replica_ids: list[int],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each rollout latent, find the nearest neighbor in the training set
    (Frobenius / flattened-L2 distance over [L, d_single]).

    Returns:
        nn_frames:    int64 [T]   — MD frame index of nearest neighbor
        nn_replicas:  int64 [T]   — ATLAS replica of nearest neighbor
        nn_distances: float32 [T] — L2 distance to nearest neighbor
    """
    T = trajectory.shape[0]
    N = latents.shape[0]

    traj_flat = trajectory.view(T, -1).float()   # [T, L*d]
    lat_flat  = latents.view(N, -1).float()      # [N, L*d]

    nn_indices   = np.empty(T, dtype=np.int64)
    nn_distances = np.empty(T, dtype=np.float32)

    for start in range(0, T, batch_size):
        end   = min(start + batch_size, T)
        chunk = traj_flat[start:end]              # [B, L*d]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        d2 = (
            (chunk ** 2).sum(dim=1, keepdim=True)
            + (lat_flat ** 2).sum(dim=1).unsqueeze(0)
            - 2 * chunk @ lat_flat.T
        ).clamp(min=0.0)
        best = d2.argmin(dim=1)
        nn_indices[start:end]   = best.numpy()
        nn_distances[start:end] = d2[torch.arange(end - start), best].sqrt().numpy()

    nn_frames   = np.array([frame_idxs[i]  for i in nn_indices], dtype=np.int64)
    nn_replicas = np.array([replica_ids[i] for i in nn_indices], dtype=np.int64)
    return nn_frames, nn_replicas, nn_distances


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def extract_ca_trajectory(
    atlas_raw: Path,
    protein_id: str,
    frame_indices: np.ndarray,   # [T] — frame within each replica's trajectory
    replica_ids: np.ndarray,     # [T] — which replica to open
) -> np.ndarray:
    """
    Extract Cα coordinates for each (frame_idx, replica) pair.

    Caches open MDAnalysis Universe objects so each XTC file is opened at
    most once regardless of how many frames are drawn from it.

    Returns:
        coords: float32 [T, n_ca, 3] — Angstroms
    """
    import MDAnalysis as mda  # cluster-only dependency; import lazily

    protein_dir = atlas_raw / protein_id
    pdb_path    = protein_dir / f"{protein_id}.pdb"

    universes: dict[int, mda.Universe] = {}

    def _get_universe(replica: int) -> mda.Universe:
        if replica not in universes:
            xtc = protein_dir / f"{protein_id}_prod_R{replica}_fit.xtc"
            universes[replica] = mda.Universe(str(pdb_path), str(xtc))
        return universes[replica]

    coords: list[np.ndarray] = []
    for frame_idx, replica in zip(frame_indices.tolist(), replica_ids.tolist()):
        u = _get_universe(int(replica))
        u.trajectory[int(frame_idx)]
        coords.append(u.select_atoms("name CA").positions.copy())

    return np.stack(coords, axis=0).astype(np.float32)  # [T, n_ca, 3]


# ---------------------------------------------------------------------------
# Chai-1 diffusion module decoder
# ---------------------------------------------------------------------------
#
# 2026-07 correction: an earlier version of this function called a
# nonexistent "structure_module.pt" component directly with a 4-argument
# single-shot signature. Verified against the real chai_lab source
# (chai_lab/chai1.py) on the cluster: the actual coordinate-generating
# component is "diffusion_module.pt", and it requires a full multi-step
# EDM/Karras denoising loop (noise schedule, second-order correction, random
# augmentation between steps) plus several other feature tensors built
# during trunk input embedding — not a single forward call.
#
# Rather than hand-reimplementing that ~300-line pipeline (real risk of
# silent transcription bugs), this monkeypatches ONLY chai_lab's own
# `load_exported("trunk.pt", ...)` call to return our substituted
# (single, pair) representation instead of running Chai-1's real
# sequence-derived trunk, then calls chai_lab's real, unmodified
# `run_folding_on_context` for everything else (feature construction,
# diffusion sampling, confidence scoring, CIF writing) — reusing 100% of
# their tested code for every part we don't intend to change.
#
# Correctness must be validated independently of whether FluxionDiT's
# generative model works: see scripts/verify_chai1_decode.py, which decodes
# a REAL already-extracted (single, pair) latent for a known ATLAS frame and
# checks the resulting structure is close (low RMSD) to that frame's real,
# known coordinates.

# Process-lifetime cache for loaded Chai-1 components (feature_embedding,
# bond_loss_input_proj, token_embedder, diffusion_module, confidence_head —
# everything EXCEPT trunk, which is always our fake substitute). Decoding an
# ensemble means calling chai1_decode_to_cif many times in a loop for the
# SAME protein; without this, every call would reload ~40GB of weights from
# scratch via load_exported, which is the dominant cost for anything beyond
# a single one-off decode. Keyed by (comp_key, device) so it's safe if a
# process ever decodes on more than one device.
_CHAI1_COMPONENT_CACHE: dict[tuple[str, str], object] = {}


def build_chai1_feature_context(fasta_path: Path):
    """Build the Chai-1 feature context once, for reuse across many
    chai1_decode_to_cif calls on the same protein (same sequence -> same
    context; only the substituted single/pair representation changes)."""
    from chai_lab.chai1 import make_all_atom_feature_context  # type: ignore[import]

    esm_cache = fasta_path.parent / "esm_cache"
    return make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=esm_cache,
        use_esm_embeddings=True,
        esm_device=torch.device("cpu"),
    )


def chai1_decode_to_cif(
    single: torch.Tensor,        # [L, d_single] or [1, L, d_single], raw Chai-1 scale
    pair: torch.Tensor,          # [L, L, d_pair] or [1, L, L, d_pair], raw Chai-1 scale
    fasta_path: Path,
    output_dir: Path,
    device: torch.device,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 1,
    seed: int | None = None,
    feature_context=None,
):
    """
    Decode a single/pair representation to 3D structure via Chai-1's real
    diffusion inference pipeline, substituting our own trunk representation
    in place of Chai-1's own sequence-derived one.

    Args:
        feature_context: optional pre-built context from
            build_chai1_feature_context(fasta_path) — pass this when calling
            in a loop over many samples for the SAME protein, to skip
            rebuilding it (and re-running ESM) on every call. If None,
            builds it fresh (fine for a single one-off decode).

    Returns the chai_lab StructureCandidates object. CIF files are written
    to output_dir/pred.model_idx_<i>.cif (one per num_diffn_samples).
    """
    try:
        import chai_lab.chai1 as chai1_module               # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "chai-lab is not installed or not importable. "
            "Activate the fluxion conda environment."
        ) from exc

    from unittest.mock import patch

    if single.dim() == 2:
        single = single.unsqueeze(0)
    if pair.dim() == 3:
        pair = pair.unsqueeze(0)
    single = single.to(device).float()
    pair = pair.to(device).float()

    if feature_context is None:
        feature_context = build_chai1_feature_context(fasta_path)

    real_load_exported = chai1_module.load_exported
    device_key = str(device)

    class _FakeTrunk:
        """Stand-in for the real trunk module: ignores every input and
        always returns our substituted (single, pair) representation,
        matching the (token_single_trunk_repr, token_pair_trunk_repr)
        tuple contract of the real trunk.forward()."""

        def forward(self, **kwargs):
            return single, pair

    def _patched_load_exported(comp_key: str, dev: torch.device):
        if comp_key == "trunk.pt":
            return _FakeTrunk()
        cache_key = (comp_key, device_key)
        if cache_key not in _CHAI1_COMPONENT_CACHE:
            _CHAI1_COMPONENT_CACHE[cache_key] = real_load_exported(comp_key, dev)
        return _CHAI1_COMPONENT_CACHE[cache_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    with patch.object(chai1_module, "load_exported", _patched_load_exported):
        candidates = chai1_module.run_folding_on_context(
            feature_context,
            output_dir=output_dir,
            num_trunk_recycles=1,  # irrelevant — fake trunk ignores recycling
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            seed=seed,
            device=device,
            low_memory=False,
        )

    return candidates


def extract_ca_from_cif(cif_path: Path) -> np.ndarray:
    """
    Extract Cα coordinates from a chai_lab-written CIF file.

    MDAnalysis (as installed on the cluster, 2.9.0) does not support CIF as
    a topology format — verified directly (`ValueError: 'CIF' isn't a valid
    topology format`) rather than assumed — so this parses the mmCIF
    `_atom_site` loop directly instead of pulling in a new CIF library.
    Column order is read from the header block rather than hardcoded, since
    mmCIF writers don't guarantee a fixed column order across versions.

    Returns:
        coords: float32 [n_ca, 3]
    """
    lines = Path(cif_path).read_text().splitlines()

    columns: list[str] = []
    rows: list[list[str]] = []
    in_atom_site_loop = False

    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()

        if stripped == "loop_":
            j = i + 1
            peek_columns = []
            while j < n and lines[j].strip().startswith("_atom_site."):
                peek_columns.append(lines[j].strip()[len("_atom_site."):])
                j += 1
            if peek_columns:
                columns = peek_columns
                in_atom_site_loop = True
                i = j
                continue
            in_atom_site_loop = False
            i += 1
            continue

        if in_atom_site_loop:
            if stripped.startswith("ATOM") or stripped.startswith("HETATM"):
                fields = stripped.split()
                if len(fields) == len(columns):
                    rows.append(fields)
                i += 1
                continue
            in_atom_site_loop = False
            i += 1
            continue

        i += 1

    if not columns:
        raise ValueError(f"No _atom_site loop found in {cif_path}")

    atom_id_idx = columns.index("label_atom_id")
    x_idx = columns.index("Cartn_x")
    y_idx = columns.index("Cartn_y")
    z_idx = columns.index("Cartn_z")

    ca_coords = [
        (float(row[x_idx]), float(row[y_idx]), float(row[z_idx]))
        for row in rows
        if row[atom_id_idx] == "CA"
    ]
    if not ca_coords:
        raise ValueError(f"No CA atoms found in {cif_path}")

    return np.array(ca_coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# RMSD evaluation
# ---------------------------------------------------------------------------

def _kabsch_rmsd(mobile: np.ndarray, ref: np.ndarray) -> float:
    """
    Kabsch-aligned RMSD between two point clouds (pure numpy, no MDAnalysis).

    Args:
        mobile: [N, 3] — coordinates to rotate
        ref:    [N, 3] — reference coordinates
    Returns:
        RMSD in the same units as the input coordinates.
    """
    mc = mobile - mobile.mean(0)
    rc = ref    - ref.mean(0)
    H  = mc.T @ rc
    U, _, Vt = np.linalg.svd(H)
    d  = np.sign(np.linalg.det(Vt.T @ U.T))
    R  = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    rotated = mc @ R.T
    return float(np.sqrt(((rotated - rc) ** 2).sum(-1).mean()))


def compute_pairwise_rmsds(
    coords: np.ndarray,   # [T, n_ca, 3]
    ref_idx: int = 0,
) -> np.ndarray:
    """Kabsch-aligned Cα RMSD of each frame against the reference frame."""
    ref = coords[ref_idx]
    return np.array(
        [_kabsch_rmsd(coords[i], ref) for i in range(len(coords))],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Trajectory writers
# ---------------------------------------------------------------------------

def write_pdb_trajectory(
    coords: np.ndarray,   # [T, n_atoms, 3]
    pdb_template: Path,
    out_pdb: Path,
    selection: str = "name CA",
) -> None:
    """Write a multi-frame PDB from [T, n_atoms, 3] Cα coordinates."""
    import MDAnalysis as mda  # cluster-only dependency; import lazily
    u   = mda.Universe(str(pdb_template))
    sel = u.select_atoms(selection)
    if coords.shape[1] != len(sel):
        raise ValueError(
            f"coords have {coords.shape[1]} atoms but '{selection}' "
            f"selects {len(sel)} atoms from {pdb_template.name}"
        )
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(out_pdb), sel.n_atoms) as writer:
        for frame_coords in coords:
            sel.positions = frame_coords
            writer.write(sel)
    print(f"Wrote {len(coords)}-frame PDB → {out_pdb}", flush=True)


def write_xtc_trajectory(
    coords: np.ndarray,   # [T, n_atoms, 3]
    pdb_template: Path,
    out_xtc: Path,
    selection: str = "name CA",
) -> None:
    """Write a multi-frame XTC from [T, n_atoms, 3] Cα coordinates."""
    import MDAnalysis as mda  # cluster-only dependency; import lazily
    u   = mda.Universe(str(pdb_template))
    sel = u.select_atoms(selection)
    if coords.shape[1] != len(sel):
        raise ValueError(
            f"coords have {coords.shape[1]} atoms but '{selection}' "
            f"selects {len(sel)} atoms from {pdb_template.name}"
        )
    out_xtc.parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(out_xtc), sel.n_atoms) as writer:
        for frame_coords in coords:
            sel.positions = frame_coords
            writer.write(sel)
    print(f"Wrote {len(coords)}-frame XTC → {out_xtc}", flush=True)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_nn_summary(
    nn_frames: np.ndarray,
    nn_replicas: np.ndarray,
    nn_distances: np.ndarray,
    rmsds: Optional[np.ndarray],
    out_dir: Path,
    protein_id: str,
) -> None:
    """Four-panel summary plot saved as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.", flush=True)
        return

    T      = len(nn_frames)
    t_axis = np.linspace(0.0, 1.0, T)
    n_panels = 4 if rmsds is not None else 3

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)

    axes[0].plot(t_axis, nn_frames, lw=0.8)
    axes[0].set_ylabel("NN frame index")
    axes[0].set_title(f"{protein_id} — Fluxion rollout analysis")

    axes[1].scatter(t_axis, nn_replicas, s=2, alpha=0.4)
    axes[1].set_ylabel("ATLAS replica")
    axes[1].set_yticks(sorted(set(nn_replicas.tolist())))

    axes[2].plot(t_axis, nn_distances, lw=0.8, color="tab:orange")
    axes[2].set_ylabel("NN distance (L2)")

    if rmsds is not None:
        axes[3].plot(t_axis, rmsds, lw=0.8, color="tab:green")
        axes[3].set_ylabel("Cα RMSD to t=0 (Å)")
        axes[3].set_xlabel("t")
    else:
        axes[2].set_xlabel("t")

    plt.tight_layout()
    out_path = out_dir / f"{protein_id}_rollout_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode FluxionDiT rollout trajectory to 3D coordinates"
    )
    parser.add_argument("--rollout",     required=True, type=Path,
                        help=".pt file produced by rollout.py or infer.py")
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--atlas-raw",   required=True, type=Path)
    parser.add_argument("--protein-id",  required=True)
    parser.add_argument("--out-dir",     required=True, type=Path)
    parser.add_argument("--mode",        default="nn", choices=["nn", "chai1"])
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading rollout from {args.rollout}", flush=True)
    rollout_data = torch.load(args.rollout, weights_only=False, map_location="cpu")
    trajectory   = rollout_data["trajectory"]   # [T+1, L, d_single]
    t_eval       = rollout_data["t_eval"]
    protein_id   = rollout_data.get("protein_id", args.protein_id)
    print(f"  shape={tuple(trajectory.shape)}  t=[{t_eval[0]:.2f}..{t_eval[-1]:.2f}]",
          flush=True)

    if args.mode == "nn":
        # ── Nearest-neighbor ────────────────────────────────────────────────
        print("Loading latent store …", flush=True)
        latents, frame_idxs, replica_ids = load_all_latents(
            args.latents_dir, protein_id
        )
        print(f"  {len(frame_idxs)} frames across replicas "
              f"{sorted(set(replica_ids))}", flush=True)

        print("Running nearest-neighbor search …", flush=True)
        nn_frames, nn_replicas, nn_distances = nn_decode(
            trajectory, latents, frame_idxs, replica_ids
        )
        print(f"  frame range [{nn_frames.min()}, {nn_frames.max()}]  "
              f"dist mean={nn_distances.mean():.3f}", flush=True)

        # Save NN mapping
        npz_path = args.out_dir / f"{protein_id}_nn_frames.npz"
        np.savez(
            npz_path,
            nn_frames=nn_frames,
            nn_replicas=nn_replicas,
            nn_distances=nn_distances,
            t_eval=t_eval.numpy(),
        )
        print(f"Saved NN data → {npz_path}", flush=True)

        # Extract Cα coordinates
        print("Extracting Cα coordinates from MD trajectories …", flush=True)
        ca_coords = extract_ca_trajectory(
            args.atlas_raw, protein_id, nn_frames, nn_replicas
        )
        print(f"  Cα shape: {ca_coords.shape}", flush=True)

        # RMSD progression
        rmsds = compute_pairwise_rmsds(ca_coords, ref_idx=0)
        print(f"  Cα RMSD range: {rmsds.min():.2f}–{rmsds.max():.2f} Å", flush=True)

        # Write PDB + XTC
        pdb_template = args.atlas_raw / protein_id / f"{protein_id}.pdb"
        write_pdb_trajectory(
            ca_coords, pdb_template,
            args.out_dir / f"{protein_id}_rollout_nn.pdb",
        )
        write_xtc_trajectory(
            ca_coords, pdb_template,
            args.out_dir / f"{protein_id}_rollout_nn.xtc",
        )

        plot_nn_summary(nn_frames, nn_replicas, nn_distances, rmsds,
                        args.out_dir, protein_id)

    elif args.mode == "chai1":
        # ── Chai-1 diffusion module (real multi-step sampling per frame) ──────
        pair_path = args.latents_dir / protein_id / "pair.pt"
        pair      = torch.load(pair_path, weights_only=True).float()
        fasta_path = args.latents_dir / protein_id / f"{protein_id}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        chai1_out_dir = args.out_dir / f"{protein_id}_chai1_cif"
        T = trajectory.shape[0]
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

        out_pt = args.out_dir / f"{protein_id}_rollout_chai1.pt"
        torch.save({"coords": torch.from_numpy(coords), "t_eval": t_eval}, out_pt)
        print(f"Saved Chai-1-decoded trajectory → {out_pt}", flush=True)


if __name__ == "__main__":
    main()
