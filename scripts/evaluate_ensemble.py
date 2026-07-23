"""
scripts/evaluate_ensemble.py

Evaluate a generated static conformational ensemble (from
inference/sample_ensemble.py) against the real MD ensemble for the same
protein. This is the "does this actually work, not just does the loss look
good" check — three complementary metrics:

1. Diversity (mode-collapse detector): all-pairs Kabsch-RMSD distribution
   among generated structures vs. a matched-size random sample of real MD
   structures. A collapsed model (e.g. always predicting the mean latent)
   shows near-zero generated diversity relative to real.
2. RMSF profile correlation: per-residue root-mean-square fluctuation,
   Pearson correlation between generated and real MD profiles — the
   field-standard flexibility metric (BioEmu/Mac-Diff report this directly).
3. PCA projection overlap: project both ensembles onto the top-2 principal
   components of the REAL MD ensemble, then a JS-divergence-style overlap of
   the resulting 2D histograms (same spirit as Mac-Diff's JS-TIC, using PCA
   since generated samples are i.i.d. and have no time-ordering for TICA).

Also reports the fraction of unique nearest-neighbor real frames used by the
generated ensemble — a second, very direct collapse signal: if 1,000
generated samples all map to only a handful of real frames, that's collapse.

Usage
-----
python scripts/evaluate_ensemble.py \\
    --ensemble    results/1bxy_A_ensemble.pt \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --atlas-raw   /scratch/arjunc4/atlas_raw \\
    --out-dir     results/1bxy_A_eval \\
    --n-real-samples 1000
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
    nn_decode,
    extract_ca_trajectory,
    _kabsch_rmsd,
)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def kabsch_align(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Superimpose `mobile` [n_atoms, 3] onto `ref` [n_atoms, 3] via the Kabsch
    algorithm. Returns the aligned (rotated + translated) mobile coordinates,
    same shape as input.
    """
    mobile_mean = mobile.mean(0)
    ref_mean = ref.mean(0)
    mc = mobile - mobile_mean
    rc = ref - ref_mean
    H = mc.T @ rc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return mc @ R.T + ref_mean


def align_ensemble(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align every frame of coords [N, n_atoms, 3] onto ref [n_atoms, 3]."""
    return np.stack([kabsch_align(c, ref) for c in coords], axis=0)


# ---------------------------------------------------------------------------
# Diversity (mode-collapse detector)
# ---------------------------------------------------------------------------

def pairwise_rmsd_distribution(
    coords: np.ndarray,        # [N, n_atoms, 3], already aligned
    n_pairs: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    All-pairs (or a random subsample of pairs, for large N) Kabsch RMSD
    among ensemble members. Returns a 1D array of RMSD values.
    """
    N = coords.shape[0]
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if n_pairs is not None and len(all_pairs) > n_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs
    return np.array(
        [_kabsch_rmsd(coords[i], coords[j]) for i, j in pairs],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# RMSF
# ---------------------------------------------------------------------------

def compute_rmsf(aligned_coords: np.ndarray) -> np.ndarray:
    """
    Per-atom RMSF (root-mean-square fluctuation around the ensemble mean
    structure). aligned_coords: [N, n_atoms, 3], already Kabsch-aligned to a
    common reference. Returns [n_atoms].
    """
    mean_struct = aligned_coords.mean(axis=0)                      # [n_atoms, 3]
    sq_dev = ((aligned_coords - mean_struct) ** 2).sum(axis=-1)     # [N, n_atoms]
    return np.sqrt(sq_dev.mean(axis=0))                             # [n_atoms]


# ---------------------------------------------------------------------------
# PCA projection overlap
# ---------------------------------------------------------------------------

def fit_pca(aligned_coords: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA via SVD on flattened aligned coordinates. Returns (mean, components)."""
    flat = aligned_coords.reshape(aligned_coords.shape[0], -1)
    mean = flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, Vt[:n_components]


def project_pca(aligned_coords: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = aligned_coords.reshape(aligned_coords.shape[0], -1)
    return (flat - mean) @ components.T


def histogram_overlap_js(proj_a: np.ndarray, proj_b: np.ndarray, bins: int = 50) -> float:
    """
    Jensen-Shannon divergence between 2D histograms of two projected point
    sets (same spirit as Mac-Diff's JS-TIC metric). Lower = more similar
    distributions; 0 = identical, ~0.693 (ln 2) = disjoint supports.
    """
    combined = np.concatenate([proj_a, proj_b], axis=0)
    x_edges = np.linspace(combined[:, 0].min(), combined[:, 0].max(), bins + 1)
    y_edges = np.linspace(combined[:, 1].min(), combined[:, 1].max(), bins + 1)

    hist_a, _, _ = np.histogram2d(proj_a[:, 0], proj_a[:, 1], bins=[x_edges, y_edges])
    hist_b, _, _ = np.histogram2d(proj_b[:, 0], proj_b[:, 1], bins=[x_edges, y_edges])

    eps = 1e-6
    p = hist_a.flatten() + eps
    p = p / p.sum()
    q = hist_b.flatten() + eps
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(
    gen_rmsf: np.ndarray, real_rmsf: np.ndarray,
    gen_diversity: np.ndarray, real_diversity: np.ndarray,
    proj_gen: np.ndarray, proj_real: np.ndarray,
    out_dir: Path, protein_id: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.", flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(real_rmsf, label="real MD", lw=1.2)
    axes[0].plot(gen_rmsf, label="generated", lw=1.2)
    axes[0].set_xlabel("residue index")
    axes[0].set_ylabel("RMSF (Å)")
    axes[0].set_title(f"{protein_id} — RMSF profile")
    axes[0].legend()

    axes[1].hist(real_diversity, bins=40, alpha=0.5, label="real MD", density=True)
    axes[1].hist(gen_diversity, bins=40, alpha=0.5, label="generated", density=True)
    axes[1].set_xlabel("pairwise Cα RMSD (Å)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Ensemble diversity")
    axes[1].legend()

    axes[2].scatter(proj_real[:, 0], proj_real[:, 1], s=4, alpha=0.3, label="real MD")
    axes[2].scatter(proj_gen[:, 0], proj_gen[:, 1], s=4, alpha=0.3, label="generated")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title("PCA projection overlap")
    axes[2].legend()

    plt.tight_layout()
    out_path = out_dir / f"{protein_id}_ensemble_eval.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary plot → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate_ensemble] Loading generated ensemble from {args.ensemble}", flush=True)
    ensemble_data = torch.load(args.ensemble, weights_only=False, map_location="cpu")
    ensemble = ensemble_data["ensemble"]          # [N, L, d_single]
    protein_id = ensemble_data["protein_id"]
    N = ensemble.shape[0]
    print(f"  {N} generated samples for {protein_id!r}", flush=True)

    print("[evaluate_ensemble] Loading real latent store …", flush=True)
    latents, frame_idxs, replica_ids = load_all_latents(args.latents_dir, protein_id)
    total_real = len(frame_idxs)
    print(f"  {total_real} real frames across replicas {sorted(set(replica_ids))}", flush=True)

    print("[evaluate_ensemble] Nearest-neighbor decoding generated ensemble …", flush=True)
    nn_frames, nn_replicas, nn_distances = nn_decode(
        ensemble, latents, frame_idxs, replica_ids, batch_size=args.nn_batch_size,
    )
    unique_frac = len(set(zip(nn_frames.tolist(), nn_replicas.tolist()))) / N
    print(f"  NN mean distance={nn_distances.mean():.4f}  "
          f"unique-frame fraction={unique_frac:.3f}", flush=True)

    print("[evaluate_ensemble] Extracting generated-ensemble Cα coordinates …", flush=True)
    gen_coords = extract_ca_trajectory(args.atlas_raw, protein_id, nn_frames, nn_replicas)

    print(f"[evaluate_ensemble] Sampling {args.n_real_samples} real MD reference frames …",
          flush=True)
    rng = np.random.default_rng(args.seed)
    n_real = min(args.n_real_samples, total_real)
    real_idx = rng.choice(total_real, size=n_real, replace=False)
    real_frame_idxs = np.array([frame_idxs[i] for i in real_idx])
    real_replica_ids = np.array([replica_ids[i] for i in real_idx])
    real_coords = extract_ca_trajectory(args.atlas_raw, protein_id, real_frame_idxs, real_replica_ids)

    print("[evaluate_ensemble] Aligning both ensembles to a common reference …", flush=True)
    ref = real_coords[0]
    gen_aligned = align_ensemble(gen_coords, ref)
    real_aligned = align_ensemble(real_coords, ref)

    print("[evaluate_ensemble] Computing diversity distributions …", flush=True)
    max_pairs = args.max_diversity_pairs
    gen_diversity = pairwise_rmsd_distribution(gen_aligned, n_pairs=max_pairs, seed=args.seed)
    real_diversity = pairwise_rmsd_distribution(real_aligned, n_pairs=max_pairs, seed=args.seed)

    print("[evaluate_ensemble] Computing RMSF profiles …", flush=True)
    gen_rmsf = compute_rmsf(gen_aligned)
    real_rmsf = compute_rmsf(real_aligned)
    rmsf_correlation = float(np.corrcoef(gen_rmsf, real_rmsf)[0, 1])

    print("[evaluate_ensemble] Computing PCA projection overlap …", flush=True)
    pca_mean, pca_components = fit_pca(real_aligned, n_components=2)
    proj_real = project_pca(real_aligned, pca_mean, pca_components)
    proj_gen = project_pca(gen_aligned, pca_mean, pca_components)
    js_pca = histogram_overlap_js(proj_real, proj_gen)

    metrics = {
        "protein_id": protein_id,
        "n_generated": N,
        "n_real_reference": n_real,
        "nn_mean_distance": float(nn_distances.mean()),
        "nn_max_distance": float(nn_distances.max()),
        "unique_nn_frame_fraction": float(unique_frac),
        "diversity_generated_mean_A": float(gen_diversity.mean()),
        "diversity_generated_std_A": float(gen_diversity.std()),
        "diversity_real_mean_A": float(real_diversity.mean()),
        "diversity_real_std_A": float(real_diversity.std()),
        "diversity_ratio_gen_over_real": float(gen_diversity.mean() / real_diversity.mean()),
        "rmsf_correlation": rmsf_correlation,
        "js_pca_overlap": js_pca,
    }

    metrics_path = args.out_dir / f"{protein_id}_ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate_ensemble] Saved metrics → {metrics_path}", flush=True)

    plot_summary(
        gen_rmsf, real_rmsf, gen_diversity, real_diversity,
        proj_gen, proj_real, args.out_dir, protein_id,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a generated static ensemble against real MD data"
    )
    parser.add_argument("--ensemble",    required=True, type=Path)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--atlas-raw",   required=True, type=Path)
    parser.add_argument("--out-dir",     required=True, type=Path)
    parser.add_argument("--n-real-samples",     default=1000, type=int,
                        help="Number of real MD frames to subsample as reference (default 1000)")
    parser.add_argument("--max-diversity-pairs", default=20_000, type=int,
                        help="Cap on number of pairs for the diversity metric (default 20000)")
    parser.add_argument("--nn-batch-size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    metrics = run(args)

    print("\n── Evaluation complete ─────────────────────────────────", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
