"""
inference/sample_ensemble.py

Generate a static conformational ensemble for a protein: draw N i.i.d.
Gaussian noise samples and integrate the trained OT-CFM velocity field from
t=0 (noise) to t=1 (a generated equilibrium-sample latent), for the
static-ensemble pivot (see training/train_static.py).

Reuses inference/rollout.py's tested ODE integrators (euler/heun/rk4) and
load_model unchanged — the underlying math is identical to the temporal
rollout's ODE integration. The one difference: in the temporal case, z0 is a
raw-scale MD latent that rollout() pre-normalizes via (z0-mean)/std before
integrating. Here, z0 is fresh noise already living in the SAME normalized
space z1 occupies (training used z0 = torch.randn_like(z1) directly, no
separate normalization of the noise itself — see training/train_static.py).
So rollout() is called with norm_stats=None to skip its normalization, and
the final generated latent is denormalized back to raw Chai-1 scale manually.

Usage
-----
python inference/sample_ensemble.py \\
    --checkpoint  /scratch/arjunc4/fluxion_data/checkpoints_static/checkpoint.pt \\
    --latents-dir /scratch/arjunc4/fluxion_data/latents \\
    --protein-id  1bxy_A \\
    --n-samples   1000 \\
    --batch-size  32 \\
    --n-steps     100 \\
    --method      euler \\
    --out         results/1bxy_A_ensemble.pt \\
    --device      cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.rollout import load_model, rollout
from training.dit_model import FluxionDiT


@torch.inference_mode()
def generate_ensemble(
    model: FluxionDiT,
    pair: torch.Tensor,                          # [L, L, d_pair], raw scale
    n_samples: int,
    norm_stats: dict[str, torch.Tensor],
    batch_size: int = 32,
    n_steps: int = 100,
    method: str = "euler",
    device: torch.device | None = None,
    seed: int | None = None,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """
    Draw n_samples i.i.d. equilibrium-ensemble conformations by integrating
    the trained velocity field from fresh Gaussian noise (t=0) to t=1.

    Samples are drawn in chunks of at most `batch_size` to bound peak GPU
    memory for large ensembles (the field convention is ~1,000 samples per
    protein for evaluation, e.g. Mac-Diff/BioEmu).

    Args:
        noise_scale: multiplier on z0's standard deviation (default 1.0 =
            standard N(0,I), matching training). Diagnostic use: if
            evaluation shows the generated ensemble's diversity is a
            uniformly-scaled-down version of the real ensemble's (strong
            RMSF shape correlation, suppressed magnitude — variance
            underestimation, not broken/off-manifold generation), sampling
            with noise_scale > 1 tests whether that's recoverable at
            inference time without retraining (2026-07 diagnosis).

    Returns:
        ensemble: float32 [n_samples, L, d_single], raw Chai-1 latent scale.
    """
    if seed is not None:
        torch.manual_seed(seed)

    L = pair.shape[0]
    mean = norm_stats["mean"].float()
    std = norm_stats["std"].float()

    samples: list[torch.Tensor] = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        z0 = torch.randn(b, L, model.d_single) * noise_scale
        pair_b = pair.unsqueeze(0).expand(b, -1, -1, -1)

        traj, _ = rollout(
            model, z0, pair_b,
            n_steps=n_steps,
            t_end=1.0,
            method=method,
            device=device,
            norm_stats=None,  # noise is already in the model's native space
        )
        # traj: [n_steps+1, b, L, d_single] (normalized space). Guard against
        # rollout()'s squeeze(1) collapsing the batch dim when b == 1 (the
        # last chunk of an n_samples not evenly divisible by batch_size).
        z1_gen = traj[-1]
        if z1_gen.dim() == 2:
            z1_gen = z1_gen.unsqueeze(0)

        z1_gen = z1_gen * std + mean  # denormalize -> raw Chai-1 latent scale
        samples.append(z1_gen)
        remaining -= b

    return torch.cat(samples, dim=0)


def load_pair(latents_dir: Path, protein_id: str) -> torch.Tensor:
    """Load the static pair conditioning tensor for a protein."""
    pair_path = latents_dir / protein_id / "pair.pt"
    if not pair_path.exists():
        raise FileNotFoundError(
            f"pair.pt not found for {protein_id}. Run extract_pairs.slurm first."
        )
    return torch.load(pair_path, weights_only=True).float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static conformational ensemble from a trained "
                     "Fluxion static-ensemble checkpoint"
    )
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--protein-id",  required=True)
    parser.add_argument("--n-samples",   default=1000, type=int,
                        help="Number of i.i.d. ensemble members to generate (default 1000)")
    parser.add_argument("--batch-size",  default=32,   type=int,
                        help="Samples per ODE-integration batch (default 32)")
    parser.add_argument("--n-steps",     default=100,  type=int,
                        help="ODE integration steps (default 100)")
    parser.add_argument("--method",      default="euler",
                        choices=["euler", "heun", "rk4", "dopri5", "bosh3"])
    parser.add_argument("--no-ema",      action="store_true",
                        help="Use raw model weights instead of EMA")
    parser.add_argument("--seed",        default=None, type=int)
    parser.add_argument("--noise-scale", default=1.0,  type=float,
                        help="Multiplier on z0 std (default 1.0). See "
                             "generate_ensemble() docstring — diagnostic use "
                             "for testing variance-underestimation recovery.")
    parser.add_argument("--out",         required=True, type=Path)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[sample_ensemble] Loading model from {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint, device, use_ema=not args.no_ema)

    norm_stats_path = args.latents_dir / "norm_stats.pt"
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {norm_stats_path}. "
            "Run data_pipeline/compute_norm_stats.py first."
        )
    norm_stats = torch.load(norm_stats_path, weights_only=True)

    print(f"[sample_ensemble] Loading pair conditioning for {args.protein_id!r}", flush=True)
    pair = load_pair(args.latents_dir, args.protein_id)
    print(f"  pair shape: {tuple(pair.shape)}", flush=True)

    print(f"[sample_ensemble] Generating {args.n_samples} samples "
          f"(batch_size={args.batch_size}, method={args.method}, n_steps={args.n_steps}) …",
          flush=True)
    t0 = time.perf_counter()
    ensemble = generate_ensemble(
        model, pair, args.n_samples, norm_stats,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        method=args.method,
        device=device,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    elapsed = time.perf_counter() - t0
    print(f"  ensemble shape: {tuple(ensemble.shape)}  ({elapsed:.1f}s, "
          f"{elapsed / args.n_samples * 1000:.1f} ms/sample)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ensemble":    ensemble,          # [n_samples, L, d_single], raw scale
        "protein_id":  args.protein_id,
        "n_samples":   args.n_samples,
        "method":      args.method,
        "n_steps":     args.n_steps,
        "checkpoint":  str(args.checkpoint),
        "seed":        args.seed,
        "noise_scale": args.noise_scale,
    }, args.out)
    print(f"[sample_ensemble] Saved ensemble → {args.out}", flush=True)


if __name__ == "__main__":
    main()
