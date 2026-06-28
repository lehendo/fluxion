"""
inference/rollout.py

ODE trajectory rollout using the trained FluxionDiT velocity field.

Given an initial single-token latent z_0 and the corresponding static pair
representation, integrates  dz/dt = v_θ(z, pair, t)  from t=0 to t=T to
produce a predicted conformational trajectory in latent space.

Usage (command-line):
    python inference/rollout.py \
        --checkpoint  /scratch/arjunc4/fluxion_checkpoints/checkpoint.pt \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --protein-id  16pk_A \
        --frame       0 \
        --n-steps     200 \
        --method      euler \
        --out         rollout_16pk_A.pt \
        --device      cuda

Output:
    .pt file containing a dict:
        "trajectory"  float32 [T+1, L, d_single]  — latent frames t=0..T
        "t_eval"      float32 [T+1]                — time points
        "protein_id"  str
        "frame_idx"   int                          — source MD frame index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dit_model import FluxionDiT


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool = True,
) -> FluxionDiT:
    """Load FluxionDiT from a training checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Instantiate with default architecture (same as training defaults)
    model = FluxionDiT()

    key = "ema" if (use_ema and "ema" in state) else "model"
    missing, unexpected = model.load_state_dict(state[key], strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")

    model.to(device).eval()
    print(f"Loaded {'EMA ' if use_ema and 'ema' in state else ''}model "
          f"from step {state.get('step', '?')}  ({model.param_count()} params)", flush=True)
    return model


# ---------------------------------------------------------------------------
# Fixed-step integrators
# ---------------------------------------------------------------------------

def euler(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
) -> torch.Tensor:
    """Euler integration. Returns [T, ...] trajectory at each t in t_eval."""
    frames = [z0]
    z = z0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        t_scalar = t_eval[i].expand(z.shape[0])
        z = z + dt * func(z, t_scalar)
        frames.append(z)
    return torch.stack(frames, dim=0)


def heun(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
) -> torch.Tensor:
    """Heun (RK2 predictor-corrector). 2× NFE vs Euler, smoother trajectories."""
    frames = [z0]
    z = z0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        t0 = t_eval[i].expand(z.shape[0])
        t1 = t_eval[i + 1].expand(z.shape[0])
        k1 = func(z, t0)
        z_pred = z + dt * k1
        k2 = func(z_pred, t1)
        z = z + dt * 0.5 * (k1 + k2)
        frames.append(z)
    return torch.stack(frames, dim=0)


def rk4(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
) -> torch.Tensor:
    """Classical RK4. 4× NFE vs Euler, highest accuracy among fixed-step methods."""
    frames = [z0]
    z = z0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        t_i = t_eval[i].expand(z.shape[0])
        t_m = ((t_eval[i] + t_eval[i + 1]) / 2.0).expand(z.shape[0])
        t_n = t_eval[i + 1].expand(z.shape[0])
        k1 = func(z, t_i)
        k2 = func(z + dt * 0.5 * k1, t_m)
        k3 = func(z + dt * 0.5 * k2, t_m)
        k4 = func(z + dt * k3, t_n)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        frames.append(z)
    return torch.stack(frames, dim=0)


# ---------------------------------------------------------------------------
# Adaptive integrator (torchdiffeq)
# ---------------------------------------------------------------------------

def _odeint_adaptive(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> torch.Tensor:
    """
    Wraps torchdiffeq.odeint.  Requires `pip install torchdiffeq`.

    torchdiffeq expects func(t, y) — note reversed argument order vs our
    internal convention (z, t), so we wrap with a lambda.
    """
    try:
        from torchdiffeq import odeint  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "torchdiffeq is not installed.  "
            "Run: pip install torchdiffeq\n"
            "Or use --method euler / heun / rk4 instead."
        ) from exc

    # torchdiffeq flattens state — z0 must be 1-D or torchdiffeq does it.
    # We keep batch dim B=1, so z0 is [1, L, d_single].
    trajectory = odeint(
        lambda t, z: func(z, t.expand(z.shape[0])),
        y0=z0,
        t=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return trajectory  # [T, 1, L, d_single]


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def rollout(
    model: FluxionDiT,
    z0: torch.Tensor,      # [L, d_single] or [1, L, d_single]
    pair: torch.Tensor,    # [L, L, d_pair] or [1, L, L, d_pair]
    n_steps: int = 100,
    t_end: float = 1.0,
    method: str = "euler",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate the velocity field from t=0 to t=t_end.

    Args:
        model:    trained FluxionDiT (eval mode, on `device`)
        z0:       initial latent [L, d] or [1, L, d]
        pair:     conditioning pair rep [L, L, d_p] or [1, L, L, d_p]
        n_steps:  number of integration steps (for fixed-step methods)
        t_end:    integration end time (default 1.0 = full OT-CFM trajectory)
        method:   "euler" | "heun" | "rk4" | "dopri5" | "bosh3" | ...
        device:   move tensors here if provided

    Returns:
        trajectory: [n_steps+1, L, d_single] float32 — latent at each step
        t_eval:     [n_steps+1] float32 — corresponding time points
    """
    if device is not None:
        z0 = z0.to(device)
        pair = pair.to(device)

    # Ensure batch dim is present
    if z0.dim() == 2:
        z0 = z0.unsqueeze(0)    # [1, L, d_single]
    if pair.dim() == 3:
        pair = pair.unsqueeze(0)  # [1, L, L, d_pair]

    # Pre-cast pair to match model dtype for efficiency; kept fixed throughout
    pair = pair.to(next(model.parameters()).dtype)

    # Closure captures `model` and `pair` — only z and t vary during integration
    def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = z.to(pair.dtype)
        return model(z, pair, t)

    t_eval = torch.linspace(0.0, t_end, n_steps + 1, device=z0.device)
    z0 = z0.to(pair.dtype)

    fixed_methods = {"euler": euler, "heun": heun, "rk4": rk4}

    if method in fixed_methods:
        traj = fixed_methods[method](velocity_fn, z0, t_eval)  # [T+1, 1, L, d]
    else:
        traj = _odeint_adaptive(velocity_fn, z0, t_eval, method)

    # Remove batch dim: [T+1, 1, L, d] → [T+1, L, d]
    traj = traj.squeeze(1).float().cpu()
    t_eval = t_eval.float().cpu()

    return traj, t_eval


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_z0_from_latents(
    latents_dir: Path,
    protein_id: str,
    frame_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load initial single-token latent z0 and static pair from the latent store.

    Returns:
        z0:   [L, d_single] float32
        pair: [L, L, d_pair] float32
    """
    protein_dir = latents_dir / protein_id
    pair_path = protein_dir / "pair.pt"
    if not pair_path.exists():
        raise FileNotFoundError(
            f"pair.pt not found for {protein_id}.  "
            "Run extract_pairs.slurm first."
        )
    pair = torch.load(pair_path, weights_only=True).float()  # [L, L, d_pair]

    import json
    import h5py
    import numpy as np

    with open(protein_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Walk shards to find the requested frame
    frames_seen = 0
    seen_paths: set[str] = set()
    for shard_info in manifest[protein_id]["shards"]:
        shard_path = protein_dir / shard_info["path"]
        if str(shard_path) in seen_paths:
            continue
        seen_paths.add(str(shard_path))
        n = shard_info["n_frames"]
        if frames_seen + n > frame_idx:
            local_idx = frame_idx - frames_seen
            with h5py.File(shard_path, "r") as hf:
                single = torch.from_numpy(
                    np.array(hf["single"][local_idx])
                ).float()  # [L, d_single]
            return single, pair
        frames_seen += n

    raise IndexError(
        f"frame_idx={frame_idx} out of range "
        f"(protein {protein_id} has {frames_seen} frames)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FluxionDiT ODE trajectory rollout")
    parser.add_argument("--checkpoint",  required=True,  type=Path)
    parser.add_argument("--latents-dir", required=True,  type=Path)
    parser.add_argument("--protein-id",  required=True)
    parser.add_argument("--frame",       default=0,      type=int,
                        help="MD frame index to use as z_0 (default 0)")
    parser.add_argument("--n-steps",     default=200,    type=int,
                        help="Integration steps (default 200)")
    parser.add_argument("--t-end",       default=1.0,    type=float,
                        help="End time for integration (default 1.0)")
    parser.add_argument("--method",      default="euler",
                        choices=["euler", "heun", "rk4", "dopri5", "bosh3"],
                        help="ODE integrator (default: euler)")
    parser.add_argument("--no-ema",      action="store_true",
                        help="Use raw model weights instead of EMA")
    parser.add_argument("--out",         required=True,  type=Path,
                        help="Output .pt path for the rollout trajectory")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device, use_ema=not args.no_ema)

    # Load initial latent
    print(f"Loading z_0 from protein={args.protein_id!r}, frame={args.frame}", flush=True)
    z0, pair = load_z0_from_latents(args.latents_dir, args.protein_id, args.frame)
    print(f"  z0 shape: {tuple(z0.shape)}", flush=True)
    print(f"  pair shape: {tuple(pair.shape)}", flush=True)

    # Integrate
    print(f"Integrating with method={args.method!r}, n_steps={args.n_steps} ...", flush=True)
    trajectory, t_eval = rollout(
        model, z0, pair,
        n_steps=args.n_steps,
        t_end=args.t_end,
        method=args.method,
        device=device,
    )
    print(f"Trajectory shape: {tuple(trajectory.shape)}", flush=True)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trajectory": trajectory,    # [T+1, L, d_single]
        "t_eval":     t_eval,        # [T+1]
        "protein_id": args.protein_id,
        "frame_idx":  args.frame,
        "method":     args.method,
        "n_steps":    args.n_steps,
        "checkpoint": str(args.checkpoint),
    }, args.out)
    print(f"Saved rollout → {args.out}", flush=True)


if __name__ == "__main__":
    main()
