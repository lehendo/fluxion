#!/usr/bin/env python3
"""
scripts/evaluate_checkpoint.py

Compares a trained checkpoint's real FlowMatcher loss against the trivial
(v_pred=0) baseline on the *same* sampled batches, and separately checks
whether the AdaLN-zero gates and final projection have moved away from their
zero initialization.

Why both checks matter
-----------------------
If training loss sits at/near the trivial baseline, there are two very
different explanations that look identical from the loss curve alone:

  1. Zero-initialized parameters are still ~0 -> gradients aren't reaching
     them (an optimizer/LR/gradient-flow bug). The network is still doing
     the zero-init "predict nothing" thing by construction, regardless of
     how many steps have run.

  2. Those parameters HAVE moved meaningfully away from zero, but loss is
     still ~at baseline anyway -> the model has learned something, but that
     something isn't reducing loss below what static statistics already
     achieve. For frame-to-frame MD dynamics with no velocity/momentum
     history in the conditioning, this can be close to the true achievable
     floor: a single structural snapshot alone often carries little
     information about instantaneous momentum in an equilibrated system.

Case 1 calls for debugging the optimizer/LR/architecture. Case 2 calls for
rethinking the conditioning (e.g. feeding a short history of frames instead
of a single z_t) rather than chasing a training bug that isn't there.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint  /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS" \
        --n-batches   200 --batch-size 16 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from data_pipeline.dataset import build_multi_protein_dataset, collate_fn
from compute_trivial_baseline import compute_trivial_baseline_mse

Batch = tuple


def summarize_wakeup(model: FluxionDiT) -> dict[str, float]:
    """
    Mean |weight| for every parameter that starts at exactly zero (AdaLN-zero
    gate projections in every block, and the final output projection). If
    these are still ~0 after many training steps, gradients aren't reaching
    them and the network is still architecturally equivalent to its
    zero-init identity/zero-output state.
    """
    stats: dict[str, float] = {
        "final_layer.linear.weight":    model.final_layer.linear.weight.abs().mean().item(),
        "final_layer.adaLN_mod.weight": model.final_layer.adaLN_mod[-1].weight.abs().mean().item(),
    }
    for i, block in enumerate(model.blocks):
        stats[f"block[{i}].adaLN_mod.weight"] = block.adaLN_mod[-1].weight.abs().mean().item()
    return stats


def collect_batches(dataloader: DataLoader, n_batches: int) -> list[Batch]:
    """Pulls exactly n_batches batches from dataloader into a fixed list, so
    every downstream metric (baseline, raw model, EMA model) is computed on
    the identical sample — an apples-to-apples comparison with no risk of
    RNG-determinism drift across separately-seeded DataLoader instances."""
    batches: list[Batch] = []
    it = iter(dataloader)
    for _ in range(n_batches):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(dataloader)
            batches.append(next(it))
    return batches


@torch.no_grad()
def compute_model_loss(
    model: FluxionDiT,
    matcher: FlowMatcher,
    batches: list[Batch],
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    for (z0, z1), pair, _fidx, padding_mask in batches:
        z0, z1 = z0.to(device), z1.to(device)
        pair = pair.to(device)
        padding_mask = padding_mask.to(device)

        loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)
        total_loss += loss.item() * z0.shape[0]
        total_n += z0.shape[0]

    return total_loss / total_n, total_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--proteins",    type=str, default=None)
    parser.add_argument("--n-batches",   type=int, default=200)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[eval] Loading checkpoint from {args.checkpoint}", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[eval] Checkpoint step={state.get('step', '?')} epoch={state.get('epoch', '?')}",
          flush=True)

    model = FluxionDiT()
    model.load_state_dict(state["model"])
    model.to(device)

    ema_model = FluxionDiT()
    ema_model.load_state_dict(state["ema"])
    ema_model.to(device)

    print("\n[eval] Parameter wake-up check (mean |weight| of zero-initialized params):",
          flush=True)
    stats = summarize_wakeup(model)
    for name, val in stats.items():
        print(f"  {name:32s} {val:.6f}", flush=True)
    n_awake = sum(1 for v in stats.values() if v > 1e-4)
    print(f"[eval] {n_awake}/{len(stats)} zero-init params have moved meaningfully "
          f"away from zero.", flush=True)

    protein_ids = [p.strip() for p in args.proteins.split(",")] if args.proteins else None
    dataset = build_multi_protein_dataset(args.latents_dir, protein_ids=protein_ids)

    torch.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )

    print(f"\n[eval] Sampling {args.n_batches} fixed batches for comparison...", flush=True)
    batches = collect_batches(dataloader, args.n_batches)

    matcher = FlowMatcher()

    baseline, n_baseline = compute_trivial_baseline_mse(batches, len(batches))
    raw_loss, n_raw       = compute_model_loss(model, matcher, batches, device)
    ema_loss, n_ema        = compute_model_loss(ema_model, matcher, batches, device)

    def _verdict(loss: float) -> str:
        pct = (1 - loss / baseline) * 100
        return f"[{'BELOW' if loss < baseline else 'AT/ABOVE'} baseline, {pct:+.2f}%]"

    print(f"\n[eval] ── Results (same {n_baseline} samples for all three) ──────────")
    print(f"  Trivial baseline (v_pred=0): {baseline:.6f}")
    print(f"  Raw model loss:              {raw_loss:.6f}  {_verdict(raw_loss)}")
    print(f"  EMA model loss:              {ema_loss:.6f}  {_verdict(ema_loss)}")


if __name__ == "__main__":
    main()
