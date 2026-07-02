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

Memory note
-----------
The static pair tensor is [batch_size, L, L, d_pair] float32 — at the
confirmed Chai-1 crop size L=512 and d_pair=256, that's ~4.3 GB per batch
at batch_size=16. This script streams one batch at a time and computes the
baseline, raw-model, and EMA-model losses on it before moving on, so peak
memory stays at O(1 batch) regardless of --n-batches — never materialize a
list of batches for "reuse" across metrics.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint  /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --proteins    "$PILOT_PROTEINS" \
        --n-batches   100 --batch-size 4 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from data_pipeline.dataset import build_multi_protein_dataset, collate_fn


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


@torch.no_grad()
def evaluate_stream(
    model: FluxionDiT,
    ema_model: FluxionDiT,
    matcher: FlowMatcher,
    dataloader: DataLoader,
    n_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Single streaming pass over n_batches batches, computing the trivial
    (v_pred=0) baseline, the raw model's loss, and the EMA model's loss on
    the identical samples for each batch — without ever holding more than
    one batch (and its potentially large static pair tensor) in memory.

    Returns:
        {"baseline": float, "raw_loss": float, "ema_loss": float,
         "n_tokens": int, "n_samples": int}
    """
    model.eval()
    ema_model.eval()

    total_baseline_sq = 0.0
    total_raw_loss = 0.0
    total_ema_loss = 0.0
    total_tokens = 0
    total_samples = 0

    it = iter(dataloader)
    for _ in range(n_batches):
        try:
            (z0, z1), pair, _fidx, padding_mask = next(it)
        except StopIteration:
            it = iter(dataloader)
            (z0, z1), pair, _fidx, padding_mask = next(it)

        z0, z1 = z0.to(device), z1.to(device)
        pair = pair.to(device)
        padding_mask = padding_mask.to(device)

        # Trivial (v_pred=0) baseline — token-weighted, matches FlowMatcher's
        # own padding-aware averaging.
        u_t = z1 - z0
        valid = (~padding_mask).float().unsqueeze(-1)
        total_baseline_sq += ((u_t ** 2) * valid).sum().item()
        total_tokens += valid.sum().item() * u_t.shape[-1]

        # Raw + EMA model loss on the exact same batch.
        total_raw_loss += matcher.compute_loss(model, z0, z1, pair, padding_mask).item() * z0.shape[0]
        total_ema_loss += matcher.compute_loss(ema_model, z0, z1, pair, padding_mask).item() * z0.shape[0]
        total_samples += z0.shape[0]

    return {
        "baseline": total_baseline_sq / total_tokens,
        "raw_loss": total_raw_loss / total_samples,
        "ema_loss": total_ema_loss / total_samples,
        "n_tokens": total_tokens,
        "n_samples": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--proteins",    type=str, default=None)
    parser.add_argument("--n-batches",   type=int, default=100)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
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

    matcher = FlowMatcher()

    print(f"\n[eval] Streaming {args.n_batches} batches of {args.batch_size} "
          f"({args.n_batches * args.batch_size} samples)...", flush=True)
    results = evaluate_stream(model, ema_model, matcher, dataloader, args.n_batches, device)

    baseline = results["baseline"]

    def _verdict(loss: float) -> str:
        pct = (1 - loss / baseline) * 100
        return f"[{'BELOW' if loss < baseline else 'AT/ABOVE'} baseline, {pct:+.2f}%]"

    print(f"\n[eval] ── Results ({results['n_samples']} samples, "
          f"{results['n_tokens']:.0f} tokens) ──────────")
    print(f"  Trivial baseline (v_pred=0): {baseline:.6f}")
    print(f"  Raw model loss:              {results['raw_loss']:.6f}  {_verdict(results['raw_loss'])}")
    print(f"  EMA model loss:              {results['ema_loss']:.6f}  {_verdict(results['ema_loss'])}")


if __name__ == "__main__":
    main()
