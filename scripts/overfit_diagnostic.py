#!/usr/bin/env python3
"""
scripts/overfit_diagnostic.py

Classic "can it overfit a tiny fixed batch" sanity check for the temporal
DiT training loop (training/train.py), run BEFORE spending any more time
tuning LR/warmup on the full 19-protein dataset.

Why this is the right next diagnostic
--------------------------------------
Today's training runs (10142030, 10151511, 10153220, 10190243) all show the
same pattern: stable, healthy, non-collapsing gradients (AdaLN-zero gate
weights confirmed moving meaningfully away from zero-init at step 5000),
but loss on the full dataset stays flat around ~0.0040-0.0041 across
batch sizes 4 and 8, tens of thousands of steps, never clearly beating the
trivial (v_pred=0) baseline (0.004596) or approaching the linear baseline
(0.001638).

Before concluding this needs a different LR/warmup schedule -- itself a
real (if cheap) time investment to iterate on -- the standard, cheaper,
more decisive check is: can this exact model/optimizer/loss combination
even memorize a HANDFUL of fixed examples? Loop the SAME small batch
through many optimizer steps and watch loss.

  * If loss collapses toward ~0 quickly: the optimization pipeline
    (gradients, loss computation, backprop, LR) is fundamentally sound.
    The flat loss on the full dataset is then a genuine scale/LR-magnitude/
    step-count problem for the real (harder, many-protein, random-t
    interpolation) task -- LR tuning or a longer run is then a
    well-justified next step, not a guess.
  * If it CANNOT overfit even 8 fixed examples: there's a real bug
    somewhere in the loss/gradient/masking path that no amount of LR
    tuning will fix -- needs to be found first.

Note on AdamW: raw gradient magnitude being small does NOT imply small
updates -- AdamW normalizes each parameter's step by its own running
gradient-magnitude estimate, so step size is governed mostly by LR
itself, not by how big the raw gradient happens to be. The AdaLN-zero
gate weights already moved substantially (0 -> 0.0005-0.013 mean by step
5000 in the real training runs) at LR=1e-4/1e-3, consistent with this --
parameters ARE moving by roughly the expected amount for that LR. This
test asks whether that movement, applied densely and repeatedly to a
FIXED tiny target, is enough to drive loss down at all.

Usage
-----
python scripts/overfit_diagnostic.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --protein 1a62_A \
    --batch-size 8 \
    --n-steps 1000 \
    --lr 1e-4 \
    --device cuda
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
from training.train import get_param_groups, compute_group_grad_norms
from data_pipeline.dataset import build_multi_protein_dataset, collate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=str,
                         help="Single protein ID to draw the fixed batch from "
                              "(kept single so every sample shares the same L, "
                              "no padding-size confound in this test).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    latents_dir = args.scratch_dir / "latents"
    norm_stats_path = latents_dir / "norm_stats.pt"
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats.pt not found at {norm_stats_path}")

    dataset = build_multi_protein_dataset(latents_dir, protein_ids=[args.protein])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # Grab exactly ONE batch and reuse it for every step -- this is the
    # whole point of the test, not a real training loop.
    (z0, z1), pair, _fidx, padding_mask = next(iter(loader))
    z0           = z0.to(device)
    z1           = z1.to(device)
    pair         = pair.to(device)
    padding_mask = padding_mask.to(device)
    print(f"[overfit] Fixed batch: {args.protein}, batch_size={args.batch_size}, "
          f"L={z0.shape[1]}", flush=True)

    model = FluxionDiT().to(device)
    matcher = FlowMatcher()
    optimizer = torch.optim.AdamW(
        get_param_groups(model, weight_decay=1e-5, lr=args.lr,
                          gate_lr_multiplier=args.gate_lr_multiplier),
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )

    model.train()
    losses: list[float] = []
    for step in range(1, args.n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)
        loss.backward()
        grad_norms = compute_group_grad_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if step % args.log_interval == 0 or step == 1:
            print(f"[overfit] Step {step:5d} | Loss: {loss_val:.6f} | "
                  f"GradNorm: {grad_norms['total_grad_norm']:.4f} | "
                  f"GateGradNorm: {grad_norms['gate_grad_norm']:.6f}", flush=True)

    first_10_mean = sum(losses[:10]) / 10
    last_10_mean  = sum(losses[-10:]) / 10
    reduction_pct = 100.0 * (1.0 - last_10_mean / first_10_mean) if first_10_mean > 0 else float("nan")

    print()
    print("-" * 72)
    print(f"[overfit] First 10 steps mean loss: {first_10_mean:.6f}")
    print(f"[overfit] Last  10 steps mean loss: {last_10_mean:.6f}")
    print(f"[overfit] Reduction: {reduction_pct:.2f}%")
    if reduction_pct > 90:
        verdict = ("Loss collapsed sharply -- the optimization pipeline is sound. "
                   "The full-dataset flat loss is a scale/LR/step-count problem "
                   "for the real task, not a broken training loop.")
    elif reduction_pct > 30:
        verdict = ("Partial reduction -- some capacity to fit this exact batch, "
                   "but not a clean collapse. Ambiguous; consider more steps or "
                   "a smaller batch before concluding either way.")
    else:
        verdict = ("Little to no reduction on a TINY fixed batch this model should "
                   "trivially be able to memorize. This points at a real bug in the "
                   "loss/gradient/masking path, not an LR magnitude issue -- "
                   "investigate before touching any hyperparameter.")
    print(f"[verdict] {verdict}")
    print("-" * 72)


if __name__ == "__main__":
    main()
