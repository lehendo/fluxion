#!/usr/bin/env python3
"""
scripts/compute_nonlinear_probe.py

Tests whether a small NONLINEAR model can predict the future state z1 from
(z0, z_prev) better than the linear baselines already computed by
compute_restoring_force_baseline.py and compute_velocity_baseline.py — both
of which found exactly 0.00% predictability at lag=1 across all 14 pilot
proteins. This is the last cheap check before treating a single frozen
Chai-1 snapshot (with or without one frame of history) as carrying no
exploitable dynamics signal at all, linear or otherwise.

Why this is a genuinely different kind of tool than the earlier diagnostics
-----------------------------------------------------------------------------
Every earlier diagnostic (position-mean, restoring-force, velocity) was a
closed-form computation from streaming sufficient statistics — no training,
no randomness, exact answers. A nonlinear probe requires actual gradient-
based training of a model with real capacity, which introduces a real risk
the earlier tools didn't have: a high-capacity model can show FAKE
"improvement" by memorizing training-set noise rather than learning anything
generalizable. The closed-form linear tests couldn't do this (an OLS fit
with 1-2 parameters per channel has essentially no room to overfit on tens
of thousands of samples), but an MLP with hundreds of thousands of
parameters can, easily.

The fix: train and evaluate with a HARD split at the PROTEIN level — the
validation proteins are never seen in any form during training, not even
different time windows of them. Only genuine improvement on held-out
PROTEINS counts as real evidence. All reported numbers are validation-set
numbers; training-set loss is diagnostic only and never used to draw a
conclusion.

Method
-------
1. Pool (z_prev, z0, z1) triplets across all positions and frames of the
   TRAIN proteins into one large tensor (subsampled for tractability), same
   for a held-out set of VAL proteins.
2. Fit three predictors of z1 on the train pool, using the exact same
   "regress z1 directly, never a difference" design as the earlier tools
   (avoids the mechanical self-inclusion confound entirely — see
   compute_restoring_force_baseline.py's docstring for why that matters):
     - trivial: predict z1 by its train-set mean
     - linear:  closed-form OLS regression of z1 on (z0, z_prev)
     - MLP:     small feedforward network (not the 40M-param DiT — a few
                hundred thousand parameters), trained with Adam + early
                stopping on VAL loss
3. Report all three predictors' MSE on the held-out VAL proteins.

Interpretation
---------------
- MLP val MSE close to linear val MSE (both close to trivial): no
  nonlinear structure either — strong evidence a single Chai-1 snapshot (or
  even two consecutive ones) carries no exploitable dynamics signal in this
  latent space, linear or nonlinear. Points at the representation itself
  needing to change, not the conditioning scheme sitting on top of it.
- MLP val MSE meaningfully below linear val MSE: real nonlinear structure
  exists that a linear probe couldn't see — reopens the "this is an
  architecture/training problem with the current DiT, not a representation
  dead end" hypothesis, since the (much larger, much more expressive) DiT
  should in principle be able to find whatever this small MLP found.

Usage:
    python scripts/compute_nonlinear_probe.py \
        --latents-dir /scratch/arjunc4/fluxion_data/latents \
        --train-proteins "1a62_A,1ab1_A,..." \
        --val-proteins   "1bxy_A,1bxy_B,1btk_A" \
        --lag 1 --velocity-lag 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline.dataset import LatentDataset


def _collate_triplet(batch):
    zprev_list, z0_list, z1_list = [], [], []
    for (z_prev, z0, z1), _pair, _fidx in batch:
        zprev_list.append(z_prev)
        z0_list.append(z0)
        z1_list.append(z1)
    return torch.stack(zprev_list), torch.stack(z0_list), torch.stack(z1_list)


@torch.no_grad()
def load_pooled_samples(
    latents_dir: Path,
    protein_ids: list[str],
    norm_stats: dict[str, torch.Tensor] | None,
    lag: int,
    velocity_lag: int,
    max_samples_per_protein: int,
    seed: int,
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pools (z_prev, z0, z1) across every position and (non-self-paired) frame
    of the given proteins into three flat [N, d] tensors, subsampled to
    roughly max_samples_per_protein rows per protein.

    Memory note: a single pilot protein has ~30,000 frames x up to 512
    positions ~= 15M (position, frame) pairs — at 3*d_single*4 bytes each
    that's over 70 GB if accumulated in full before subsampling. This
    function applies per-sample Bernoulli thinning INSIDE the streaming
    loop (keep_prob estimated from len(dataset) and the sequence length L
    seen in the first batch) so peak memory stays close to
    max_samples_per_protein throughout, never anything close to the full
    per-protein size. A final hard trim guards against the small overshoot
    inherent to random thinning.

    Returns:
        z_prev, z0, z1: float32 [N, d_single]
    """
    torch.manual_seed(seed)
    zprev_chunks, z0_chunks, z1_chunks = [], [], []

    for protein_id in protein_ids:
        ds = LatentDataset(
            latents_dir, protein_id, load_pair=False, norm_stats=norm_stats,
            lag=lag, velocity_lag=velocity_lag,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_collate_triplet, drop_last=False,
        )

        n_frames = len(ds)
        keep_prob: float | None = None  # computed once L is known, from the first batch
        prot_zprev, prot_z0, prot_z1 = [], [], []

        for zp, z0, z1 in loader:
            # Exclude self-paired boundary artifacts (see
            # compute_velocity_baseline.py for why this matters).
            head_selfpair = (zp - z0).abs().sum(dim=(1, 2)) == 0
            tail_selfpair = (z1 - z0).abs().sum(dim=(1, 2)) == 0
            valid = ~(head_selfpair | tail_selfpair)
            zp, z0, z1 = zp[valid], z0[valid], z1[valid]
            if zp.shape[0] == 0:
                continue

            if keep_prob is None:
                L = zp.shape[1]
                total_estimate = max(n_frames * L, 1)
                keep_prob = min(1.0, max_samples_per_protein / total_estimate)

            if keep_prob < 1.0:
                # Thin across the ENTIRE stream (not just until we've "seen
                # enough") for uniform coverage of the whole trajectory —
                # stopping early once near the target count would bias the
                # sample toward the beginning of the trajectory instead.
                # Flatten [B, L, d] -> [B*L, d] before thinning, matching the
                # final pooled shape.
                zp_flat = zp.reshape(-1, zp.shape[-1])
                z0_flat = z0.reshape(-1, z0.shape[-1])
                z1_flat = z1.reshape(-1, z1.shape[-1])
                keep = torch.rand(zp_flat.shape[0]) < keep_prob
                zp, z0, z1 = zp_flat[keep], z0_flat[keep], z1_flat[keep]
                if zp.shape[0] == 0:
                    continue
            # Flatten [B, L, d] -> [B*L, d]: every residue position is an
            # independent training example for this pooled, position-
            # agnostic probe.
            prot_zprev.append(zp.reshape(-1, zp.shape[-1]))
            prot_z0.append(z0.reshape(-1, z0.shape[-1]))
            prot_z1.append(z1.reshape(-1, z1.shape[-1]))

        prot_zprev = torch.cat(prot_zprev, dim=0)
        prot_z0    = torch.cat(prot_z0, dim=0)
        prot_z1    = torch.cat(prot_z1, dim=0)

        if prot_zprev.shape[0] > max_samples_per_protein:
            perm = torch.randperm(prot_zprev.shape[0])[:max_samples_per_protein]
            prot_zprev, prot_z0, prot_z1 = prot_zprev[perm], prot_z0[perm], prot_z1[perm]

        zprev_chunks.append(prot_zprev)
        z0_chunks.append(prot_z0)
        z1_chunks.append(prot_z1)
        print(f"[probe] {protein_id}: pooled {prot_zprev.shape[0]} samples", flush=True)

    return torch.cat(zprev_chunks, dim=0), torch.cat(z0_chunks, dim=0), torch.cat(z1_chunks, dim=0)


def fit_linear_baseline(
    z_prev_train: torch.Tensor,
    z0_train: torch.Tensor,
    z1_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Closed-form multivariate OLS: z1 ~= a + k0*z0 + k1*z_prev, fit per output
    channel using torch.linalg.lstsq (design matrix [1, z0, z_prev] per
    channel, shared across all d_single output channels via broadcasting
    would require per-channel design too since inputs vary by channel — here
    we fit one SHARED regression using each channel's own 3 scalar features
    as a per-channel regression, matching the diagonal/per-channel design of
    the earlier closed-form tools).

    Returns:
        (a, k0, k1): each float32 [d_single] — one intercept/slope pair per
        channel, fit independently (matching the per-(position,channel)
        design of compute_velocity_baseline.py, but pooled across positions
        here since this probe treats every position as an interchangeable
        sample).
    """
    d = z0_train.shape[1]
    a  = torch.zeros(d)
    k0 = torch.zeros(d)
    k1 = torch.zeros(d)
    for c in range(d):
        X = torch.stack([
            torch.ones(z0_train.shape[0]), z0_train[:, c], z_prev_train[:, c],
        ], dim=1).double()
        y = z1_train[:, c].double()
        sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
        a[c], k0[c], k1[c] = sol[0].float(), sol[1].float(), sol[2].float()
    return a, k0, k1


def linear_predict(
    z_prev: torch.Tensor, z0: torch.Tensor, a: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor,
) -> torch.Tensor:
    return a.unsqueeze(0) + k0.unsqueeze(0) * z0 + k1.unsqueeze(0) * z_prev


class MLPProbe(nn.Module):
    """Small feedforward nonlinear predictor of z1 from (z0, z_prev) — NOT
    the 40M-param DiT, a few hundred thousand parameters at most, just
    enough capacity to detect nonlinear structure if it exists."""

    def __init__(self, d_single: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_single, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_single),
        )

    def forward(self, z_prev: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z0, z_prev], dim=-1))


def train_mlp(
    z_prev_train: torch.Tensor, z0_train: torch.Tensor, z1_train: torch.Tensor,
    z_prev_val: torch.Tensor, z0_val: torch.Tensor, z1_val: torch.Tensor,
    hidden: int = 256,
    lr: float = 1e-3,
    batch_size: int = 1024,
    max_epochs: int = 30,
    patience: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[nn.Module, float, list[float]]:
    """
    Trains MLPProbe with early stopping on held-out val MSE (never trained
    on). Returns (best_model, best_val_mse, val_mse_history).
    """
    torch.manual_seed(seed)
    device_t = torch.device(device)
    model = MLPProbe(d_single=z0_train.shape[1], hidden=hidden).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    z_prev_val_d = z_prev_val.to(device_t)
    z0_val_d     = z0_val.to(device_t)
    z1_val_d     = z1_val.to(device_t)

    n = z0_train.shape[0]
    best_val_mse = float("inf")
    best_state = None
    epochs_since_improvement = 0
    val_history: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            zp = z_prev_train[idx].to(device_t)
            z0 = z0_train[idx].to(device_t)
            z1 = z1_train[idx].to(device_t)

            optimizer.zero_grad(set_to_none=True)
            pred = model(zp, z0)
            loss = nn.functional.mse_loss(pred, z1)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(z_prev_val_d, z0_val_d)
            val_mse = nn.functional.mse_loss(val_pred, z1_val_d).item()
        val_history.append(val_mse)
        print(f"[probe] epoch {epoch+1}/{max_epochs}  val_mse={val_mse:.6f}", flush=True)

        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[probe] Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mse, val_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents-dir", required=True, type=Path)
    parser.add_argument("--train-proteins", type=str, required=True,
                         help="Comma-separated protein IDs for training")
    parser.add_argument("--val-proteins", type=str, required=True,
                         help="Comma-separated protein IDs for validation "
                              "(must not overlap with --train-proteins)")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--velocity-lag", type=int, default=1)
    parser.add_argument("--max-samples-per-protein", type=int, default=150_000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_proteins = [p.strip() for p in args.train_proteins.split(",")]
    val_proteins   = [p.strip() for p in args.val_proteins.split(",")]
    overlap = set(train_proteins) & set(val_proteins)
    if overlap:
        raise ValueError(f"train/val proteins must not overlap: {overlap}")

    norm_stats_path = args.latents_dir / "norm_stats.pt"
    norm_stats = None
    if norm_stats_path.exists():
        norm_stats = torch.load(norm_stats_path, weights_only=True)
        print(f"[probe] Loaded normalization stats from {norm_stats_path}", flush=True)
    else:
        print(f"[probe] No normalization stats found at {norm_stats_path} — "
              f"proceeding WITHOUT latent normalization.", flush=True)

    print(f"[probe] Train proteins ({len(train_proteins)}): {train_proteins}", flush=True)
    print(f"[probe] Val proteins ({len(val_proteins)}): {val_proteins}", flush=True)

    print("\n[probe] Pooling training samples...", flush=True)
    zp_train, z0_train, z1_train = load_pooled_samples(
        args.latents_dir, train_proteins, norm_stats, args.lag, args.velocity_lag,
        args.max_samples_per_protein, args.seed, num_workers=args.num_workers,
    )
    print(f"[probe] Total train samples: {zp_train.shape[0]}", flush=True)

    print("\n[probe] Pooling validation samples...", flush=True)
    zp_val, z0_val, z1_val = load_pooled_samples(
        args.latents_dir, val_proteins, norm_stats, args.lag, args.velocity_lag,
        args.max_samples_per_protein, args.seed + 1, num_workers=args.num_workers,
    )
    print(f"[probe] Total val samples: {zp_val.shape[0]}", flush=True)

    # --- Trivial baseline: predict z1 by its TRAIN-set mean ---
    z1_mean = z1_train.mean(dim=0, keepdim=True)
    trivial_val_mse = nn.functional.mse_loss(
        z1_mean.expand_as(z1_val), z1_val
    ).item()

    # --- Linear baseline: closed-form OLS on train, evaluated on val ---
    print("\n[probe] Fitting closed-form linear baseline...", flush=True)
    a, k0, k1 = fit_linear_baseline(zp_train, z0_train, z1_train)
    linear_val_pred = linear_predict(zp_val, z0_val, a, k0, k1)
    linear_val_mse = nn.functional.mse_loss(linear_val_pred, z1_val).item()

    # --- Nonlinear MLP: trained with early stopping on val loss ---
    print("\n[probe] Training nonlinear MLP probe...", flush=True)
    _model, mlp_val_mse, _history = train_mlp(
        zp_train, z0_train, z1_train, zp_val, z0_val, z1_val,
        hidden=args.hidden, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience,
        device=args.device, seed=args.seed,
    )

    pct_vs_trivial = (1 - mlp_val_mse / trivial_val_mse) * 100 if trivial_val_mse > 0 else 0.0
    pct_vs_linear  = (1 - mlp_val_mse / linear_val_mse) * 100 if linear_val_mse > 0 else 0.0

    print("\n[summary] ── Held-out validation-set results (never trained on) ──")
    print(f"[summary] Trivial baseline (predict z1's train mean): {trivial_val_mse:.6f}")
    print(f"[summary] Linear baseline (OLS on z0, z_prev):        {linear_val_mse:.6f}")
    print(f"[summary] Nonlinear MLP probe:                        {mlp_val_mse:.6f}")
    print(f"[summary] MLP reduction vs. trivial baseline:         {pct_vs_trivial:.2f}%")
    print(f"[summary] MLP reduction vs. linear baseline:          {pct_vs_linear:.2f}%")
    print("\n[summary] The LAST number is the one that matters: it isolates whether")
    print("[summary] a nonlinear model finds ANY structure a linear regression could")
    print("[summary] not. If it's ~0%, no simple function of (z0, z_prev) predicts z1 —")
    print("[summary] strong evidence the representation itself has no exploitable")
    print("[summary] dynamics signal at this lag, not just this training run's DiT.")


if __name__ == "__main__":
    main()
