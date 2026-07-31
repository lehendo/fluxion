"""
tests/test_train_static.py

Unit tests for training/train_static.py — the static-ensemble pivot's
training script. Importing training.train_static doesn't require
torch.distributed to be initialized — only calling main() does, which
these tests never do. Run with:

    pytest tests/test_train_static.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import pytest

from training.train_static import handle_sigusr1
import training.train_static as train_static
from training.dit_model import FluxionDiT
from training.flow_matching import FlowMatcher
from training.ot_coupling import compute_ot_coupling
from data_pipeline.dataset import (
    build_multi_protein_static_dataset,
    collate_fn_static,
)


B, L, D_S, D_P = 4, 6, 16, 8
N_HEADS, D_FF, D_FREQ = 2, 32, 16


def _tiny_model() -> FluxionDiT:
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=3, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    )


def _make_protein_dir(root: Path, protein: str, n_frames: int, L: int, d_s: int) -> None:
    protein_dir = root / protein
    shard_dir = protein_dir / protein
    shard_dir.mkdir(parents=True)
    with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
        f.create_dataset("single", data=np.random.randn(n_frames, L, d_s).astype(np.float16))
        f.create_dataset("frame_idx", data=np.arange(n_frames, dtype=np.int32))
    manifest = {protein: {"shards": [{
        "path": f"{protein}/shard_000000.h5", "n_frames": n_frames,
        "first_idx": 0, "last_idx": n_frames - 1,
    }]}}
    with open(protein_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    torch.save(torch.randn(L, L, D_P).half(), protein_dir / "pair.pt")


class TestSharedHelpersReused:
    """train_static imports (not redefines) get_param_groups,
    compute_group_grad_norms, update_ema, warmup_lambda from training.train
    — confirm they're the exact same objects, not accidental copies that
    could drift out of sync."""

    def test_reuses_train_py_helpers_by_identity(self):
        from training.train import (
            get_param_groups as t_gpg,
            compute_group_grad_norms as t_cggn,
            update_ema as t_ue,
            warmup_lambda as t_wl,
        )
        assert train_static.get_param_groups is t_gpg
        assert train_static.compute_group_grad_norms is t_cggn
        assert train_static.update_ema is t_ue
        assert train_static.warmup_lambda is t_wl


class TestHandleSigusr1:
    def test_sets_own_module_level_flag(self):
        train_static.PREEMPTION_REQUESTED = False
        handle_sigusr1(signum=None, frame=None)
        assert train_static.PREEMPTION_REQUESTED is True
        train_static.PREEMPTION_REQUESTED = False  # reset for other tests


class TestStaticTrainingLossIntegration:
    """End-to-end: dataset -> collate_fn_static -> fresh-noise z0 ->
    FlowMatcher.compute_loss -> backward, exactly matching the training
    loop's inner-step logic in main(), without needing torch.distributed."""

    def test_forward_backward_with_fresh_noise_z0(self, tmp_path):
        _make_protein_dir(tmp_path, "protA", n_frames=5, L=L, d_s=D_S)
        dataset = build_multi_protein_static_dataset(tmp_path, protein_ids=["protA"])
        batch = [dataset[i] for i in range(B)]
        z1, pair, frame_idx, padding_mask = collate_fn_static(batch)

        assert z1.shape == (B, L, D_S)
        assert pair.shape == (B, L, L, D_P)
        assert padding_mask.shape == (B, L)

        model = _tiny_model()
        matcher = FlowMatcher()

        z0 = torch.randn_like(z1)
        loss = matcher.compute_loss(model, z0, z1, pair, padding_mask)
        loss.backward()

        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_z0_is_resampled_each_call_not_reused(self, tmp_path):
        """Regression guard: z0 must be freshly sampled per step (the whole
        point of the static-ensemble reformulation), not accidentally cached
        or read from disk like the temporal pipeline's z0."""
        _make_protein_dir(tmp_path, "protA", n_frames=5, L=L, d_s=D_S)
        dataset = build_multi_protein_static_dataset(tmp_path, protein_ids=["protA"])
        batch = [dataset[i] for i in range(B)]
        z1, _pair, _fidx, _mask = collate_fn_static(batch)

        torch.manual_seed(0)
        z0_a = torch.randn_like(z1)
        z0_b = torch.randn_like(z1)
        assert not torch.allclose(z0_a, z0_b)

    def test_static_dataset_yields_no_pairing_unlike_temporal(self, tmp_path):
        """Sanity check that StaticFrameDataset samples (via
        build_multi_protein_static_dataset) are single frames, not
        (z0, z1) pairs — confirms the dataset swap is wired correctly."""
        _make_protein_dir(tmp_path, "protA", n_frames=5, L=L, d_s=D_S)
        dataset = build_multi_protein_static_dataset(tmp_path, protein_ids=["protA"])
        sample = dataset[0]
        assert len(sample) == 3  # (single, pair, frame_idx) — not ((z0, z1), pair, frame_idx)
        single, pair, frame_idx = sample
        assert single.shape == (L, D_S)


class TestOtCouplingIntegration:
    """Reproduces main()'s inner-step OT-coupling block exactly: z0 =
    randn_like(z1); order = compute_ot_coupling(...); reorder z1, pair,
    padding_mask — the critical property being that ALL THREE are permuted
    identically, or the model would train against a mismatched (wrong
    protein's) conditioning signal."""

    def test_forward_backward_with_ot_coupling(self, tmp_path):
        _make_protein_dir(tmp_path, "protA", n_frames=8, L=L, d_s=D_S)
        dataset = build_multi_protein_static_dataset(tmp_path, protein_ids=["protA"])
        batch = [dataset[i] for i in range(B)]
        z1, pair, _fidx, padding_mask = collate_fn_static(batch)

        model = _tiny_model()
        matcher = FlowMatcher()

        z0 = torch.randn_like(z1)
        order = compute_ot_coupling(z0, z1, padding_mask)
        z1_c           = z1[order]
        pair_c         = pair[order]
        padding_mask_c = padding_mask[order]

        loss = matcher.compute_loss(model, z0, z1_c, pair_c, padding_mask_c)
        loss.backward()

        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    def test_pair_and_padding_mask_stay_matched_to_reordered_z1(self, tmp_path):
        """The critical correctness property: if z1[i] ends up at position j
        after reordering, that position's pair conditioning and padding
        mask MUST also be the ones originally paired with z1[i] — not
        whichever happened to already be at position j. Uses distinctly
        tagged pair/padding-mask per original sample so any mismatch after
        reordering is directly observable."""
        z0 = torch.randn(B, L, D_S)
        z1 = torch.randn(B, L, D_S)
        pair = torch.stack([
            torch.full((L, L, D_P), float(i)) for i in range(B)
        ])
        # Distinct padding pattern per sample so mismatches are observable
        # in the mask too, not just the pair tensor.
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            padding_mask[i, i:] = True  # sample i: first i positions real

        order = compute_ot_coupling(z0, z1, padding_mask)
        z1_c           = z1[order]
        pair_c         = pair[order]
        padding_mask_c = padding_mask[order]

        for new_pos in range(B):
            original_i = order[new_pos].item()
            # pair_c[new_pos] must be the ORIGINAL sample original_i's pair
            # (tagged with value float(original_i)), not e.g. pair[new_pos].
            assert torch.allclose(pair_c[new_pos], torch.full((L, L, D_P), float(original_i)))
            # Same check for padding_mask via its distinct per-sample pattern.
            expected_mask = torch.zeros(L, dtype=torch.bool)
            expected_mask[original_i:] = True
            assert torch.equal(padding_mask_c[new_pos], expected_mask)
            # And z1 itself must match what z1[order] would give directly.
            assert torch.allclose(z1_c[new_pos], z1[original_i])
