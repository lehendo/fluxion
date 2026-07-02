"""
tests/test_evaluate_checkpoint.py

Unit tests for scripts/evaluate_checkpoint.py.

CPU-only, synthetic data — no cluster checkpoint or latent store needed.
Run with:

    pytest tests/test_evaluate_checkpoint.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_checkpoint import (  # noqa: E402
    summarize_wakeup,
    collect_batches,
    compute_model_loss,
)
from training.dit_model import FluxionDiT  # noqa: E402
from training.flow_matching import FlowMatcher  # noqa: E402


B, L, D_S, D_P = 2, 6, 16, 8
N_HEADS, D_FF, D_FREQ = 2, 32, 16


def _tiny_model() -> FluxionDiT:
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=3, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    )


def _synthetic_batch():
    z0   = torch.randn(B, L, D_S)
    z1   = torch.randn(B, L, D_S)
    pair = torch.randn(B, L, L, D_P)
    fidx = torch.zeros(B, dtype=torch.int64)
    mask = torch.zeros(B, L, dtype=torch.bool)
    return (z0, z1), pair, fidx, mask


class TestSummarizeWakeup:
    def test_all_zero_at_init(self):
        """Fresh FluxionDiT is zero-init everywhere this function inspects."""
        model = _tiny_model()
        stats = summarize_wakeup(model)
        # final_layer.linear + final_layer.adaLN_mod + one entry per block
        assert len(stats) == 2 + len(model.blocks)
        for name, val in stats.items():
            assert val == pytest.approx(0.0, abs=1e-12), f"{name} should be exactly 0 at init"

    def test_nonzero_after_perturbation(self):
        model = _tiny_model()
        with torch.no_grad():
            model.final_layer.linear.weight.fill_(0.5)
            model.final_layer.adaLN_mod[-1].weight.fill_(0.5)
            for block in model.blocks:
                block.adaLN_mod[-1].weight.fill_(0.5)
        stats = summarize_wakeup(model)
        for name, val in stats.items():
            assert val > 0.0, f"{name} should be nonzero after perturbation"

    def test_keys_cover_every_block(self):
        model = _tiny_model()
        stats = summarize_wakeup(model)
        for i in range(len(model.blocks)):
            assert f"block[{i}].adaLN_mod.weight" in stats


class TestCollectBatches:
    def test_collects_exact_count(self):
        batches = [_synthetic_batch() for _ in range(5)]
        collected = collect_batches(batches, n_batches=5)
        assert len(collected) == 5

    def test_wraps_around_when_n_batches_exceeds_source(self):
        batches = [_synthetic_batch() for _ in range(2)]
        collected = collect_batches(batches, n_batches=5)
        assert len(collected) == 5


class TestComputeModelLoss:
    def test_matches_direct_flowmatcher_call(self):
        """compute_model_loss must average FlowMatcher.compute_loss exactly,
        weighted by batch size, over a fixed list of batches."""
        torch.manual_seed(0)
        model = _tiny_model().eval()
        matcher = FlowMatcher()
        batches = [_synthetic_batch() for _ in range(3)]

        avg_loss, n = compute_model_loss(model, matcher, batches, torch.device("cpu"))

        total, total_n = 0.0, 0
        with torch.no_grad():
            for (z0, z1), pair, _fidx, mask in batches:
                loss = matcher.compute_loss(model, z0, z1, pair, mask)
                total += loss.item() * z0.shape[0]
                total_n += z0.shape[0]
        expected = total / total_n

        assert avg_loss == pytest.approx(expected, rel=1e-5)
        assert n == total_n == B * 3

    def test_zero_init_model_matches_trivial_baseline(self):
        """At zero-init, the model always predicts v=0, so its loss must
        exactly equal the trivial (v_pred=0) baseline on the same batches."""
        from compute_trivial_baseline import compute_trivial_baseline_mse

        model = _tiny_model().eval()
        matcher = FlowMatcher()
        batches = [_synthetic_batch() for _ in range(4)]

        model_loss, _ = compute_model_loss(model, matcher, batches, torch.device("cpu"))
        baseline, _ = compute_trivial_baseline_mse(batches, len(batches))

        assert model_loss == pytest.approx(baseline, rel=1e-4)
