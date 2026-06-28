"""
tests/test_flow_matching.py

Unit tests for training/flow_matching.py.

CPU-only.  Run with:

    pytest tests/test_flow_matching.py -v
"""

from __future__ import annotations

import torch
import pytest

from training.flow_matching import FlowMatcher
from training.dit_model import FluxionDiT


B, L, D_S, D_P = 2, 6, 16, 8
N_HEADS, D_FF   = 2, 32
D_FREQ          = 16


@pytest.fixture
def model():
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=1, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    )


@pytest.fixture
def matcher():
    return FlowMatcher(eps=1e-4)


class TestFlowMatcher:
    def test_loss_scalar(self, model, matcher):
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        loss = matcher.compute_loss(model, z0, z1, pair)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_loss_nonnegative(self, model, matcher):
        for _ in range(5):
            z0   = torch.randn(B, L, D_S)
            z1   = torch.randn(B, L, D_S)
            pair = torch.randn(B, L, L, D_P)
            assert matcher.compute_loss(model, z0, z1, pair).item() >= 0

    def test_loss_zero_when_identical_frames(self, model, matcher):
        """If z0 == z1, target velocity is zero. Zero-init model predicts zero → loss=0."""
        z    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        loss = matcher.compute_loss(model, z, z, pair)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_with_padding_mask(self, model, matcher):
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[0, -2:] = True   # pad last 2 tokens of first sample
        loss = matcher.compute_loss(model, z0, z1, pair, padding_mask=mask)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_padding_mask_excludes_padded_positions(self, model, matcher):
        """Loss should differ when padding mask changes which positions count."""
        torch.manual_seed(0)
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, -3:] = True
        loss_masked   = matcher.compute_loss(model, z0, z1, pair, padding_mask=mask)
        loss_unmasked = matcher.compute_loss(model, z0, z1, pair, padding_mask=None)
        # They won't be identical because the averaging denominator differs
        assert loss_masked.item() != loss_unmasked.item()

    def test_loss_is_differentiable(self, model, matcher):
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        loss = matcher.compute_loss(model, z0, z1, pair)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any()

    def test_all_padding_does_not_crash(self, model, matcher):
        """Edge case: every token is masked. Should not divide by zero."""
        z0   = torch.randn(B, L, D_S)
        z1   = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        mask = torch.ones(B, L, dtype=torch.bool)   # everything masked
        # Should not raise
        loss = matcher.compute_loss(model, z0, z1, pair, padding_mask=mask)
        assert not loss.isnan()
