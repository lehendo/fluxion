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


class TestTDistribution:
    """Phase 3a: the t sampler is the only thing that may differ between arms."""

    def test_default_is_uniform_and_unchanged(self):
        from training.flow_matching import FlowMatcher
        torch.manual_seed(0)
        t = FlowMatcher().sample_t(20000, "cpu")
        assert abs(t.mean().item() - 0.5) < 0.01 and abs(t.var().item() - 1 / 12) < 0.005
        assert t.min() >= 1e-5 and t.max() <= 1 - 1e-5

    def test_beta_puts_mass_at_both_endpoints(self):
        from training.flow_matching import FlowMatcher
        torch.manual_seed(0)
        t = FlowMatcher(t_dist="beta").sample_t(20000, "cpu")
        assert abs(t.mean().item() - 0.5) < 0.01                       # symmetric
        assert ((t < 0.1) | (t > 0.9)).float().mean() > 0.38            # arcsine: 0.41; uniform: 0.20
        assert ((t > 0.4) & (t < 0.6)).float().mean() < 0.15            # arcsine: 0.128; uniform: 0.20
        assert t.min() >= 1e-5 and t.max() <= 1 - 1e-5

    def test_beta_shape_parameters_respected(self):
        from training.flow_matching import FlowMatcher
        torch.manual_seed(0)
        heavy = FlowMatcher(t_dist="beta", beta_a=0.1, beta_b=0.1).sample_t(20000, "cpu")
        assert ((heavy < 0.1) | (heavy > 0.9)).float().mean() > 0.75      # arcsine-like limit: 0.81

    def test_bad_name_rejected(self):
        import pytest
        from training.flow_matching import FlowMatcher
        with pytest.raises(ValueError):
            FlowMatcher(t_dist="logit")


class TestClippedT:
    def test_clamp_gives_point_masses_at_the_bounds(self):
        from training.flow_matching import FlowMatcher
        torch.manual_seed(0)
        t = FlowMatcher(t_dist="beta", beta_a=0.25, beta_b=0.25, t_clip=(0.01, 0.99)).sample_t(100000, "cpu")
        assert t.min() >= 0.01 and t.max() <= 0.99
        assert abs((t == 0.01).float().mean().item() - 0.17) < 0.01 and abs((t == 0.99).float().mean().item() - 0.17) < 0.01   # Beta(.25,.25) CDF(0.01) = 0.17
        assert abs(t.mean().item() - 0.5) < 0.01

    def test_clamped_uniform(self):
        from training.flow_matching import FlowMatcher
        torch.manual_seed(0)
        t = FlowMatcher(t_clip=(0.01, 0.99)).sample_t(100000, "cpu")
        assert abs((t == 0.01).float().mean().item() - 0.01) < 0.003 and t.min() >= 0.01 and t.max() <= 0.99

    def test_bad_range_rejected(self):
        import pytest
        from training.flow_matching import FlowMatcher
        with pytest.raises(ValueError):
            FlowMatcher(t_clip=(0.6, 0.4))


class TestReturnStats:
    def test_v0_loss_is_zero_predictor_loss_and_default_unchanged(self):
        from training.flow_matching import FlowMatcher
        class Zero(torch.nn.Module):
            def forward(self, z, pair, t, mask):
                return torch.zeros_like(z)
        torch.manual_seed(0)
        z0, z1 = torch.randn(4, 6, 3), torch.randn(4, 6, 3)
        mask = torch.zeros(4, 6, dtype=torch.bool); mask[:, 4:] = True
        fm = FlowMatcher()
        loss, v0 = fm.compute_loss(Zero(), z0, z1, None, mask, return_stats=True)
        assert torch.allclose(loss, v0)                                     # a zero predictor attains exactly its own v=0 loss
        expect = (((z1 - z0) ** 2)[:, :4]).mean(dim=(1, 2)).mean()
        assert torch.allclose(v0, expect)
        assert fm.compute_loss(Zero(), z0, z1, None, mask).dim() == 0       # default return unchanged
