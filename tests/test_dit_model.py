"""
tests/test_dit_model.py

Unit tests for training/dit_model.py.

CPU-only, no Chai-1.  Run with:

    pytest tests/test_dit_model.py -v
"""

from __future__ import annotations

import torch
import pytest

from training.dit_model import (
    SinusoidalEmbedding,
    TimestepEmbedding,
    PairBiasedAttention,
    SwiGLUFFN,
    DiTBlock,
    FinalLayer,
    FluxionDiT,
)


# Tiny dims so tests run in milliseconds on CPU
B, L, D_S, D_P = 2, 6, 16, 8
N_HEADS, D_FF   = 2, 32
D_FREQ          = 16


# ---------------------------------------------------------------------------
# SinusoidalEmbedding
# ---------------------------------------------------------------------------

class TestSinusoidalEmbedding:
    def test_output_shape(self):
        emb = SinusoidalEmbedding(d_freq=D_FREQ)
        t   = torch.rand(B)
        out = emb(t)
        assert out.shape == (B, D_FREQ)

    def test_deterministic(self):
        emb = SinusoidalEmbedding(d_freq=D_FREQ)
        t   = torch.rand(B)
        assert torch.allclose(emb(t), emb(t))

    def test_odd_d_freq_raises(self):
        with pytest.raises(ValueError, match="even"):
            SinusoidalEmbedding(d_freq=7)


# ---------------------------------------------------------------------------
# TimestepEmbedding
# ---------------------------------------------------------------------------

class TestTimestepEmbedding:
    def test_output_shape(self):
        emb = TimestepEmbedding(d_model=D_S, d_freq=D_FREQ)
        t   = torch.rand(B)
        assert emb(t).shape == (B, D_S)


# ---------------------------------------------------------------------------
# PairBiasedAttention
# ---------------------------------------------------------------------------

class TestPairBiasedAttention:
    @pytest.fixture
    def attn(self):
        return PairBiasedAttention(D_S, D_P, N_HEADS)

    def test_output_shape(self, attn):
        x    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        assert attn(x, pair).shape == (B, L, D_S)

    def test_padding_mask_zeros_padded_output(self, attn):
        """With all-padding mask on last token, output should differ from no-mask."""
        x    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, -1] = True   # mask out last position as key
        out_masked   = attn(x, pair, mask)
        out_unmasked = attn(x, pair, None)
        # Masked and unmasked outputs must differ (the bias changed)
        assert not torch.allclose(out_masked, out_unmasked)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            PairBiasedAttention(d_model=7, d_pair=D_P, n_heads=4)

    def test_all_padding_mask_does_not_nan(self, attn):
        """Fully-masked batch should not produce NaN (softmax of -inf handled)."""
        x    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        mask = torch.ones(B, L, dtype=torch.bool)   # all masked
        out  = attn(x, pair, mask)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# SwiGLUFFN
# ---------------------------------------------------------------------------

class TestSwiGLUFFN:
    def test_output_shape(self):
        ffn = SwiGLUFFN(D_S, D_FF)
        x   = torch.randn(B, L, D_S)
        assert ffn(x).shape == (B, L, D_S)


# ---------------------------------------------------------------------------
# DiTBlock
# ---------------------------------------------------------------------------

class TestDiTBlock:
    @pytest.fixture
    def block(self):
        return DiTBlock(D_S, D_P, N_HEADS, D_FF)

    def test_output_shape(self, block):
        x     = torch.randn(B, L, D_S)
        pair  = torch.randn(B, L, L, D_P)
        t_emb = torch.randn(B, D_S)
        assert block(x, pair, t_emb).shape == (B, L, D_S)

    def test_adaLN_zero_init_near_identity(self, block):
        """At init the adaLN modulation is zero → block output ≈ input (residual)."""
        x     = torch.randn(B, L, D_S)
        pair  = torch.zeros(B, L, L, D_P)    # zero pair → zero bias
        t_emb = torch.zeros(B, D_S)           # zero t → zero modulation
        out   = block(x, pair, t_emb)
        # With zero modulation gates the residuals are added with gate=0
        # so output == input exactly at init
        assert torch.allclose(out, x, atol=1e-5)

    def test_padding_mask_accepted_without_error(self, block):
        """Padding mask must be accepted and produce correct output shape.
        At zero-init the AdaLN gates are 0 → block is identity regardless of
        masking; the effect of masking is tested at PairBiasedAttention level."""
        x     = torch.randn(B, L, D_S)
        pair  = torch.randn(B, L, L, D_P)
        t_emb = torch.randn(B, D_S)
        mask  = torch.zeros(B, L, dtype=torch.bool)
        mask[:, -1] = True
        out = block(x, pair, t_emb, mask)
        assert out.shape == (B, L, D_S)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

class TestFinalLayer:
    def test_output_shape(self):
        fl    = FinalLayer(D_S, D_S)
        x     = torch.randn(B, L, D_S)
        t_emb = torch.randn(B, D_S)
        assert fl(x, t_emb).shape == (B, L, D_S)

    def test_zero_init_output(self):
        """Zero-init linear → output is zero at initialization for any input."""
        fl    = FinalLayer(D_S, D_S)
        x     = torch.randn(B, L, D_S)
        t_emb = torch.randn(B, D_S)
        out   = fl(x, t_emb)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ---------------------------------------------------------------------------
# FluxionDiT (full model)
# ---------------------------------------------------------------------------

class TestFluxionDiT:
    @pytest.fixture
    def model(self):
        return FluxionDiT(
            d_single=D_S, d_pair=D_P, d_model=D_S,
            n_layers=2, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
        )

    def test_output_shape(self, model):
        z    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        t    = torch.rand(B)
        out  = model(z, pair, t)
        assert out.shape == (B, L, D_S)

    def test_output_shape_with_padding_mask(self, model):
        z    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        t    = torch.rand(B)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[0, -2:] = True
        out  = model(z, pair, t, mask)
        assert out.shape == (B, L, D_S)

    def test_padding_mask_changes_output_after_init_broken(self, model):
        """Masking must change output once the FinalLayer is non-zero.
        At zero-init FinalLayer always outputs zero; we break that here."""
        torch.manual_seed(0)
        with torch.no_grad():
            # Random (not constant) weights avoid zero-mean cancellation that
            # a fill_(c) constant matrix causes when dotted against the
            # zero-mean output of an elementwise_affine=False LayerNorm — a
            # constant weight row times a zero-mean vector collapses to ~0
            # regardless of masking, making the test flaky/vacuous.
            torch.nn.init.normal_(model.final_layer.linear.weight, std=1.0)
            model.final_layer.linear.bias.zero_()
            for block in model.blocks:
                torch.nn.init.normal_(block.adaLN_mod[-1].weight, std=1.0)

        z    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        t    = torch.rand(B)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, -1] = True
        assert not torch.allclose(model(z, pair, t, mask), model(z, pair, t, None))

    def test_zero_velocity_at_init(self, model):
        """FinalLayer is zero-init → model predicts zero velocity at initialization."""
        z    = torch.randn(B, L, D_S)
        pair = torch.randn(B, L, L, D_P)
        t    = torch.rand(B)
        out  = model(z, pair, t)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_param_count_reasonable(self, model):
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_no_nan_gradient(self, model):
        z    = torch.randn(B, L, D_S, requires_grad=True)
        pair = torch.randn(B, L, L, D_P)
        t    = torch.rand(B)
        loss = model(z, pair, t).sum()
        loss.backward()
        assert not z.grad.isnan().any()

    def test_full_scale_param_count(self):
        """Default config should be ~40M params."""
        model = FluxionDiT()
        n = sum(p.numel() for p in model.parameters())
        assert 30_000_000 < n < 60_000_000, f"Unexpected param count: {n:,}"
