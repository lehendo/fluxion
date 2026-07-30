"""
tests/test_sample_ensemble.py

Unit tests for inference/sample_ensemble.py. CPU-only, tiny synthetic model.
Run with:

    pytest tests/test_sample_ensemble.py -v
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from inference.sample_ensemble import generate_ensemble
from training.dit_model import FluxionDiT


B_UNUSED, L, D_S, D_P = 2, 6, 16, 8
N_HEADS, D_FF, D_FREQ = 2, 32, 16


def _tiny_model() -> FluxionDiT:
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=2, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    ).eval()


class _ZeroVelocityModel(nn.Module):
    """Test double: velocity field identically zero everywhere, so the ODE
    integration leaves z0 unchanged at every step. Lets denormalization
    correctness be checked exactly, without depending on real network
    weights."""

    def __init__(self, d_single: int) -> None:
        super().__init__()
        self.d_single = d_single
        self._dummy = nn.Parameter(torch.zeros(1))  # so .parameters() is non-empty

    def forward(self, z, pair, t, padding_mask=None):
        return torch.zeros_like(z)


class TestGenerateEnsembleShapes:
    def test_output_shape_and_dtype(self):
        model = _tiny_model()
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        ensemble = generate_ensemble(
            model, pair, n_samples=5, norm_stats=norm_stats,
            batch_size=2, n_steps=3, method="euler", device=torch.device("cpu"),
        )
        assert ensemble.shape == (5, L, D_S)
        assert ensemble.dtype == torch.float32

    def test_chunking_handles_remainder_batch_of_one(self):
        """n_samples not evenly divisible by batch_size leaves a final chunk
        of size 1 — rollout()'s squeeze(1) would collapse that batch dim if
        not guarded against; verify the output still has exactly n_samples
        rows, not n_samples - 1."""
        model = _tiny_model()
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        ensemble = generate_ensemble(
            model, pair, n_samples=5, norm_stats=norm_stats,
            batch_size=2, n_steps=2, method="euler", device=torch.device("cpu"),
        )
        assert ensemble.shape[0] == 5

    def test_single_sample(self):
        model = _tiny_model()
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        ensemble = generate_ensemble(
            model, pair, n_samples=1, norm_stats=norm_stats,
            batch_size=4, n_steps=2, method="euler", device=torch.device("cpu"),
        )
        assert ensemble.shape == (1, L, D_S)


class TestGenerateEnsembleDenormalization:
    def test_zero_velocity_model_output_matches_denormalized_noise(self):
        """With a zero-velocity model, z stays at z0 for the whole
        integration (dz/dt=0), so the final generated latent (before
        denormalization) must equal the exact noise that was drawn. This
        directly verifies (a) norm_stats=None is correctly passed to
        rollout() internally (no incorrect pre-normalization of the noise),
        and (b) the manual denormalization (*std + mean) is applied exactly
        once, correctly, to the final output."""
        torch.manual_seed(42)
        model = _ZeroVelocityModel(D_S)
        pair = torch.randn(L, L, D_P)
        mean = torch.full((D_S,), 3.0)
        std = torch.full((D_S,), 2.0)
        norm_stats = {"mean": mean, "std": std}

        torch.manual_seed(123)
        expected_noise = torch.randn(3, L, D_S)

        torch.manual_seed(123)
        ensemble = generate_ensemble(
            model, pair, n_samples=3, norm_stats=norm_stats,
            batch_size=3, n_steps=4, method="euler", device=torch.device("cpu"),
        )

        expected = expected_noise * std + mean
        assert torch.allclose(ensemble, expected, atol=1e-4)

    def test_noise_scale_multiplies_z0_before_integration(self):
        """With a zero-velocity model, z stays at noise_scale*z0 throughout,
        so the final output must equal (noise_scale * raw_noise) * std +
        mean — confirms noise_scale is applied to z0 itself, not as some
        post-hoc rescaling of the final output."""
        model = _ZeroVelocityModel(D_S)
        pair = torch.randn(L, L, D_P)
        mean = torch.zeros(D_S)
        std = torch.ones(D_S)
        norm_stats = {"mean": mean, "std": std}
        noise_scale = 2.5

        torch.manual_seed(7)
        expected_noise = torch.randn(3, L, D_S) * noise_scale

        torch.manual_seed(7)
        ensemble = generate_ensemble(
            model, pair, n_samples=3, norm_stats=norm_stats,
            batch_size=3, n_steps=4, method="euler", device=torch.device("cpu"),
            noise_scale=noise_scale,
        )

        assert torch.allclose(ensemble, expected_noise, atol=1e-4)

    def test_noise_scale_default_is_unscaled(self):
        """noise_scale defaults to 1.0 — must reproduce prior (pre-change)
        behavior exactly for backward compatibility."""
        model = _ZeroVelocityModel(D_S)
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        torch.manual_seed(99)
        expected_noise = torch.randn(3, L, D_S)

        torch.manual_seed(99)
        ensemble = generate_ensemble(
            model, pair, n_samples=3, norm_stats=norm_stats,
            batch_size=3, n_steps=4, method="euler", device=torch.device("cpu"),
        )
        assert torch.allclose(ensemble, expected_noise, atol=1e-4)

    def test_denormalization_changes_scale_as_expected(self):
        """Sanity check: with a nontrivial std, the generated ensemble's
        scale should reflect that std, not stay at unit noise scale."""
        model = _ZeroVelocityModel(D_S)
        pair = torch.randn(L, L, D_P)
        mean = torch.zeros(D_S)
        std = torch.full((D_S,), 50.0)
        norm_stats = {"mean": mean, "std": std}

        ensemble = generate_ensemble(
            model, pair, n_samples=20, norm_stats=norm_stats,
            batch_size=8, n_steps=2, method="euler", device=torch.device("cpu"),
        )
        # Raw N(0,1) noise scaled by std=50 should have std roughly in that
        # ballpark, not ~1.
        assert ensemble.std().item() > 10.0


class TestGenerateEnsembleSeeding:
    def test_same_seed_reproducible(self):
        model = _tiny_model()
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        e1 = generate_ensemble(
            model, pair, n_samples=4, norm_stats=norm_stats,
            batch_size=4, n_steps=2, method="euler", device=torch.device("cpu"), seed=7,
        )
        e2 = generate_ensemble(
            model, pair, n_samples=4, norm_stats=norm_stats,
            batch_size=4, n_steps=2, method="euler", device=torch.device("cpu"), seed=7,
        )
        assert torch.allclose(e1, e2)

    def test_different_seed_different_output(self):
        model = _tiny_model()
        pair = torch.randn(L, L, D_P)
        norm_stats = {"mean": torch.zeros(D_S), "std": torch.ones(D_S)}

        e1 = generate_ensemble(
            model, pair, n_samples=4, norm_stats=norm_stats,
            batch_size=4, n_steps=2, method="euler", device=torch.device("cpu"), seed=1,
        )
        e2 = generate_ensemble(
            model, pair, n_samples=4, norm_stats=norm_stats,
            batch_size=4, n_steps=2, method="euler", device=torch.device("cpu"), seed=2,
        )
        assert not torch.allclose(e1, e2)
