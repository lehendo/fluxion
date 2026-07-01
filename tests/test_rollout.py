"""
tests/test_rollout.py

Unit tests for inference/rollout.py.

CPU-only, no Chai-1, no checkpoint files needed.  The integrators are tested
against a linear velocity field whose exact solution is known analytically.

Run with:
    pytest tests/test_rollout.py -v
"""

from __future__ import annotations

import math

import torch
import pytest

from inference.rollout import euler, heun, rk4, rollout
from training.dit_model import FluxionDiT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, L, D_S, D_P = 1, 4, 8, 4
N_HEADS, D_FF, D_FREQ = 2, 16, 8


def _constant_velocity(v: torch.Tensor):
    """Returns a velocity function that always outputs `v`, ignoring z and t."""
    def func(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return v.expand_as(z)
    return func


def _linear_ode_solution(z0: torch.Tensor, v: torch.Tensor, t: float) -> torch.Tensor:
    """Exact solution of dz/dt = v: z(t) = z0 + v*t."""
    return z0 + v * t


def _tiny_model() -> FluxionDiT:
    return FluxionDiT(
        d_single=D_S, d_pair=D_P, d_model=D_S,
        n_layers=1, n_heads=N_HEADS, d_ff=D_FF, d_freq=D_FREQ,
    )


# ---------------------------------------------------------------------------
# Fixed-step integrators — shape and correctness
# ---------------------------------------------------------------------------

class TestEuler:
    def test_output_shape(self):
        z0 = torch.zeros(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = euler(_constant_velocity(torch.ones(B, L, D_S)), z0, t_eval)
        assert traj.shape == (11, B, L, D_S)

    def test_constant_velocity_exact(self):
        """Euler is exact for constant velocity (linear ODE)."""
        v  = torch.randn(B, L, D_S)
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 6)
        traj = euler(_constant_velocity(v), z0, t_eval)
        for i, t in enumerate(t_eval):
            expected = _linear_ode_solution(z0, v, t.item())
            assert torch.allclose(traj[i], expected, atol=1e-5), \
                f"Euler failed at t={t:.2f}"

    def test_no_nan(self):
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = euler(_constant_velocity(torch.randn(B, L, D_S)), z0, t_eval)
        assert not traj.isnan().any()

    def test_first_frame_equals_z0(self):
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = euler(_constant_velocity(torch.zeros(B, L, D_S)), z0, t_eval)
        assert torch.allclose(traj[0], z0)


class TestHeun:
    def test_output_shape(self):
        z0 = torch.zeros(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = heun(_constant_velocity(torch.ones(B, L, D_S)), z0, t_eval)
        assert traj.shape == (11, B, L, D_S)

    def test_constant_velocity_exact(self):
        """Heun is also exact for constant velocity."""
        v  = torch.randn(B, L, D_S)
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 6)
        traj = heun(_constant_velocity(v), z0, t_eval)
        for i, t in enumerate(t_eval):
            expected = _linear_ode_solution(z0, v, t.item())
            assert torch.allclose(traj[i], expected, atol=1e-5)

    def test_more_accurate_than_euler_nonlinear(self):
        """For a nonlinear ODE, Heun should be closer to Euler with 2× steps."""
        # dz/dt = sin(t)*ones → z(t) = z0 + (1 - cos(t))*ones
        z0 = torch.zeros(B, L, D_S)
        def sinusoidal(z, t):
            return torch.sin(t[0]) * torch.ones_like(z)

        n = 20
        t_eval = torch.linspace(0, 1, n + 1)
        exact  = 1.0 - math.cos(1.0)

        traj_euler = euler(sinusoidal, z0, t_eval)
        traj_heun  = heun(sinusoidal, z0, t_eval)

        err_euler = abs(traj_euler[-1].mean().item() - exact)
        err_heun  = abs(traj_heun[-1].mean().item() - exact)
        assert err_heun < err_euler, \
            f"Heun ({err_heun:.2e}) should beat Euler ({err_euler:.2e})"

    def test_no_nan(self):
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = heun(_constant_velocity(torch.randn(B, L, D_S)), z0, t_eval)
        assert not traj.isnan().any()


class TestRK4:
    def test_output_shape(self):
        z0 = torch.zeros(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = rk4(_constant_velocity(torch.ones(B, L, D_S)), z0, t_eval)
        assert traj.shape == (11, B, L, D_S)

    def test_constant_velocity_exact(self):
        v  = torch.randn(B, L, D_S)
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 6)
        traj = rk4(_constant_velocity(v), z0, t_eval)
        for i, t in enumerate(t_eval):
            expected = _linear_ode_solution(z0, v, t.item())
            assert torch.allclose(traj[i], expected, atol=1e-5)

    def test_more_accurate_than_heun_nonlinear(self):
        """RK4 should beat Heun on a nonlinear ODE."""
        z0 = torch.zeros(B, L, D_S)
        def sinusoidal(z, t):
            return torch.sin(t[0]) * torch.ones_like(z)

        n = 10  # coarse grid to exaggerate differences
        t_eval = torch.linspace(0, 1, n + 1)
        exact  = 1.0 - math.cos(1.0)

        traj_heun = heun(sinusoidal, z0, t_eval)
        traj_rk4  = rk4(sinusoidal, z0, t_eval)

        err_heun = abs(traj_heun[-1].mean().item() - exact)
        err_rk4  = abs(traj_rk4[-1].mean().item() - exact)
        assert err_rk4 < err_heun, \
            f"RK4 ({err_rk4:.2e}) should beat Heun ({err_heun:.2e})"

    def test_no_nan(self):
        z0 = torch.randn(B, L, D_S)
        t_eval = torch.linspace(0, 1, 11)
        traj = rk4(_constant_velocity(torch.randn(B, L, D_S)), z0, t_eval)
        assert not traj.isnan().any()


# ---------------------------------------------------------------------------
# rollout() — the main public API
# ---------------------------------------------------------------------------

class TestRollout:
    @pytest.fixture
    def model(self):
        return _tiny_model().eval()

    def test_output_shapes(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, t_eval = rollout(model, z0, pair, n_steps=10)
        assert traj.shape   == (11, L, D_S)
        assert t_eval.shape == (11,)

    def test_output_shapes_batched_input(self, model):
        """rollout should also accept a pre-batched [1, L, D_S] z0."""
        z0   = torch.randn(1, L, D_S)
        pair = torch.randn(1, L, L, D_P)
        traj, t_eval = rollout(model, z0, pair, n_steps=5)
        assert traj.shape == (6, L, D_S)

    def test_first_frame_equals_z0(self, model):
        """trajectory[0] must equal the input z0."""
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=5)
        assert torch.allclose(traj[0], z0, atol=1e-6)

    def test_t_eval_range(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        _, t_eval = rollout(model, z0, pair, n_steps=10, t_end=0.5)
        assert abs(t_eval[0].item()  - 0.0) < 1e-6
        assert abs(t_eval[-1].item() - 0.5) < 1e-6

    def test_no_nan_euler(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=20, method="euler")
        assert not traj.isnan().any()

    def test_no_nan_heun(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=10, method="heun")
        assert not traj.isnan().any()

    def test_no_nan_rk4(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=10, method="rk4")
        assert not traj.isnan().any()

    def test_output_is_float32_cpu(self, model):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, t_eval = rollout(model, z0, pair, n_steps=5)
        assert traj.dtype   == torch.float32
        assert t_eval.dtype == torch.float32
        assert traj.device.type   == "cpu"
        assert t_eval.device.type == "cpu"

    def test_zero_init_model_stays_at_z0(self, model):
        """FinalLayer is zero-init → velocity = 0 → trajectory is constant."""
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=10)
        # All frames should equal z0 (velocity is zero at init)
        for i in range(traj.shape[0]):
            assert torch.allclose(traj[i], z0, atol=1e-5), \
                f"Frame {i} diverged from z0 at zero-init"

    def test_nonzero_model_moves_away_from_z0(self, model):
        """After breaking zero-init, the trajectory should not stay at z0."""
        torch.manual_seed(123)
        with torch.no_grad():
            # Random (not constant) weights avoid zero-mean cancellation
            # that a fill_(c) constant matrix causes with LayerNorm outputs
            torch.nn.init.normal_(model.final_layer.linear.weight, std=1.0)
            model.final_layer.linear.bias.zero_()
            for block in model.blocks:
                torch.nn.init.normal_(block.adaLN_mod[-1].weight, std=1.0)

        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=20, method="euler")
        # Trajectory must vary over time (velocity ≠ 0)
        assert not torch.allclose(traj[-1], traj[0], atol=1e-3), \
            "Nonzero model should produce a trajectory that moves away from z0"


# ---------------------------------------------------------------------------
# rollout() — normalization round-trip
# ---------------------------------------------------------------------------

class TestRolloutNormalization:
    """
    rollout()'s public contract is raw-scale latents in, raw-scale latents
    out, even though it internally integrates in normalized space when
    norm_stats is given. These tests pin that round-trip.
    """

    @pytest.fixture
    def model(self):
        return _tiny_model().eval()

    @pytest.fixture
    def norm_stats(self):
        torch.manual_seed(0)
        return {"mean": torch.randn(D_S) * 5.0, "std": torch.rand(D_S) + 0.5}

    def test_first_frame_equals_raw_z0(self, model, norm_stats):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=5, norm_stats=norm_stats)
        assert torch.allclose(traj[0], z0, atol=1e-4)

    def test_zero_init_model_stays_at_raw_z0(self, model, norm_stats):
        """Zero-init model predicts zero velocity regardless of input scale,
        so with norm_stats the trajectory must still stay flat at the raw z0
        once mapped back out of normalized space."""
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=10, norm_stats=norm_stats)
        for i in range(traj.shape[0]):
            assert torch.allclose(traj[i], z0, atol=1e-4), \
                f"Frame {i} diverged from raw z0 under zero-init model with norm_stats"

    def test_no_nan_with_norm_stats(self, model, norm_stats):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, _ = rollout(model, z0, pair, n_steps=20, method="euler", norm_stats=norm_stats)
        assert not traj.isnan().any()

    def test_output_shapes_with_norm_stats(self, model, norm_stats):
        z0   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        traj, t_eval = rollout(model, z0, pair, n_steps=10, norm_stats=norm_stats)
        assert traj.shape   == (11, L, D_S)
        assert t_eval.shape == (11,)
