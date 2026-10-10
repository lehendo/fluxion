"""Tests for analysis/stats.py and analysis/probes.py (Phase 1 evidence code)."""
import numpy as np
import pytest
import torch

from analysis.stats import (
    entry_groups, pdb_entry, pct_improvement, lag_frames_to_label, time_blocks,
    block_bootstrap_improvement, entry_bootstrap_mean, ci_excludes_zero,
)
from analysis.probes import (
    fit_linear_diag, predict_linear_diag, fit_linear_full, predict_linear_full,
    train_mlp, row_sqerr,
)


class TestGrouping:
    def test_pdb_entry(self):
        assert pdb_entry("1bxy_A") == "1bxy"

    def test_siblings_share_a_group(self):
        g = entry_groups(["1bxy_A", "1bxy_B", "1b2s_E", "1b2s_F", "16pk_A"])
        assert g["1bxy"] == ["1bxy_A", "1bxy_B"]
        assert g["1b2s"] == ["1b2s_E", "1b2s_F"]
        assert g["16pk"] == ["16pk_A"]

    def test_pilot_set_has_17_entries(self):
        pilot = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,"
                 "1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
        assert len(pilot) == 19 and len(entry_groups(pilot)) == 17


class TestImprovement:
    def test_identity_is_zero_by_definition(self):
        assert pct_improvement(0.002079, 0.002079) == 0.0

    def test_paper_arithmetic(self):
        # Correction 1 of the brief: vs the MEAN baseline identity looks like 89.3%
        assert pct_improvement(0.002079, 0.01944) == pytest.approx(89.3, abs=0.05)
        # ...but vs identity the same probes are 21.2% / 18.2%
        assert pct_improvement(0.001638, 0.002079) == pytest.approx(21.2, abs=0.05)
        assert pct_improvement(0.001701, 0.002079) == pytest.approx(18.2, abs=0.05)

    def test_bad_reference(self):
        with pytest.raises(ValueError):
            pct_improvement(1.0, 0.0)

    def test_lag_labels(self):
        assert [lag_frames_to_label(x) for x in (1, 10, 100, 1000, 5000)] == [
            "10 ps", "100 ps", "1 ns", "10 ns", "50 ns"]


class TestCenteringInvariance:
    """The claim that makes probe MSE and flow-loss MSE comparable: subtracting any
    per-position constant cannot change identity MSE, because it cancels in z1-z0."""

    def test_identity_mse_invariant_to_centering(self):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(500, 384, generator=g)
        z1 = z0 + 0.05 * torch.randn(500, 384, generator=g)
        shift = torch.randn(1, 384, generator=g) * 3.0
        a = row_sqerr(z0, z1).mean()
        b = row_sqerr(z0 - shift, z1 - shift).mean()
        assert torch.allclose(a, b, atol=1e-6)

    def test_mean_baseline_is_not_invariant(self):
        # ...whereas the predict-the-mean baseline depends entirely on the centering.
        g = torch.Generator().manual_seed(1)
        z = torch.randn(500, 384, generator=g) + 5.0
        raw = row_sqerr(z.mean(0, keepdim=True).expand_as(z), z).mean()
        assert raw > 0.5  # variance, not 0; and not comparable to a displacement MSE


class TestLinear:
    def test_diag_recovers_known_coefficients(self):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(20000, 8, generator=g)
        k = torch.linspace(0.2, 0.9, 8)
        a = torch.linspace(-0.1, 0.1, 8)
        z1 = a + k * z0
        a_hat, k_hat, _ = fit_linear_diag(z0, z1)
        assert torch.allclose(k_hat, k, atol=1e-4) and torch.allclose(a_hat, a, atol=1e-4)

    def test_diag_with_prev_recovers_both_slopes(self):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(5000, 3, generator=g); zp = torch.randn(5000, 3, generator=g)
        z1 = 0.4 * z0 - 0.3 * zp + 0.1
        a, k0, k1 = fit_linear_diag(z0, z1, zp)
        assert torch.allclose(k0, torch.full((3,), 0.4), atol=1e-4)
        assert torch.allclose(k1, torch.full((3,), -0.3), atol=1e-4)

    def test_full_recovers_matrix(self):
        g = torch.Generator().manual_seed(0)
        A = torch.randn(6, 6, generator=g) * 0.3
        b = torch.randn(6, generator=g)
        z0 = torch.randn(20000, 6, generator=g)
        z1 = z0 @ A.T + b
        A_hat, b_hat = fit_linear_full(z0, z1)
        assert torch.allclose(A_hat, A, atol=1e-3) and torch.allclose(b_hat, b, atol=1e-3)
        assert torch.allclose(predict_linear_full(z0, A_hat, b_hat), z1, atol=1e-2)

    def test_identity_is_a_special_case_of_linear(self):
        # OLS can always match identity in-sample: it must never be WORSE on its own train set.
        g = torch.Generator().manual_seed(2)
        z0 = torch.randn(4000, 5, generator=g)
        z1 = z0 + 0.1 * torch.randn(4000, 5, generator=g)
        a, k, _ = fit_linear_diag(z0, z1)
        assert row_sqerr(predict_linear_diag(z0, a, k), z1).mean() <= row_sqerr(z0, z1).mean() + 1e-9


class TestBootstrap:
    def test_block_ci_contains_zero_under_null(self):
        rng = np.random.default_rng(0)
        ref = rng.exponential(1.0, 3000)
        mod = rng.exponential(1.0, 3000)           # same distribution -> true improvement 0
        blocks = np.arange(3000) // 50
        est, lo, hi = block_bootstrap_improvement(mod, ref, blocks, n_boot=1000)
        assert lo < 0 < hi and not ci_excludes_zero(lo, hi)

    def test_block_ci_excludes_zero_for_real_effect(self):
        rng = np.random.default_rng(0)
        ref = rng.exponential(1.0, 3000)
        mod = 0.7 * rng.exponential(1.0, 3000)     # true improvement 30%
        blocks = np.arange(3000) // 50
        est, lo, hi = block_bootstrap_improvement(mod, ref, blocks, n_boot=1000)
        assert ci_excludes_zero(lo, hi) and 25 < est < 35

    def test_entry_bootstrap_keeps_siblings_together(self):
        vals = {"1bxy_A": 5.0, "1bxy_B": 5.0, "a_A": 1.0, "b_A": 2.0, "c_A": 3.0}
        est, lo, hi = entry_bootstrap_mean(vals, n_boot=2000)
        assert est == pytest.approx(np.mean(list(vals.values())))
        assert lo <= est <= hi

    def test_time_blocks_do_not_mix_replicas(self):
        rep = np.array([0, 0, 1, 1]); fr = np.array([0, 10, 0, 10])
        ids = time_blocks(rep, fr, block_len=500)
        assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


class TestMLP:
    def test_mlp_learns_a_linear_map_and_early_stops_on_train_side_only(self):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(6000, 16, generator=g)
        z1 = 0.5 * z0
        m, best, ep = train_mlp(z0[:5000], z1[:5000], z0[5000:], z1[5000:],
                                hidden=64, max_epochs=15, batch_size=256)
        assert best < 0.01 * z1.var().item() + 1e-3 and ep >= 1
