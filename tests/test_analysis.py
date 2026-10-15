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


from analysis.stats import integrated_autocorr_time, block_len, n_eff_blocks, n_independent_windows
from analysis.mask import real_mask, verify_mask


class TestAutocorr:
    def test_exponential_decay_recovers_time_constant(self):
        T = 40.0
        taus = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
        tau_int = integrated_autocorr_time(taus, np.exp(-taus / T))
        assert tau_int == pytest.approx(T, rel=0.2)

    def test_floor_stops_noisy_tail_from_being_integrated(self):
        taus = np.array([1, 10, 100, 1000, 5000], dtype=float)
        rho = np.array([0.9, 0.5, 0.02, 0.4, 0.4])   # spurious late bump must be ignored
        clean = integrated_autocorr_time(taus, rho[:3])
        assert integrated_autocorr_time(taus, rho) == pytest.approx(clean)

    def test_block_len_covers_the_lag(self):
        assert block_len(1, 0.0) == 1 and block_len(100, 20.0) == 140

    def test_long_lag_has_few_independent_blocks(self):
        reps = [10001] * 3
        assert n_independent_windows(reps, 5000) == 6           # 2 per replica
        assert n_eff_blocks(reps, 5000, 300.0) < 6              # fewer once tau_int is counted
        assert n_eff_blocks(reps, 1, 300.0) > 40                # short lag: plenty


class TestMask:
    def test_mask_from_length(self):
        m = real_mask(8, 5)
        assert m.tolist() == [True] * 5 + [False] * 3

    def test_verify_accepts_correct_mask_and_keeps_rigid_residues(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(20, 8, 4, generator=g)
        z[:, 5:] = 3.0                      # padding is constant
        z[:, 2] = 1.0                       # a genuinely rigid REAL residue
        info = verify_mask(real_mask(8, 5), z)
        assert info["n_padded"] == 3 and info["n_real_with_zero_variance"] == 1
        assert real_mask(8, 5)[2]           # rigid residue stays in the mask

    def test_verify_rejects_a_wrong_mask(self):
        z = torch.randn(20, 8, 4)
        z[:, 6:] = 3.0                      # only 2 constant positions
        with pytest.raises(AssertionError):
            verify_mask(real_mask(8, 5), z)  # claims 3 padded; one of them varies


from analysis.probes import train_old_mlp, OldMLPProbe


class TestOldProbeFaithfulness:
    def test_vectorized_two_input_ols_matches_the_old_lstsq_loop(self):
        g = torch.Generator().manual_seed(3)
        z0 = torch.randn(3000, 7, generator=g); zp = torch.randn(3000, 7, generator=g)
        z1 = 0.6 * z0 - 0.2 * zp + 0.05 + 0.1 * torch.randn(3000, 7, generator=g)
        a, k0, k1 = fit_linear_diag(z0, z1, zp)
        for c in range(7):                                       # the old script's per-channel lstsq
            X = torch.stack([torch.ones(3000), z0[:, c], zp[:, c]], 1).double()
            sol = torch.linalg.lstsq(X, z1[:, c].double().unsqueeze(1)).solution.squeeze(1)
            assert torch.allclose(torch.stack([a[c], k0[c], k1[c]]).double(), sol, atol=1e-4)

    def test_old_mlp_architecture_matches_old_probe_parameter_count(self):
        # old MLPProbe(d=384, hidden=256): (768*256+256)+(256*256+256)+(256*384+384)
        n = sum(p.numel() for p in OldMLPProbe(384, 256).parameters())
        assert n == (768 * 256 + 256) + (256 * 256 + 256) + (256 * 384 + 384)

    def test_old_mlp_trains(self):
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(4000, 8, generator=g); zp = torch.randn(4000, 8, generator=g)
        z1 = 0.5 * z0 + 0.2 * zp
        m, best, ep = train_old_mlp(zp[:3000], z0[:3000], z1[:3000], zp[3000:], z0[3000:], z1[3000:],
                                    hidden=32, max_epochs=30, batch_size=64)
        assert best < 0.5 * z1.var().item()


class TestGradientSNREstimators:
    """scripts/phase1_gradsnr.estimates: unbiased ||E g||^2 and tr Cov(g) from running sums."""

    def test_recovers_known_signal_and_noise(self):
        from scripts.phase1_gradsnr import estimates
        g = torch.Generator().manual_seed(0)
        D, M = 200, 4000
        mu = torch.zeros(D); mu[:5] = 0.3                       # ||mu||^2 = 0.45
        sigma = 1.0                                              # tr Cov = D * sigma^2 = 200
        gs = mu + sigma * torch.randn(M, D, generator=g)
        mu2, tr = estimates(gs.sum(0), float((gs.double() ** 2).sum()), M)
        assert abs(mu2 - 0.45) < 0.06 and abs(tr - 200) < 4
        kstar = tr / mu2
        assert 300 < kstar < 600                                 # true 200/0.45 = 444

    def test_pure_noise_has_mean_square_norm_near_zero_not_biased_upward(self):
        from scripts.phase1_gradsnr import estimates
        g = torch.Generator().manual_seed(1)
        M, D = 3000, 100
        gs = torch.randn(M, D, generator=g)
        mu2, _ = estimates(gs.sum(0), float((gs.double() ** 2).sum()), M)
        # the naive plug-in ||mean||^2 would be ~ D/M = 0.033; the unbiased estimator is centered on 0
        assert abs(mu2) < 0.03


class TestClosedForm:
    def test_attenuation_limits_and_bounds(self):
        from analysis.theory import attenuation_R
        assert abs(attenuation_R(1.0) - 1 / 3) < 1e-6
        assert 0.42 < attenuation_R(0.0) < 0.43
        for rho in np.linspace(0, 0.999, 25):
            assert 1 / 3 - 1e-6 <= attenuation_R(rho) <= 0.4293      # 1/R in [2.33, 3.0] for any rho

    def test_curve_vanishes_at_midpoint_and_is_symmetric(self):
        from analysis.theory import f_closed_form
        t = np.array([0.1, 0.3])
        assert abs(f_closed_form(0.5, 0.6)) < 1e-12
        assert np.allclose(f_closed_form(t, 0.6), f_closed_form(1 - t, 0.6))

    def test_matches_monte_carlo_gaussian_pairs(self):
        from analysis.theory import f_closed_form
        rng = np.random.default_rng(0)
        rho, n = 0.7, 400_000
        z0 = rng.standard_normal(n); z1 = rho * z0 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        for t in (0.0, 0.2, 0.35):
            zt, d = (1 - t) * z0 + t * z1, z1 - z0
            slope = (zt * d).mean() / (zt * zt).mean()
            f_mc = 1 - ((d - slope * zt) ** 2).mean() / (d ** 2).mean()
            assert abs(f_mc - f_closed_form(t, rho)) < 0.004


class TestKabschRMSD:
    def test_pairwise_rmsd_is_invariant_to_rigid_motion_and_matches_noise_level(self):
        from analysis.geometry import kabsch_pairwise_rmsd
        rng = np.random.default_rng(0)
        L = 60
        X0 = rng.standard_normal((L, 3)) * 5
        th = 0.7; Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        X1 = X0 @ Rz.T + np.array([3.0, -2.0, 7.0])                 # rigid motion: RMSD 0
        X2 = X0 + 0.5 * rng.standard_normal((L, 3))                  # noise: RMSD ~ 0.5*sqrt(3) minus fit shrinkage
        D = kabsch_pairwise_rmsd(np.stack([X0, X1, X2]))
        assert D[0, 1] < 1e-6
        assert 0.7 < D[0, 2] < 0.95
        assert np.allclose(D, D.T, atol=1e-6)


class TestKabschAlign:
    def test_alignment_recovers_a_rotated_translated_copy_and_never_reflects(self):
        from analysis.geometry import kabsch_align
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 3)) * 4
        th = 1.1; R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        moved = ref @ R.T + np.array([5.0, 1.0, -3.0])
        out = kabsch_align(moved[None], ref)[0]
        assert np.abs(out - ref).max() < 1e-6
        mirror = ref * np.array([1, 1, -1])                          # a reflection must NOT be undone by a proper rotation
        assert np.abs(kabsch_align(mirror[None], ref)[0] - ref).max() > 1e-2


class TestPCAScores:
    def test_scores_have_the_training_eigenvalue_variances_and_are_orthogonal(self):
        from analysis.geometry import pca_scores
        g = torch.Generator().manual_seed(0)
        n, D, r = 300, 500, 5
        W = torch.randn(r, D, generator=g); scale = torch.tensor([5., 4., 3., 2., 1.]).unsqueeze(1)
        X = torch.randn(n, r, generator=g) @ (W * scale) + 0.01 * torch.randn(n, D, generator=g)
        S = pca_scores(X, X, k=5)
        var = S.var(0, unbiased=True)
        ref = torch.linalg.svdvals(X - X.mean(0)) ** 2 / (n - 1)
        assert torch.allclose(var, ref[:5], rtol=1e-3)
        C = (S - S.mean(0)).T @ (S - S.mean(0)) / (n - 1)
        assert (C - torch.diag(torch.diag(C))).abs().max() < 1e-2 * var.max()

    def test_heldout_projection_uses_only_training_axes(self):
        from analysis.geometry import pca_scores
        g = torch.Generator().manual_seed(1)
        Xtr = torch.randn(200, 30, generator=g) @ torch.diag(torch.linspace(3, 0.1, 30))
        Xte = torch.randn(50, 30, generator=g) @ torch.diag(torch.linspace(3, 0.1, 30))
        S_all = pca_scores(Xtr, torch.cat([Xtr, Xte]), k=10)
        assert torch.allclose(S_all[:200], pca_scores(Xtr, Xtr, k=10), atol=1e-4)


class TestRidgeDiag:
    def test_c_zero_is_ols_and_slopes_shrink_monotonically(self):
        import torch
        from analysis.probes import fit_linear_diag, fit_ridge_diag
        torch.manual_seed(0)
        z0 = torch.randn(500, 6) * torch.tensor([3.0, 2.0, 1.0, 0.5, 0.2, 0.1]); z1 = 0.6 * z0 + 0.5 * torch.randn(500, 6)
        a0, k0 = fit_ridge_diag(z0, z1, 0.0)
        a_ols, k_ols, _ = fit_linear_diag(z0, z1)
        assert torch.allclose(k0, k_ols, atol=1e-4) and torch.allclose(a0, a_ols, atol=1e-4)
        norms = [fit_ridge_diag(z0, z1, c)[1].abs().sum().item() for c in (0.0, 0.1, 1.0, 10.0)]
        assert all(x > y for x, y in zip(norms, norms[1:]))

    def test_selection_prefers_regularization_when_data_are_scarce_and_noisy(self):
        import torch
        from analysis.probes import select_ridge_strength
        torch.manual_seed(1)
        d, n = 60, 40                                                   # many channels, few pairs: OLS slopes are mostly noise
        tr0 = torch.randn(n, d); tr1 = 0.3 * tr0 + 1.0 * torch.randn(n, d)
        va0 = torch.randn(400, d); va1 = 0.3 * va0 + 1.0 * torch.randn(400, d)
        assert select_ridge_strength(tr0, tr1, va0, va1) > 0

    def test_selection_prefers_ols_when_data_are_plentiful_and_clean(self):
        import torch
        from analysis.probes import select_ridge_strength
        torch.manual_seed(2)
        tr0 = torch.randn(20000, 4); tr1 = 0.8 * tr0 + 0.05 * torch.randn(20000, 4)
        va0 = torch.randn(5000, 4); va1 = 0.8 * va0 + 0.05 * torch.randn(5000, 4)
        assert select_ridge_strength(tr0, tr1, va0, va1) <= 0.01


class TestSuperposedRmsd:
    def test_rotated_translated_copy_has_zero_rmsd(self):
        import numpy as np
        from analysis.geometry import superposed_rmsd
        rng = np.random.default_rng(0)
        a = rng.standard_normal((40, 3)) * 5
        th = 0.7; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        b = a @ R.T + np.array([3.0, -2.0, 10.0])
        assert superposed_rmsd(a, b) < 1e-9

    def test_known_displacement_is_recovered(self):
        import numpy as np
        from analysis.geometry import superposed_rmsd
        rng = np.random.default_rng(1)
        a = rng.standard_normal((200, 3)) * 5
        noise = rng.standard_normal((200, 3)) * 0.3
        b = a + noise
        r = superposed_rmsd(a, b)
        assert 0.0 < r <= np.sqrt((noise ** 2).sum(-1).mean()) + 1e-9        # superposition can only reduce the unaligned RMSD
        assert abs(r - np.sqrt((noise ** 2).sum(-1).mean())) < 0.03

    def test_mirror_image_is_not_aligned_by_a_reflection(self):
        import numpy as np
        from analysis.geometry import superposed_rmsd
        rng = np.random.default_rng(2)
        a = rng.standard_normal((50, 3)) * 5
        b = a * np.array([-1.0, 1.0, 1.0])
        assert superposed_rmsd(a, b) > 1.0


class TestPcaFitProject:
    def test_streamed_projection_equals_pca_scores(self):
        import torch
        from analysis.geometry import pca_fit, pca_project, pca_scores
        g = torch.Generator().manual_seed(0)
        Xtr, Xall = torch.randn(60, 40, generator=g), torch.randn(130, 40, generator=g)
        ref = pca_scores(Xtr, Xall, 7, chunk=17)
        mu, comp = pca_fit(Xtr, 7)
        streamed = torch.cat([pca_project(Xall[i:i + 31], mu, comp) for i in range(0, 130, 31)])
        assert torch.allclose(ref, streamed, atol=1e-5)

    def test_axes_are_orthonormal(self):
        import torch
        from analysis.geometry import pca_fit
        mu, comp = pca_fit(torch.randn(50, 30), 5)
        assert torch.allclose(comp.T @ comp, torch.eye(5), atol=1e-4)
