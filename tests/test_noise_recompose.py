import numpy as np
from fluxion.analysis.noise_recompose import corrected_gain, flags, recompose


def _simulate(rho, nu, n=400_000, seed=0):
    """AR(1) signal with unit variance and lag-1 autocorrelation rho, plus independent noise so that noise / (signal + noise) = nu; returns the observed shrinkage gain and the noise-free truth (1 - rho)/2."""
    rng = np.random.default_rng(seed)
    s = np.empty(n); s[0] = rng.standard_normal()
    e = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + e[t]
    sig_n2 = nu / (1 - nu) * 1.0                                   # noise variance relative to a unit signal variance
    z = s + rng.standard_normal(n) * np.sqrt(sig_n2)
    z0, z1 = z[:-1], z[1:]
    alpha = (z0 * z1).mean() / (z0 * z0).mean()                    # the least-squares one-parameter shrinkage coefficient
    mse_shrink = ((z1 - alpha * z0) ** 2).mean(); mse_id = ((z1 - z0) ** 2).mean()
    return 1 - mse_shrink / mse_id, (1 - rho) / 2, z.var(), sig_n2


def test_correction_recovers_the_noise_free_gain_in_simulation():
    for rho, nu in ((0.8, 0.06), (0.9, 0.15), (0.5, 0.3)):
        g_obs, truth, var, sig_n2 = _simulate(rho, nu)
        assert g_obs > truth                                              # noise inflates the apparent contraction
        assert abs(float(corrected_gain(g_obs, sig_n2 / var)) - truth) < 5e-3
        assert abs(g_obs - ((1 - rho) / 2 + rho * nu / 2)) < 5e-3         # the decomposition (1 - rho)/2 + rho nu / 2


def test_flags_and_vectorization():
    g = np.array([0.15, 0.15, -0.1]); nu = np.array([0.05, 0.6, 0.05])
    assert flags(g, nu).tolist() == [False, True, True]                  # nu >= 0.5; rho > 1 for a negative gain
    assert corrected_gain(g, nu).shape == (3,)


def test_recompose_point_matches_direct_mean_and_interval_widens_with_measurement_noise():
    P = ["1aaa_A", "1bbb_A", "1ccc_A", "1ddd_A", "1eee_A", "1fff_A"]
    var = {p: 0.05 + 0.01 * i for i, p in enumerate(P)}; g = {p: 0.13 + 0.005 * i for i, p in enumerate(P)}
    s2 = {p: 0.0006 for p in P}
    tight = {p: np.full(50, 0.0006) for p in P}; loose = {p: 0.0006 * (1 + 0.5 * np.random.default_rng(i).standard_normal(500)) for i, p in enumerate(P)}
    e1, lo1, hi1 = recompose(P, var, g, s2, tight, n_boot=3000); e2, lo2, hi2 = recompose(P, var, g, s2, loose, n_boot=3000)
    direct = 100 * np.mean([corrected_gain(g[p], s2[p] / var[p]) for p in P])
    assert abs(e1 - direct) < 1e-9 and abs(e2 - direct) < 1e-9
    assert (hi2 - lo2) > (hi1 - lo1) and lo1 < e1 < hi1


def test_chains_of_one_entry_are_resampled_together():
    P = ["1aaa_A", "1aaa_B", "1bbb_A"]; var = {p: 0.05 for p in P}; g = {"1aaa_A": 0.10, "1aaa_B": 0.10, "1bbb_A": 0.20}
    s2 = {p: 0.0 for p in P}; d = {p: np.zeros(4) for p in P}
    e, lo, hi = recompose(P, var, g, s2, d, n_boot=2000)
    # with zero noise the corrected gain is the observed gain; entry resampling gives means from {0.10, 0.15, 0.20}-like combinations, never a 1:2 mix of 'aaa' chains with a lone 'bbb'
    assert abs(e - 100 * np.mean([0.10, 0.10, 0.20])) < 1e-9 and 9.9 < lo <= hi < 20.1


def test_return_diff_is_observed_minus_corrected_with_its_own_interval():
    P = ["1aaa_A", "1bbb_A", "1ccc_A", "1ddd_A"]; var = {p: 0.05 for p in P}; g = {p: 0.14 for p in P}
    s2 = {p: 0.0005 for p in P}; d = {p: np.full(20, 0.0005) for p in P}
    (e, lo, hi), (de, dlo, dhi) = recompose(P, var, g, s2, d, n_boot=500, return_diff=True)
    assert abs(de - (14.0 - e)) < 1e-9 and de > 0 and dlo - 1e-9 <= de <= dhi + 1e-9
