import numpy as np
import torch
from analysis.b1_controls import BANDS, band_contrib, band_unweighted, pct_vs_identity, per_dim_mse


def _ar1(n, k, rho, scales, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, k)
    x[0] = torch.randn(k, generator=g)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + (1 - rho ** 2) ** 0.5 * torch.randn(k, generator=g)
    return x * scales


def _fold(whiten=False, seed=0, k=12):
    scales = torch.logspace(1, -2, k)                                # strongly variance-ordered dimensions
    tr = _ar1(6000, k, 0.7, scales, seed); te = _ar1(3000, k, 0.7, scales, seed + 1)
    std = tr.std(0) if whiten else None
    return per_dim_mse((tr[:-1], tr[1:]), (te[:-1], te[1:]), std)


def test_bands_sum_to_pct_and_halves_sum_to_pct():
    f = [_fold(seed=s) for s in (0, 2)]
    for pred in ("linear_diag", "shrink", "noint"):
        bands, h1, h2 = band_contrib(f, pred, bands=[(0, 3), (3, 12)])
        assert abs(sum(bands) - pct_vs_identity(f, pred)) < 1e-9 and abs(h1 + h2 - pct_vs_identity(f, pred)) < 1e-9


def test_ar1_per_channel_is_near_optimal_and_beats_identity():
    f = [_fold(seed=s) for s in (0, 2)]
    # AR(1) with rho = 0.7: identity MSE 2(1-rho), optimal 1-rho^2, so the improvement over identity is (1-rho)/2 = 15% in every channel
    assert abs(pct_vs_identity(f, "linear_diag") - 15.0) < 1.5 and abs(pct_vs_identity(f, "noint") - 15.0) < 1.5


def test_slopes_are_scale_invariant_so_whitening_changes_only_the_metric_weighting():
    raw, wh = _fold(whiten=False), _fold(whiten=True)
    # per-dimension relative improvement of the per-channel probe is identical with or without whitening
    r = 1 - raw["linear_diag"] / raw["identity"]; w = 1 - wh["linear_diag"] / wh["identity"]
    assert np.allclose(r, w, atol=1e-6)


def test_unweighted_view_has_one_entry_per_band():
    f = [_fold()]
    assert len(band_unweighted(f, "linear_diag", bands=[(0, 5), (5, 12)])) == 2
