import torch
from fluxion.analysis.axis_stability import axis_cos, capture_ratio, fit_axes, heldout_variance, principal_cos2


def _data(n, scales, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, len(scales), generator=g) * scales


def test_clear_head_is_stable_and_flat_tail_is_not():
    scales = torch.cat([torch.tensor([10.0, 6.0, 4.0]), torch.ones(37)])          # 3 well-separated axes, then an isotropic tail
    (_, UA, _), (_, UB, _) = fit_axes(_data(300, scales, 0), 40), fit_axes(_data(300, scales, 1), 40)
    ac = axis_cos(UA, UB)
    assert ac[:3].min() > 0.95 and ac[10:].mean() < 0.5
    assert principal_cos2(UA, UB, 3) > 0.95 and principal_cos2(UA, UB, 3) > principal_cos2(UA, UB, 30)


def test_capture_ratio_is_one_for_own_basis_and_at_most_one_otherwise():
    scales = torch.cat([torch.tensor([10.0, 6.0, 4.0]), torch.ones(37)])
    Xh = _data(300, scales, 2); _, U_own, _ = fit_axes(Xh, 20); _, U_other, _ = fit_axes(_data(300, scales, 3), 20)
    assert abs(capture_ratio(Xh, U_own, 10) - 1.0) < 1e-4
    assert capture_ratio(Xh, U_other, 10) <= 1.0 + 1e-6


def test_heldout_variance_matches_training_variance_in_distribution():
    scales = torch.cat([torch.tensor([10.0, 6.0]), torch.ones(18)])
    mu, U, var = fit_axes(_data(2000, scales, 0), 5)
    hv = heldout_variance(_data(2000, scales, 1), mu, U)
    assert (hv[:2] / var[:2] - 1).abs().max() < 0.15
