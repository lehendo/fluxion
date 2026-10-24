import numpy as np
from scipy.integrate import solve_ivp
from check_flow_involution import field, make_mixture


def _flow(v, x0, a, b):
    s = solve_ivp(v, [a, b], x0, rtol=1e-11, atol=1e-12)
    assert s.success
    return s.y[:, -1]


def test_solutions_of_an_odd_in_time_field_are_even_so_the_flow_is_the_identity():
    # a generic nonlinear analytic field, odd in s, NOT derived from any pair
    def v(s, x):
        return s * np.sin(np.array([[1.0, 0.5], [-0.3, 0.8]]) @ x) + s ** 3 * np.cos(x[::-1]) + s ** 5 * np.tanh(x)
    y = np.array([0.4, -0.7])
    fwd, bwd = _flow(v, y, 0.0, 0.5), _flow(v, y, 0.0, -0.5)
    assert np.abs(fwd - bwd).max() < 1e-9                                # phi(1/2) = phi(-1/2)
    assert np.abs(_flow(v, bwd, -0.5, 0.5) - bwd).max() < 1e-9           # Phi = identity


def test_non_gaussian_reversible_mixture_has_identity_flow():
    rng = np.random.default_rng(0)
    for d in (1, 2):
        comps = make_mixture(rng, d, 2, 1.5); f = field(comps, d)
        a0, _, S = comps[0]; x0 = rng.multivariate_normal(a0, S[:d, :d])
        assert np.abs(_flow(f, x0, 0.0, 1.0) - x0).max() < 1e-8
