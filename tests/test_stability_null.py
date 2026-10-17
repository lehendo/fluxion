import numpy as np
from scripts.phase1b_stability_null import capture_null, kappa, null_axis_cos, null_overlap, verdict


def test_nulls_match_simulation_small_case():
    rng = np.random.default_rng(1); D, m = 60, 10
    s = [np.mean(np.linalg.svd(np.linalg.qr(rng.standard_normal((D, m)))[0].T @ np.linalg.qr(rng.standard_normal((D, m)))[0], compute_uv=False) ** 2) for _ in range(300)]
    assert abs(np.mean(s) - null_overlap(m, D)) < 0.01
    v = rng.standard_normal((4000, D)); w = rng.standard_normal((4000, D))
    c = np.abs((v * w).sum(1) / np.linalg.norm(v, axis=1) / np.linalg.norm(w, axis=1)).mean()
    assert abs(c - null_axis_cos(D)) < 0.01


def test_kappa_and_capture_null():
    assert kappa(0.5, 0.0) == 0.5 and abs(kappa(0.75, 0.5) - 0.5) < 1e-12 and np.isnan(kappa(1.0, 1.0))
    assert capture_null(100, 1000, np.ones(400)) > 0.1 * 400 / 100 * 0 and capture_null(100, 100, np.ones(100)) == 1.0


def test_verdicts():
    assert verdict(0.2, (0.1, 0.3), 0.15, (0.05, 0.25), 0.7, 0.1).startswith("CARTESIAN DEFICIT")
    assert verdict(-0.1, (-0.2, 0.0), 0.0, (-0.05, 0.05), 0.7, 0.1).startswith("NO CARTESIAN DEFICIT")
    assert verdict(-0.1, (-0.2, 0.0), 0.0, (-0.05, 0.05), 0.95, 0.5).startswith("UNINFORMATIVE")
