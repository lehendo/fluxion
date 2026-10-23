import numpy as np
from phase1b_controls_report import band_vec, read_c1, read_c2, read_c3, tail_contrib


def test_c1_thresholds():
    assert read_c1(9.0, 5.0, 1.0, 9.0) == "SURVIVES"
    assert read_c1(9.0, 3.0, 0.5, 6.0) == "INTERMEDIATE"      # between 0.25x and 0.5x
    assert read_c1(9.0, 1.0, -2.0, 4.0) == "COLLAPSES" and read_c1(9.0, 6.0, -0.1, 12.0) == "COLLAPSES"


def test_c2_and_c3_thresholds():
    assert read_c2(0.6, [1, 1]) == "TAIL" and read_c2(0.1, [-1, 2]) == "HEAD" and read_c2(0.1, [-1, -2]) == "MIXED" and read_c2(0.4, [1, 1]) == "MIXED"
    assert read_c3(9.0, 3.0, -1.0, 7.0) == "INTERCEPT-DRIVEN" and read_c3(9.0, 8.0, 2.0, 14.0) == "SLOPE-DRIVEN" and read_c3(9.0, 5.0, 1.0, 9.0) == "INTERMEDIATE"


def test_band_vec_sums_to_total_and_halves_agree():
    rng = np.random.default_rng(0); mid = np.sort(rng.random(100))[::-1] * 10; mp = mid * rng.uniform(0.5, 1.0, 100)
    total = 100 * (mid - mp).sum() / mid.sum()
    assert abs(band_vec(mid, mp).sum() - total) < 1e-9 and abs(sum(tail_contrib(mid, mp)) - total) < 1e-9
