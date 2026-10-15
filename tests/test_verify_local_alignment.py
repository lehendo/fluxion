import numpy as np
from scripts.verify_atlas_local_alignment import MIN_ALIGNED, MIN_MARGIN, N_PAIRS, N_START, WINDOW, M_OFFSET, shifted_starts, start_frames, verdict


def test_start_frames_are_fixed_unique_and_in_range():
    a, b = start_frames(), start_frames()
    assert (a == b).all() and len(set(a.tolist())) == N_PAIRS
    assert a.min() >= 0 and a.max() + M_OFFSET < WINDOW + 1


def test_shifted_starts_stay_valid_and_move_by_half_window():
    js = start_frames(); sh = shifted_starts(js)
    assert sh.min() >= 0 and sh.max() + M_OFFSET <= WINDOW
    assert ((sh - js) % N_START == (WINDOW // 2) % N_START).all()


def test_verdict_thresholds():
    assert verdict(0.6, [0.2, 0.3])["pass"]
    assert not verdict(0.6, [0.2, 0.46])["pass"]           # margin 0.14 < 0.15
    assert not verdict(0.19, [-0.5])["pass"]                # below the absolute floor
    assert verdict(MIN_ALIGNED + MIN_MARGIN + 1e-9, [0.0])["pass"] or MIN_ALIGNED + MIN_MARGIN > 1
