from phase1b_stability_report import reading


def test_reading_rules():
    assert reading(0.10, (0.03, 0.2), 0.08, (0.01, 0.15)) == "MECHANISM IDENTIFIED"
    assert reading(0.02, (-0.03, 0.07), 0.01, (-0.04, 0.05)) == "MECHANISM REJECTED"
    assert reading(-0.04, (-0.1, 0.0), 0.10, (0.02, 0.2)) == "MECHANISM REJECTED"        # Cartesian more stable in one
    assert reading(0.10, (0.03, 0.2), 0.02, (-0.02, 0.06)) == "INTERMEDIATE"
