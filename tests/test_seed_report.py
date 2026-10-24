from phase3_seed_report import PRED, classify


def _r(a, de, dr):
    return {"att_ema": a, "D_ema": de, "D_raw": dr}


def test_classification():
    ok = [_r(PRED, 0.5, 0.1), _r(0.9 * PRED, 0.4, 0.5), _r(1.1 * PRED, 0.3, 0.2)]
    assert classify(ok) == "ROBUST"
    assert classify(ok[:2] + [_r(0.4 * PRED, 0.3, 0.2)]) == "CONDITIONAL"
    assert classify(ok[:2] + [_r(PRED, 0.9, 0.2)]) == "OTHER"
    assert classify(ok[:2] + [_r(float("nan"), 0.1, 0.1)]) == "CONDITIONAL"
