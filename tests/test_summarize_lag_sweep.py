"""
tests/test_summarize_lag_sweep.py

Unit tests for scripts/summarize_lag_sweep.py's log-parsing logic.

CPU-only, synthetic log text — no cluster output needed. Run with:

    pytest tests/test_summarize_lag_sweep.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from summarize_lag_sweep import parse_sweep_output  # noqa: E402


def _sample_output(lag: int, zero_mse: float, mean_mse: float, reduction: float) -> str:
    return f"""[posmean] Loaded normalization stats from /scratch/x/norm_stats.pt
[posmean] lag={lag} frames ({lag * 10}ps)
Protein             ZeroMSE   PosMeanMSE   %Reduction          N
----------------------------------------------------------------
1a62_A             0.000633     0.000633        0.00%      30003
----------------------------------------------------------------

[summary] Aggregate zero-baseline MSE:        {zero_mse:.6f}
[summary] Aggregate position-mean baseline MSE: {mean_mse:.6f}
[summary] Reduction from static position mean:  {reduction:.2f}%

[summary] If this reduction is small (a few percent), frame-to-frame
"""


class TestParseSweepOutput:
    def test_parses_complete_output(self):
        text = _sample_output(lag=100, zero_mse=0.000838, mean_mse=0.000700, reduction=16.47)
        result = parse_sweep_output(text)
        assert result is not None
        assert result["lag_frames"] == 100
        assert result["zero_baseline_mse"] == 0.000838
        assert result["position_mean_mse"] == 0.0007
        assert result["reduction_pct"] == 16.47

    def test_returns_none_for_incomplete_output(self):
        """A job killed mid-run (e.g. walltime) won't have the summary block."""
        text = "[posmean] lag=1 frames (10ps)\nProtein ...\n1a62_A ... \n"
        assert parse_sweep_output(text) is None

    def test_returns_none_for_unrelated_text(self):
        assert parse_sweep_output("some other log entirely") is None

    def test_handles_zero_lag_reduction(self):
        text = _sample_output(lag=1, zero_mse=0.000838, mean_mse=0.000838, reduction=0.00)
        result = parse_sweep_output(text)
        assert result["lag_frames"] == 1
        assert result["reduction_pct"] == 0.0

    def test_handles_large_lag_values(self):
        text = _sample_output(lag=5000, zero_mse=0.05, mean_mse=0.01, reduction=80.0)
        result = parse_sweep_output(text)
        assert result["lag_frames"] == 5000
        assert result["reduction_pct"] == 80.0
