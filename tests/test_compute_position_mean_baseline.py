"""
tests/test_compute_position_mean_baseline.py

Unit tests for scripts/compute_position_mean_baseline.py.

CPU-only, synthetic latent store — no cluster data needed. Run with:

    pytest tests/test_compute_position_mean_baseline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compute_position_mean_baseline import compute_position_mean_baseline  # noqa: E402
from data_pipeline.dataset import LatentDataset  # noqa: E402


def _write_protein_with_frames(
    root: Path,
    protein: str,
    frames: np.ndarray,  # [n_frames, L, d] float32
) -> None:
    protein_dir = root / protein
    shard_dir = protein_dir / protein
    shard_dir.mkdir(parents=True)
    n, L, d = frames.shape
    with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
        f.create_dataset("single", data=frames.astype(np.float16))
        f.create_dataset("frame_idx", data=np.arange(n, dtype=np.int32))
    manifest = {protein: {"shards": [{
        "path": f"{protein}/shard_000000.h5", "n_frames": n,
        "first_idx": 0, "last_idx": n - 1,
    }]}}
    with open(protein_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    pair = torch.randn(L, L, 4).half()
    torch.save(pair, protein_dir / "pair.pt")


class TestComputePositionMeanBaseline:
    def test_deterministic_drift_fully_explained_by_position_mean(self, tmp_path):
        """Each residue drifts by a FIXED, position-specific vector every
        frame with zero noise. The position-mean predictor should explain
        nearly all of it, while the zero baseline sees the full drift
        magnitude.

        Two precision traps to avoid in the synthetic data:
          1. LatentDataset self-pairs the trajectory's last frame (u_t=0
             there, by design — see test_dataset.py::test_last_frame_wraps),
             so with too few frames that one boundary sample dominates the
             residual sum of squares. Need n large enough that it's
             statistically negligible.
          2. Shards store frames as float16. A cumulative sum of a constant
             drift grows frame magnitude to n*drift, and float16's ULP
             scales with magnitude/2^11 — so for large n, the per-step drift
             becomes smaller than the storage's own quantization step and
             gets swallowed by rounding noise. Need n*drift bounded well
             within float16's precise range.
        Both push in opposite directions on n, so keep n moderate (200) and
        assert a generous (not razor-edge) reduction threshold.
        """
        L, d, n = 4, 3, 200
        rng = np.random.default_rng(0)
        drift = rng.standard_normal((L, d)).astype(np.float32) * 1.0  # fixed per-position drift

        frames = np.zeros((n, L, d), dtype=np.float32)
        for i in range(1, n):
            frames[i] = frames[i - 1] + drift
        _write_protein_with_frames(tmp_path, "protDrift", frames)

        ds = LatentDataset(tmp_path, "protDrift")
        result = compute_position_mean_baseline(ds, batch_size=64, num_workers=0)

        assert result["zero_baseline_mse"] > 0.1   # substantial — real drift magnitude
        reduction = 1 - result["position_mean_mse"] / result["zero_baseline_mse"]
        assert reduction > 0.9, f"position mean should explain most of the drift, got {reduction:.4f}"

    def test_pure_noise_position_mean_does_not_help(self, tmp_path):
        """Zero-mean random noise frame-to-frame: the position mean should
        converge close to zero, giving position_mean_mse ~= zero_baseline_mse."""
        L, d, n = 4, 3, 500
        rng = np.random.default_rng(1)
        frames = np.cumsum(
            rng.standard_normal((n, L, d)).astype(np.float32) * 0.1, axis=0
        )
        _write_protein_with_frames(tmp_path, "protNoise", frames)

        ds = LatentDataset(tmp_path, "protNoise")
        result = compute_position_mean_baseline(ds, batch_size=32, num_workers=0)

        # With enough samples, the two should be close (mean of near-zero-mean
        # noise contributes little reduction) — not exactly equal due to finite
        # sample size, but well within a generous relative tolerance.
        ratio = result["position_mean_mse"] / result["zero_baseline_mse"]
        assert ratio > 0.8, f"position mean reduced MSE by too much for pure noise: ratio={ratio}"

    def test_n_samples_and_L_reported_correctly(self, tmp_path):
        L, d, n = 5, 3, 20
        frames = np.random.default_rng(2).standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protShape", frames)

        ds = LatentDataset(tmp_path, "protShape")
        result = compute_position_mean_baseline(ds, batch_size=8, num_workers=0)

        # LatentDataset self-pairs the last frame (no i+lag partner); that one
        # artificial zero-displacement sample is excluded from the statistics.
        assert result["n_samples"] == n - 1
        assert result["n_excluded_self_pairs"] == 1
        assert result["L"] == L

    def test_excludes_self_paired_samples_at_large_lag(self, tmp_path):
        """At lag close to the trajectory length, most samples are artificial
        self-pairs and must be excluded, not diluted into the statistics."""
        L, d, n = 4, 3, 100
        rng = np.random.default_rng(5)
        frames = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32) * 0.2, axis=0)
        _write_protein_with_frames(tmp_path, "protLagExclude", frames)

        lag = 90  # only the first (n - lag) = 10 frames have a real partner
        ds = LatentDataset(tmp_path, "protLagExclude", lag=lag)
        result = compute_position_mean_baseline(ds, batch_size=16, num_workers=0)

        assert result["n_excluded_self_pairs"] == lag
        assert result["n_samples"] == n - lag

    def test_position_mean_never_worse_than_zero_by_more_than_float_noise(self, tmp_path):
        """The position mean is the least-squares-optimal constant predictor
        per position — it can never do WORSE than zero (zero is just the
        special case of an all-zero mean)."""
        L, d, n = 4, 3, 30
        frames = np.random.default_rng(3).standard_normal((n, L, d)).astype(np.float32) * 5.0
        _write_protein_with_frames(tmp_path, "protGeneric", frames)

        ds = LatentDataset(tmp_path, "protGeneric")
        result = compute_position_mean_baseline(ds, batch_size=8, num_workers=0)

        assert result["position_mean_mse"] <= result["zero_baseline_mse"] + 1e-6

    def test_longer_lag_reveals_signal_buried_in_short_lag_noise(self, tmp_path):
        """This is the exact mechanism slurm/analysis/sweep_lag_baseline.sh is built
        to detect: a small constant per-step position-specific drift, buried
        under large per-step independent noise. At lag=1 the noise dominates
        and the drift is barely detectable; the drift accumulates linearly
        with lag while the noise's std only grows as sqrt(lag), so a longer
        lag should improve the drift's share of explainable variance."""
        L, d, n = 4, 3, 2000
        rng = np.random.default_rng(4)
        drift = rng.standard_normal((L, d)).astype(np.float32) * 0.05   # small, per-step
        noise_std = 2.0                                                 # large, per-step

        steps = drift[None, :, :] + rng.standard_normal((n - 1, L, d)).astype(np.float32) * noise_std
        frames = np.concatenate([np.zeros((1, L, d), dtype=np.float32), np.cumsum(steps, axis=0)])
        _write_protein_with_frames(tmp_path, "protBuriedDrift", frames)

        ds_short = LatentDataset(tmp_path, "protBuriedDrift", lag=1)
        ds_long  = LatentDataset(tmp_path, "protBuriedDrift", lag=20)

        result_short = compute_position_mean_baseline(ds_short, batch_size=64, num_workers=0)
        result_long  = compute_position_mean_baseline(ds_long,  batch_size=64, num_workers=0)

        reduction_short = 1 - result_short["position_mean_mse"] / result_short["zero_baseline_mse"]
        reduction_long  = 1 - result_long["position_mean_mse"]  / result_long["zero_baseline_mse"]

        assert reduction_long > reduction_short, (
            f"longer lag should reveal more of the buried drift signal: "
            f"lag=1 reduction={reduction_short:.4f}, lag=20 reduction={reduction_long:.4f}"
        )
