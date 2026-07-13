"""
tests/test_compute_velocity_baseline.py

Unit tests for scripts/compute_velocity_baseline.py.

CPU-only, synthetic latent store — no cluster data needed. Run with:

    pytest tests/test_compute_velocity_baseline.py -v

Design note: like compute_restoring_force_baseline.py's z1_predictability_mse,
this tool regresses z1 DIRECTLY (never a difference) on z0 and z_prev, to
avoid the mechanical self-inclusion artifact that a naive u-on-v regression
would have (verified algebraically before writing the tool: regressing
u=z1-z0 on v=z0-z_prev hits a spurious 25% "reduction" ceiling under true
independence, for the same reason the z0-only restoring-force test hit 50%).
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

from compute_velocity_baseline import compute_velocity_baseline  # noqa: E402
from data_pipeline.dataset import LatentDataset  # noqa: E402


def _write_protein_with_frames(root: Path, protein: str, frames: np.ndarray) -> None:
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


class TestComputeVelocityBaseline:
    def test_independent_frames_show_no_signal_at_either_level(self, tmp_path):
        """Fully independent frames: neither z0 alone nor z0+velocity should
        show any predictability of z1 — confirms the two-predictor design
        avoided the mechanical-artifact trap (no spurious 25%/50% ceiling)."""
        L, d, n = 4, 3, 3000
        frames = np.random.default_rng(30).standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protIID", frames)

        ds = LatentDataset(tmp_path, "protIID", velocity_lag=1)
        result = compute_velocity_baseline(ds, batch_size=128, num_workers=0)

        z0_reduction = 1 - result["z0_only_mse"] / result["z1_variance_mse"]
        vel_reduction = 1 - result["z0_and_velocity_mse"] / result["z0_only_mse"]
        assert z0_reduction < 0.05, f"z0-only should show ~0% for iid data, got {z0_reduction:.4f}"
        assert vel_reduction < 0.05, f"velocity should add ~0% for iid data, got {vel_reduction:.4f}"

    def test_markov_ar1_process_shows_no_incremental_velocity_benefit(self, tmp_path):
        """A true first-order Markov process (x_{t+1} depends ONLY on x_t, not
        on x_{t-1}): z0 alone should show real signal (matches the
        restoring-force test), but adding z_prev should add ~nothing more —
        that's the definition of the Markov property."""
        L, d, n = 4, 3, 3000
        rng = np.random.default_rng(31)
        phi = 0.4
        eps_std = 0.05
        frames = np.zeros((n, L, d), dtype=np.float32)
        frames[0] = rng.standard_normal((L, d)).astype(np.float32)
        for i in range(n - 1):
            frames[i + 1] = phi * frames[i] + rng.standard_normal((L, d)).astype(np.float32) * eps_std
        _write_protein_with_frames(tmp_path, "protMarkov", frames)

        ds = LatentDataset(tmp_path, "protMarkov", velocity_lag=1)
        result = compute_velocity_baseline(ds, batch_size=128, num_workers=0)

        z0_reduction = 1 - result["z0_only_mse"] / result["z1_variance_mse"]
        vel_reduction = 1 - result["z0_and_velocity_mse"] / result["z0_only_mse"]
        assert z0_reduction > 0.1, f"z0-only should show real signal (phi^2={phi**2}), got {z0_reduction:.4f}"
        assert vel_reduction < 0.05, (
            f"a true Markov process should show ~0% incremental benefit from "
            f"velocity, got {vel_reduction:.4f}"
        )

    def test_momentum_process_shows_incremental_velocity_benefit(self, tmp_path):
        """A genuine second-order/momentum process — displacement persists in
        the same direction as recent velocity — x_{t+1} = x_t + c*(x_t -
        x_{t-1}) + eps. Here z_prev SHOULD add real predictive value beyond
        z0 alone, since the true dynamics literally depend on the recent
        velocity (x_t - x_{t-1}), not just the current position."""
        L, d, n = 4, 3, 1500
        rng = np.random.default_rng(32)
        c = 0.4
        eps_std = 0.05

        frames = np.zeros((n, L, d), dtype=np.float32)
        frames[0] = rng.standard_normal((L, d)).astype(np.float32) * 0.1
        frames[1] = frames[0] + rng.standard_normal((L, d)).astype(np.float32) * eps_std
        for i in range(1, n - 1):
            velocity = frames[i] - frames[i - 1]
            frames[i + 1] = frames[i] + c * velocity + (
                rng.standard_normal((L, d)).astype(np.float32) * eps_std
            )
        _write_protein_with_frames(tmp_path, "protMomentum", frames)

        ds = LatentDataset(tmp_path, "protMomentum", velocity_lag=1)
        result = compute_velocity_baseline(ds, batch_size=128, num_workers=0)

        vel_reduction = 1 - result["z0_and_velocity_mse"] / result["z0_only_mse"]
        assert vel_reduction > 0.1, (
            f"genuine momentum dynamics should show a real incremental "
            f"reduction from adding velocity, got {vel_reduction:.4f}"
        )

    def test_invariant_monotonic_reduction_with_more_predictors(self, tmp_path):
        """Adding predictors can never increase in-sample residual:
        z0_and_velocity_mse <= z0_only_mse <= z1_variance_mse always."""
        L, d, n = 4, 3, 500
        rng = np.random.default_rng(33)
        frames = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32) * 0.1, axis=0)
        _write_protein_with_frames(tmp_path, "protGeneric", frames)

        ds = LatentDataset(tmp_path, "protGeneric", velocity_lag=1)
        result = compute_velocity_baseline(ds, batch_size=64, num_workers=0)

        assert result["z0_and_velocity_mse"] <= result["z0_only_mse"] + 1e-9
        assert result["z0_only_mse"] <= result["z1_variance_mse"] + 1e-9

    def test_excludes_head_and_tail_self_pairs(self, tmp_path):
        L, d, n = 4, 3, 100
        rng = np.random.default_rng(34)
        frames = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32) * 0.1, axis=0)
        _write_protein_with_frames(tmp_path, "protExclude", frames)

        velocity_lag, lag = 3, 5
        ds = LatentDataset(tmp_path, "protExclude", lag=lag, velocity_lag=velocity_lag)
        result = compute_velocity_baseline(ds, batch_size=16, num_workers=0)

        # First `velocity_lag` frames self-pair at head, last `lag` frames
        # self-pair at tail — total excluded is the union (no overlap here
        # since velocity_lag + lag < n).
        assert result["n_excluded_self_pairs"] == velocity_lag + lag
        assert result["n_samples"] == n - velocity_lag - lag

    def test_n_samples_and_L_reported(self, tmp_path):
        L, d, n = 5, 3, 50
        frames = np.random.default_rng(35).standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protShape", frames)

        ds = LatentDataset(tmp_path, "protShape", velocity_lag=1)
        result = compute_velocity_baseline(ds, batch_size=8, num_workers=0)

        assert result["L"] == L
        assert result["n_samples"] > 0
