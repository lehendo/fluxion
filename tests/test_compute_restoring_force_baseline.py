"""
tests/test_compute_restoring_force_baseline.py

Unit tests for scripts/compute_restoring_force_baseline.py.

CPU-only, synthetic latent store — no cluster data needed. Run with:

    pytest tests/test_compute_restoring_force_baseline.py -v

Design note: the correct NULL model for "no restoring-force signal" in this
regime is a pure random walk (z1 = z0 + small independent noise), NOT i.i.d.
frames. u = z1 - z0 mechanically contains "-z0", so regressing u on
INDEPENDENT frames produces a large spurious correlation (Cov(u,z0) =
-Var(z0) exactly) that has nothing to do with real dynamics — see the
"subtlety" section of compute_restoring_force_baseline.py's docstring. Since
our real data is highly autocorrelated (rho ~ 0.9996 at lag=1), the random
walk is the representative null; i.i.d. frames are not.
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

from compute_restoring_force_baseline import compute_restoring_force_baseline  # noqa: E402
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


class TestComputeRestoringForceBaseline:
    def test_random_walk_null_shows_no_restoring_force_signal(self, tmp_path):
        """Pure random walk (independent small steps, no mean reversion) is
        the correct 'no signal' null for this regime. Cov(u, z0) = 0 exactly
        in expectation here, so restoring_force_mse should stay close to
        position_mean_mse."""
        L, d, n = 4, 3, 2000
        rng = np.random.default_rng(10)
        steps = rng.standard_normal((n - 1, L, d)).astype(np.float32) * 0.1
        frames = np.concatenate([np.zeros((1, L, d), dtype=np.float32), np.cumsum(steps, axis=0)])
        _write_protein_with_frames(tmp_path, "protRandomWalk", frames)

        ds = LatentDataset(tmp_path, "protRandomWalk")
        result = compute_restoring_force_baseline(ds, batch_size=64, num_workers=0)

        reduction = 1 - result["restoring_force_mse"] / result["position_mean_mse"]
        assert reduction < 0.1, (
            f"pure random walk (no restoring force) should show ~no reduction "
            f"vs position-mean, got {reduction:.4f}"
        )

    def test_ou_process_reveals_restoring_force_signal(self, tmp_path):
        """Genuine mean-reverting AR(1)/OU dynamics: x_{t+1} = (1-k)*x_t + eps.
        u_t = x_{t+1} - x_t = -k*x_t + eps, so u is genuinely, strongly linearly
        predictable from z0 = x_t. Bounded/stationary process, so no float16
        cumulative-magnitude precision trap like a pure random walk would have
        at large n.

        Closed-form expected reduction: for a stationary AR(1) with reversion
        strength k (0 < k < 2) and any noise level, reduction_vs_position_mean
        = k/2 exactly (independent of eps_std — derived from the AR(1)
        stationary-variance identity Var(x) = eps_std^2 / (2k - k^2)). For
        k=0.6 that's 30% — use a comfortable margin below that theoretical
        value to stay robust to finite-sample noise, well above the random-
        walk null's near-zero reduction.
        """
        L, d, n = 4, 3, 2000
        rng = np.random.default_rng(11)
        k_true = 0.6
        eps_std = 0.05

        frames = np.zeros((n, L, d), dtype=np.float32)
        frames[0] = rng.standard_normal((L, d)).astype(np.float32)
        for i in range(n - 1):
            frames[i + 1] = (1 - k_true) * frames[i] + (
                rng.standard_normal((L, d)).astype(np.float32) * eps_std
            )
        _write_protein_with_frames(tmp_path, "protOU", frames)

        ds = LatentDataset(tmp_path, "protOU")
        result = compute_restoring_force_baseline(ds, batch_size=64, num_workers=0)

        reduction = 1 - result["restoring_force_mse"] / result["position_mean_mse"]
        assert reduction > 0.15, (
            f"genuine OU/restoring-force dynamics (theoretical reduction ~30% "
            f"for k={k_true}) should show clear reduction vs position-mean, "
            f"got {reduction:.4f}"
        )

    def test_restoring_force_mse_never_exceeds_position_mean_mse(self, tmp_path):
        """OLS with an extra free parameter (slope on z0) can never fit worse
        in-sample than the intercept-only (position-mean) model — the extra
        parameter can always be set to zero to recover it exactly."""
        L, d, n = 4, 3, 500
        rng = np.random.default_rng(12)
        steps = rng.standard_normal((n - 1, L, d)).astype(np.float32) * 0.15
        frames = np.concatenate([np.zeros((1, L, d), dtype=np.float32), np.cumsum(steps, axis=0)])
        _write_protein_with_frames(tmp_path, "protGeneric", frames)

        ds = LatentDataset(tmp_path, "protGeneric")
        result = compute_restoring_force_baseline(ds, batch_size=64, num_workers=0)

        assert result["restoring_force_mse"] <= result["position_mean_mse"] + 1e-9

    def test_self_pairs_excluded(self, tmp_path):
        L, d, n = 4, 3, 100
        rng = np.random.default_rng(13)
        frames = np.cumsum(rng.standard_normal((n, L, d)).astype(np.float32) * 0.1, axis=0)
        _write_protein_with_frames(tmp_path, "protLagExclude", frames)

        lag = 90
        ds = LatentDataset(tmp_path, "protLagExclude", lag=lag)
        result = compute_restoring_force_baseline(ds, batch_size=16, num_workers=0)

        assert result["n_excluded_self_pairs"] == lag
        assert result["n_samples"] == n - lag

    def test_n_samples_and_L_reported_correctly(self, tmp_path):
        L, d, n = 5, 3, 20
        frames = np.random.default_rng(14).standard_normal((n, L, d)).astype(np.float32)
        _write_protein_with_frames(tmp_path, "protShape", frames)

        ds = LatentDataset(tmp_path, "protShape")
        result = compute_restoring_force_baseline(ds, batch_size=8, num_workers=0)

        assert result["n_samples"] == n - 1  # last frame self-pairs, excluded
        assert result["n_excluded_self_pairs"] == 1
        assert result["L"] == L

    def test_matches_position_mean_baseline_on_shared_fields(self, tmp_path):
        """zero_baseline_mse and position_mean_mse must agree exactly with
        the standalone compute_position_mean_baseline tool on the same data —
        both use the identical sum-of-squares identities."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from compute_position_mean_baseline import compute_position_mean_baseline

        L, d, n = 4, 3, 300
        rng = np.random.default_rng(15)
        steps = rng.standard_normal((n - 1, L, d)).astype(np.float32) * 0.1
        frames = np.concatenate([np.zeros((1, L, d), dtype=np.float32), np.cumsum(steps, axis=0)])
        _write_protein_with_frames(tmp_path, "protCross", frames)

        ds1 = LatentDataset(tmp_path, "protCross")
        ds2 = LatentDataset(tmp_path, "protCross")

        result_a = compute_restoring_force_baseline(ds1, batch_size=32, num_workers=0)
        result_b = compute_position_mean_baseline(ds2, batch_size=32, num_workers=0)

        assert result_a["zero_baseline_mse"] == pytest.approx(result_b["zero_baseline_mse"], rel=1e-9)
        assert result_a["position_mean_mse"] == pytest.approx(result_b["position_mean_mse"], rel=1e-9)
