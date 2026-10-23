"""
tests/test_compute_norm_stats.py

Unit tests for data_pipeline/compute_norm_stats.py.

CPU-only, synthetic HDF5 shards — no cluster data needed. Run with:

    pytest tests/test_compute_norm_stats.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from fluxion.data_pipeline.compute_norm_stats import compute_stats


def _write_protein(root: Path, protein: str, frames: np.ndarray, L: int) -> None:
    """frames: [n_frames, L, d] float32 array of raw per-frame single latents."""
    protein_dir = root / protein
    shard_dir = protein_dir / protein
    shard_dir.mkdir(parents=True)
    n, _, _ = frames.shape
    with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
        f.create_dataset("single", data=frames.astype(np.float16))
        f.create_dataset("frame_idx", data=np.arange(n, dtype=np.int32))
    manifest = {protein: {"shards": [{
        "path": f"{protein}/shard_000000.h5",
        "n_frames": n, "first_idx": 0, "last_idx": n - 1,
    }]}}
    with open(protein_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


class TestComputeStats:
    def test_matches_numpy_reference(self, tmp_path):
        rng = np.random.default_rng(0)
        L, d, n = 5, 6, 200
        # distinct per-channel scale so normalization is actually exercised
        scale = np.array([1.0, 5.0, 0.1, 20.0, 3.0, 8.0])
        frames = (rng.standard_normal((n, L, d)) * scale).astype(np.float32)
        _write_protein(tmp_path, "protA", frames, L)

        stats = compute_stats(tmp_path)

        flat = frames.reshape(-1, d)
        expected_mean = flat.mean(axis=0)
        expected_std  = flat.std(axis=0)

        # float16 shard storage + finite-sample noise — generous tolerance
        assert np.allclose(stats["mean"].numpy(), expected_mean, atol=0.1)
        assert np.allclose(stats["std"].numpy(), expected_std, rtol=0.1, atol=0.05)
        assert stats["n_frames"] == n * L
        assert stats["d_single"] == d

    def test_protein_filter(self, tmp_path):
        L, d, n = 4, 3, 10
        frames_a = np.ones((n, L, d), dtype=np.float32)
        frames_b = np.zeros((n, L, d), dtype=np.float32)
        _write_protein(tmp_path, "protA", frames_a, L)
        _write_protein(tmp_path, "protB", frames_b, L)

        stats = compute_stats(tmp_path, protein_ids=["protA"])
        assert stats["n_frames"] == n * L
        assert np.allclose(stats["mean"].numpy(), 1.0, atol=1e-2)

    def test_multiple_proteins_combined(self, tmp_path):
        L, d, n = 4, 3, 10
        frames_a = np.zeros((n, L, d), dtype=np.float32)
        frames_b = np.full((n, L, d), 2.0, dtype=np.float32)
        _write_protein(tmp_path, "protA", frames_a, L)
        _write_protein(tmp_path, "protB", frames_b, L)

        stats = compute_stats(tmp_path)
        assert stats["n_frames"] == 2 * n * L
        # Combined mean of a 0-block and a 2-block, equal size, is 1.0
        assert np.allclose(stats["mean"].numpy(), 1.0, atol=1e-2)

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(RuntimeError):
            compute_stats(tmp_path)

    def test_ignores_non_protein_dirs(self, tmp_path):
        L, d, n = 4, 3, 10
        frames = np.ones((n, L, d), dtype=np.float32)
        _write_protein(tmp_path, "protA", frames, L)
        (tmp_path / "esm_cache").mkdir()  # no manifest.json inside — must be skipped

        stats = compute_stats(tmp_path)
        assert stats["n_frames"] == n * L
