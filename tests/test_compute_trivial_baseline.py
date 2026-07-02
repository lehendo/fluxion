"""
tests/test_compute_trivial_baseline.py

Unit tests for scripts/compute_trivial_baseline.py.

CPU-only, synthetic latent store — no cluster data needed. Run with:

    pytest tests/test_compute_trivial_baseline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compute_trivial_baseline import compute_trivial_baseline_mse  # noqa: E402
from data_pipeline.dataset import build_multi_protein_dataset, collate_fn  # noqa: E402

from tests.test_dataset import make_protein_dir, D_S, L_A, N_FRAMES_A  # noqa: E402


class TestComputeTrivialBaselineMSE:
    def test_zero_when_frames_identical(self, tmp_path):
        """If z1 always equals z0 (a static, unmoving 'trajectory'), the
        trivial-predictor baseline must be exactly zero."""
        protein_dir = tmp_path / "static"
        shard_dir = protein_dir / "static"
        shard_dir.mkdir(parents=True)

        import h5py
        import numpy as np
        import json

        n, L, d = 8, 5, D_S
        constant = np.full((n, L, d), 3.0, dtype=np.float16)
        with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
            f.create_dataset("single", data=constant)
            f.create_dataset("frame_idx", data=np.arange(n, dtype=np.int32))
        manifest = {"static": {"shards": [{
            "path": "static/shard_000000.h5", "n_frames": n, "first_idx": 0, "last_idx": n - 1,
        }]}}
        with open(protein_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        pair = torch.randn(L, L, 4).half()
        torch.save(pair, protein_dir / "pair.pt")

        dataset = build_multi_protein_dataset(tmp_path)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

        baseline, n_samples = compute_trivial_baseline_mse(loader, n_batches=2)
        assert baseline == pytest.approx(0.0, abs=1e-8)
        assert n_samples == 8

    def test_nonzero_when_frames_differ(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        dataset = build_multi_protein_dataset(tmp_path)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        baseline, n_samples = compute_trivial_baseline_mse(loader, n_batches=3)
        assert baseline > 0.0
        assert n_samples > 0

    def test_matches_manual_computation(self, tmp_path):
        """Baseline must equal mean((z1-z0)^2) computed directly, without the
        padding-aware machinery — true when there's no padding in the batch."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        dataset = build_multi_protein_dataset(tmp_path)
        loader = DataLoader(dataset, batch_size=N_FRAMES_A, shuffle=False, collate_fn=collate_fn)

        baseline, _ = compute_trivial_baseline_mse(loader, n_batches=1)

        (z0, z1), _pair, _fidx, mask = next(iter(loader))
        assert not mask.any()  # single protein, uniform length -> no padding
        expected = ((z1 - z0) ** 2).mean().item()
        assert baseline == pytest.approx(expected, rel=1e-5)

    def test_wraps_around_when_n_batches_exceeds_loader_length(self, tmp_path):
        """n_batches larger than one epoch's worth of batches should wrap
        around to a fresh iterator, not crash."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        dataset = build_multi_protein_dataset(tmp_path)
        # batch_size == dataset size -> exactly one batch per epoch, so the
        # sample count from n_batches requests is exactly n_batches * N_FRAMES_A.
        loader = DataLoader(dataset, batch_size=N_FRAMES_A, shuffle=False, collate_fn=collate_fn)

        baseline, n_samples = compute_trivial_baseline_mse(loader, n_batches=3)
        assert n_samples == 3 * N_FRAMES_A
        assert baseline >= 0.0
