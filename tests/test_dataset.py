"""
tests/test_dataset.py

Unit tests for data_pipeline/dataset.py.

Tests run entirely on CPU with synthetic data — no Chai-1, no HDF5 files
needed from the cluster.  Run with:

    pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data_pipeline.dataset import (
    LatentDataset,
    build_multi_protein_dataset,
    collate_fn,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic latent store
# ---------------------------------------------------------------------------

L_A, L_B   = 10, 15      # sequence lengths for two synthetic proteins
D_S, D_P   = 8, 4        # tiny single / pair dims for speed
N_FRAMES_A  = 7           # frames per protein A
N_FRAMES_B  = 5


def _write_shard(
    protein_dir: Path,
    shard_name: str,
    n_frames: int,
    L: int,
    d_s: int = D_S,
    start_frame: int = 0,
) -> Path:
    path = protein_dir / shard_name
    with h5py.File(path, "w") as f:
        singles = np.random.randn(n_frames, L, d_s).astype(np.float16)
        frames  = np.arange(start_frame, start_frame + n_frames, dtype=np.int32)
        f.create_dataset("single",    data=singles)
        f.create_dataset("frame_idx", data=frames)
    return path


def _write_pair(protein_dir: Path, L: int) -> None:
    pair = torch.randn(L, L, D_P).half()
    torch.save(pair, protein_dir / "pair.pt")


def _write_manifest(out_dir: Path, protein: str, shard_entries: list[dict]) -> None:
    manifest = {protein: {"shards": shard_entries}}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


def make_protein_dir(
    root: Path,
    protein: str,
    n_frames: int,
    L: int,
    *,
    shard_size: int = 4,
    duplicate_manifest_entries: bool = False,
) -> Path:
    """Create a synthetic protein directory under root."""
    protein_dir = root / protein
    shard_dir   = protein_dir / protein   # mirrors actual nested layout
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    written = 0
    shard_idx = 0
    while written < n_frames:
        batch = min(shard_size, n_frames - written)
        name = f"shard_{shard_idx:06d}.h5"
        _write_shard(shard_dir, name, batch, L, start_frame=written)
        entry = {
            "path":      f"{protein}/{name}",
            "n_frames":  batch,
            "first_idx": written,
            "last_idx":  written + batch - 1,
        }
        shards.append(entry)
        if duplicate_manifest_entries:
            shards.append(entry.copy())   # simulate concurrent-replica bug
        written    += batch
        shard_idx  += 1

    _write_manifest(protein_dir, protein, shards)
    _write_pair(protein_dir, L)
    return protein_dir


# ---------------------------------------------------------------------------
# LatentDataset tests
# ---------------------------------------------------------------------------

class TestLatentDataset:
    def test_len_single_shard(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        assert len(ds) == N_FRAMES_A

    def test_getitem_shapes(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        (z0, z1), pair, frame_idx = ds[0]
        assert z0.shape   == (L_A, D_S)
        assert z1.shape   == (L_A, D_S)
        assert pair.shape == (L_A, L_A, D_P)
        assert isinstance(frame_idx, int)

    def test_consecutive_frames(self, tmp_path):
        """z1 at index i should equal z0 at index i+1 (consecutive pair)."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        (_, z1_at_0), _, _ = ds[0]
        (z0_at_1, _), _, _ = ds[1]
        assert torch.allclose(z1_at_0, z0_at_1)

    def test_last_frame_wraps(self, tmp_path):
        """At the last frame, z0 and z1 should be the same tensor value."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        last = len(ds) - 1
        (z0, z1), _, _ = ds[last]
        assert torch.allclose(z0, z1)

    def test_load_pair_false(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA", load_pair=False)
        (_, _), pair, _ = ds[0]
        assert pair is None

    def test_missing_manifest_raises(self, tmp_path):
        (tmp_path / "ghost").mkdir()
        with pytest.raises(FileNotFoundError, match="Manifest"):
            LatentDataset(tmp_path, "ghost")

    def test_missing_pair_raises(self, tmp_path):
        protein_dir = tmp_path / "nopair"
        shard_dir   = protein_dir / "nopair"
        shard_dir.mkdir(parents=True)
        _write_shard(shard_dir, "shard_000000.h5", 3, L_A)
        _write_manifest(protein_dir, "nopair", [{
            "path": "nopair/shard_000000.h5", "n_frames": 3,
            "first_idx": 0, "last_idx": 2,
        }])
        with pytest.raises(FileNotFoundError, match="pair.pt"):
            LatentDataset(tmp_path, "nopair", load_pair=True)

    def test_duplicate_manifest_entries_deduplicated(self, tmp_path):
        """Concurrent-replica bug: same shard listed 3× — dataset must deduplicate."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A,
                         duplicate_manifest_entries=True)
        ds = LatentDataset(tmp_path, "protA")
        # After dedup, frame count should equal actual frames, not 2×
        assert len(ds) == N_FRAMES_A

    def test_float32_output(self, tmp_path):
        """Shards are stored as float16; dataset must return float32."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        (z0, z1), pair, _ = ds[0]
        assert z0.dtype   == torch.float32
        assert z1.dtype   == torch.float32
        assert pair.dtype == torch.float32


# ---------------------------------------------------------------------------
# build_multi_protein_dataset tests
# ---------------------------------------------------------------------------

class TestBuildMultiProteinDataset:
    def test_discovers_both_proteins(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        make_protein_dir(tmp_path, "protB", N_FRAMES_B, L_B)
        ds = build_multi_protein_dataset(tmp_path)
        assert len(ds) == N_FRAMES_A + N_FRAMES_B

    def test_skips_protein_missing_pair(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        # protB has manifest but no pair.pt
        protB_dir = tmp_path / "protB"
        shard_dir = protB_dir / "protB"
        shard_dir.mkdir(parents=True)
        _write_shard(shard_dir, "shard_000000.h5", N_FRAMES_B, L_B)
        _write_manifest(protB_dir, "protB", [{
            "path": "protB/shard_000000.h5", "n_frames": N_FRAMES_B,
            "first_idx": 0, "last_idx": N_FRAMES_B - 1,
        }])
        ds = build_multi_protein_dataset(tmp_path)
        assert len(ds) == N_FRAMES_A   # protB skipped

    def test_skips_stray_files(self, tmp_path):
        """Top-level manifest.json and esm_cache from a bad run must be ignored."""
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        # Simulate stray top-level artifacts
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "esm_cache").mkdir()
        ds = build_multi_protein_dataset(tmp_path)
        assert len(ds) == N_FRAMES_A

    def test_raises_if_no_valid_proteins(self, tmp_path):
        with pytest.raises(RuntimeError, match="No valid protein"):
            build_multi_protein_dataset(tmp_path)


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------

class TestCollateFn:
    def _make_sample(self, L: int):
        z0   = torch.randn(L, D_S)
        z1   = torch.randn(L, D_S)
        pair = torch.randn(L, L, D_P)
        return (z0, z1), pair, 42

    def test_output_shapes_uniform_length(self):
        batch = [self._make_sample(L_A) for _ in range(3)]
        (z0, z1), pair, fidx, mask = collate_fn(batch)
        assert z0.shape   == (3, L_A, D_S)
        assert z1.shape   == (3, L_A, D_S)
        assert pair.shape == (3, L_A, L_A, D_P)
        assert fidx.shape == (3,)
        assert mask.shape == (3, L_A)
        assert not mask.any()   # no padding needed

    def test_output_shapes_mixed_length(self):
        batch = [self._make_sample(L_A), self._make_sample(L_B)]
        L_max = max(L_A, L_B)
        (z0, z1), pair, fidx, mask = collate_fn(batch)
        assert z0.shape   == (2, L_max, D_S)
        assert pair.shape == (2, L_max, L_max, D_P)
        assert mask.shape == (2, L_max)

    def test_padding_mask_correct(self):
        """Shorter protein should have True (padded) in trailing positions."""
        batch = [self._make_sample(L_A), self._make_sample(L_B)]
        L_max = max(L_A, L_B)
        _, _, _, mask = collate_fn(batch)
        shorter_idx = 0 if L_A < L_B else 1
        longer_idx  = 1 - shorter_idx
        L_short = min(L_A, L_B)
        assert not mask[shorter_idx, :L_short].any()   # real tokens: False
        assert mask[shorter_idx, L_short:].all()        # padded: True
        assert not mask[longer_idx].any()               # longer: no padding

    def test_pad_region_is_zero(self):
        """Padded positions in z0/z1/pair must be exactly zero."""
        batch = [self._make_sample(L_A), self._make_sample(L_B)]
        (z0, _), pair, _, mask = collate_fn(batch)
        shorter_idx = 0 if L_A < L_B else 1
        L_short = min(L_A, L_B)
        assert z0[shorter_idx, L_short:].eq(0).all()
        assert pair[shorter_idx, L_short:].eq(0).all()
        assert pair[shorter_idx, :, L_short:].eq(0).all()

    def test_frame_idx_preserved(self):
        batch = [self._make_sample(L_A), self._make_sample(L_A)]
        _, _, fidx, _ = collate_fn(batch)
        assert fidx.tolist() == [42, 42]

    def test_dtypes(self):
        batch = [self._make_sample(L_A)]
        (z0, z1), pair, fidx, mask = collate_fn(batch)
        assert z0.dtype   == torch.float32
        assert pair.dtype == torch.float32
        assert fidx.dtype == torch.int64
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Integration: dataset → collate_fn round-trip
# ---------------------------------------------------------------------------

class TestDatasetCollateFnIntegration:
    def test_mixed_proteins_batch(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        make_protein_dir(tmp_path, "protB", N_FRAMES_B, L_B)
        ds = build_multi_protein_dataset(tmp_path)

        # Manually pull one sample from each protein and collate
        sample_a = ds[0]   # protA (L=10)
        sample_b = ds[N_FRAMES_A]   # first protB sample (L=15)
        (z0, z1), pair, fidx, mask = collate_fn([sample_a, sample_b])

        L_max = max(L_A, L_B)
        assert z0.shape   == (2, L_max, D_S)
        assert pair.shape == (2, L_max, L_max, D_P)
        # protA row should be padded in positions L_A:L_max
        assert mask[0, L_A:].all()
        assert not mask[1].any()   # protB is the longer one
