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


def _write_shard_valued(
    protein_dir: Path,
    shard_name: str,
    n_frames: int,
    L: int,
    d_s: int,
    start_frame: int,
    value_offset: float,
) -> Path:
    """Writes a shard whose frames are filled with distinctive constants
    (value_offset + local frame position) so cross-shard pairing can be
    verified directly from tensor values, not just frame_idx bookkeeping."""
    path = protein_dir / shard_name
    with h5py.File(path, "w") as f:
        singles = np.stack([
            np.full((L, d_s), value_offset + i, dtype=np.float16)
            for i in range(n_frames)
        ])
        frames = np.arange(start_frame, start_frame + n_frames, dtype=np.int32)
        f.create_dataset("single", data=singles)
        f.create_dataset("frame_idx", data=frames)
    return path


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
# Replica-aware pairing — the concurrent-extraction interleaving bug
# ---------------------------------------------------------------------------

class TestReplicaAwarePairing:
    def test_no_cross_replica_pairing(self, tmp_path):
        """
        Replica-extraction workers run as concurrent SLURM tasks and append
        to one shared manifest.json, so shard entries can land interleaved
        by completion time rather than grouped by replica. z0/z1 pairs must
        never span two different replicas' shards.
        """
        protein = "protX"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)

        L, d_s = 6, 4
        n_r1, n_r2 = 5, 4

        # Replica 1 values in [0, n_r1); replica 2 values in [1000, 1000+n_r2)
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", n_r1, L, d_s, 0, 0.0)
        _write_shard_valued(shard_dir, f"{protein}_R2_shard_000000.h5", n_r2, L, d_s, 0, 1000.0)

        # Manifest lists replica 2's shard BEFORE replica 1's — simulates the
        # concurrent-write race where whichever job finishes first appends
        # first, regardless of replica number.
        shards = [
            {"path": f"{protein}/{protein}_R2_shard_000000.h5", "n_frames": n_r2,
             "first_idx": 0, "last_idx": n_r2 - 1},
            {"path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": n_r1,
             "first_idx": 0, "last_idx": n_r1 - 1},
        ]
        _write_manifest(protein_dir, protein, shards)
        _write_pair(protein_dir, L)

        ds = LatentDataset(tmp_path, protein)
        assert len(ds) == n_r1 + n_r2

        for i in range(len(ds)):
            (z0, z1), _, _ = ds[i]
            z0_replica = "R1" if z0[0, 0].item() < 1000 else "R2"
            z1_replica = "R1" if z1[0, 0].item() < 1000 else "R2"
            assert z0_replica == z1_replica, (
                f"index {i}: z0 is from {z0_replica} but z1 is from {z1_replica} "
                "— pairing crossed a replica boundary"
            )

    def test_last_frame_of_each_replica_self_pairs(self, tmp_path):
        protein = "protY"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)
        L, d_s = 6, 4
        n_r1, n_r2 = 3, 3
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", n_r1, L, d_s, 0, 0.0)
        _write_shard_valued(shard_dir, f"{protein}_R2_shard_000000.h5", n_r2, L, d_s, 0, 1000.0)
        shards = [
            {"path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": n_r1,
             "first_idx": 0, "last_idx": n_r1 - 1},
            {"path": f"{protein}/{protein}_R2_shard_000000.h5", "n_frames": n_r2,
             "first_idx": 0, "last_idx": n_r2 - 1},
        ]
        _write_manifest(protein_dir, protein, shards)
        _write_pair(protein_dir, L)

        ds = LatentDataset(tmp_path, protein)

        (z0, z1), _, _ = ds[n_r1 - 1]  # last frame of replica 1
        assert torch.allclose(z0, z1)

        last = len(ds) - 1  # last frame of replica 2
        (z0, z1), _, _ = ds[last]
        assert torch.allclose(z0, z1)

    def test_shard_order_within_replica_uses_shard_index_not_manifest_order(self, tmp_path):
        """Manifest can also list a replica's own shards out of numeric order
        (a resumed job appending after an earlier one). Must still sort by
        the numeric shard index in the filename, not manifest append order."""
        protein = "protZ"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)
        L, d_s = 4, 3

        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", 3, L, d_s, 0, 0.0)
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000001.h5", 3, L, d_s, 3, 100.0)

        # Manifest lists shard_000001 before shard_000000.
        shards = [
            {"path": f"{protein}/{protein}_R1_shard_000001.h5", "n_frames": 3,
             "first_idx": 3, "last_idx": 5},
            {"path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": 3,
             "first_idx": 0, "last_idx": 2},
        ]
        _write_manifest(protein_dir, protein, shards)
        _write_pair(protein_dir, L)

        ds = LatentDataset(tmp_path, protein)
        # Frame 2 (value 2.0, last of shard 0) must pair with frame 3
        # (value 100.0, first of shard 1) — the true trajectory continuation.
        (z0, z1), _, _ = ds[2]
        assert z0[0, 0].item() == pytest.approx(2.0)
        assert z1[0, 0].item() == pytest.approx(100.0)


class TestLagParameter:
    """Configurable frame lag — see LatentDataset's `lag` docstring. Lets
    training/diagnostics target a longer timescale than the native ATLAS
    save interval without re-extracting data."""

    def test_default_lag_is_one_and_unchanged(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        assert ds.lag == 1

    def test_lag_pairs_frame_i_with_i_plus_lag(self, tmp_path):
        protein = "protLag"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)
        L, d_s, n = 4, 3, 20
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", n, L, d_s, 0, 0.0)
        _write_manifest(protein_dir, protein, [{
            "path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": n,
            "first_idx": 0, "last_idx": n - 1,
        }])
        _write_pair(protein_dir, L)

        lag = 5
        ds = LatentDataset(tmp_path, protein, lag=lag)
        assert ds.lag == lag
        for i in range(n - lag):
            (z0, z1), _, _ = ds[i]
            assert z0[0, 0].item() == pytest.approx(float(i))
            assert z1[0, 0].item() == pytest.approx(float(i + lag))

    def test_frames_within_lag_of_group_end_self_pair(self, tmp_path):
        """The last `lag` frames of a replica have no valid i+lag partner
        within the group and must self-pair (zero target), not reach past
        the group end."""
        protein = "protLagEnd"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)
        L, d_s, n = 4, 3, 10
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", n, L, d_s, 0, 0.0)
        _write_manifest(protein_dir, protein, [{
            "path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": n,
            "first_idx": 0, "last_idx": n - 1,
        }])
        _write_pair(protein_dir, L)

        lag = 3
        ds = LatentDataset(tmp_path, protein, lag=lag)
        for i in range(n - lag, n):  # last `lag` frames
            (z0, z1), _, _ = ds[i]
            assert torch.allclose(z0, z1), f"index {i} should self-pair (within {lag} of group end)"

    def test_lag_never_crosses_replica_boundary(self, tmp_path):
        protein = "protLagReplica"
        protein_dir = tmp_path / protein
        shard_dir = protein_dir / protein
        shard_dir.mkdir(parents=True)
        L, d_s = 4, 3
        n_r1, n_r2 = 10, 10
        _write_shard_valued(shard_dir, f"{protein}_R1_shard_000000.h5", n_r1, L, d_s, 0, 0.0)
        _write_shard_valued(shard_dir, f"{protein}_R2_shard_000000.h5", n_r2, L, d_s, 0, 1000.0)
        _write_manifest(protein_dir, protein, [
            {"path": f"{protein}/{protein}_R1_shard_000000.h5", "n_frames": n_r1,
             "first_idx": 0, "last_idx": n_r1 - 1},
            {"path": f"{protein}/{protein}_R2_shard_000000.h5", "n_frames": n_r2,
             "first_idx": 0, "last_idx": n_r2 - 1},
        ])
        _write_pair(protein_dir, L)

        lag = 5
        ds = LatentDataset(tmp_path, protein, lag=lag)
        for i in range(len(ds)):
            (z0, z1), _, _ = ds[i]
            z0_replica = "R1" if z0[0, 0].item() < 1000 else "R2"
            z1_replica = "R1" if z1[0, 0].item() < 1000 else "R2"
            assert z0_replica == z1_replica, f"index {i} crossed a replica boundary with lag={lag}"

    def test_lag_less_than_one_raises(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        with pytest.raises(ValueError, match="lag must be >= 1"):
            LatentDataset(tmp_path, "protA", lag=0)

    def test_repr_includes_lag(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA", lag=7)
        assert "lag=7" in repr(ds)

    def test_build_multi_protein_dataset_propagates_lag(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = build_multi_protein_dataset(tmp_path, lag=3)
        underlying = ds.datasets[0]
        assert underlying.lag == 3


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_no_normalization_by_default(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = LatentDataset(tmp_path, "protA")
        assert ds.norm_mean is None
        assert ds.norm_std is None

    def test_normalization_applied(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        mean = torch.full((D_S,), 2.0)
        std  = torch.full((D_S,), 3.0)
        ds_raw  = LatentDataset(tmp_path, "protA")
        ds_norm = LatentDataset(tmp_path, "protA", norm_stats={"mean": mean, "std": std})
        (z0_raw, z1_raw),   _, _ = ds_raw[0]
        (z0_norm, z1_norm), _, _ = ds_norm[0]
        assert torch.allclose(z0_norm, (z0_raw - mean) / std, atol=1e-5)
        assert torch.allclose(z1_norm, (z1_raw - mean) / std, atol=1e-5)

    def test_normalized_output_has_expected_scale(self, tmp_path):
        """Normalizing with the true per-channel mean/std of a large constant
        field should collapse it to ~0."""
        protein_dir = tmp_path / "protC"
        shard_dir = protein_dir / "protC"
        shard_dir.mkdir(parents=True)
        L, d_s, n = 5, 4, 6
        with h5py.File(shard_dir / "shard_000000.h5", "w") as f:
            f.create_dataset("single", data=np.full((n, L, d_s), 10.0, dtype=np.float16))
            f.create_dataset("frame_idx", data=np.arange(n, dtype=np.int32))
        _write_manifest(protein_dir, "protC", [{
            "path": "protC/shard_000000.h5", "n_frames": n, "first_idx": 0, "last_idx": n - 1,
        }])
        _write_pair(protein_dir, L)

        norm_stats = {"mean": torch.full((d_s,), 10.0), "std": torch.full((d_s,), 1.0)}
        ds = LatentDataset(tmp_path, "protC", norm_stats=norm_stats)
        (z0, _), _, _ = ds[0]
        assert torch.allclose(z0, torch.zeros_like(z0), atol=1e-2)


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

    def test_auto_detects_norm_stats(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        mean = torch.zeros(D_S)
        std  = torch.full((D_S,), 2.0)
        torch.save({"mean": mean, "std": std}, tmp_path / "norm_stats.pt")

        ds = build_multi_protein_dataset(tmp_path)
        underlying = ds.datasets[0]
        assert underlying.norm_mean is not None
        assert torch.allclose(underlying.norm_std, std)

    def test_no_norm_stats_proceeds_unnormalized(self, tmp_path):
        make_protein_dir(tmp_path, "protA", N_FRAMES_A, L_A)
        ds = build_multi_protein_dataset(tmp_path)
        underlying = ds.datasets[0]
        assert underlying.norm_mean is None


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
