"""
data_pipeline/dataset.py

Worker-safe, memory-efficient PyTorch Dataset over HDF5 latent shards produced
by extract_latents.py.  Opens each shard file per __getitem__ call so that file
handles are never shared across DataLoader workers (h5py handles are not fork-safe).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import h5py
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Streams (single, pair, frame_idx) tuples from HDF5 shards.

    Directory layout expected on disk:
        <data_dir>/<protein_name>/
            manifest.json
            pair.pt             ← static pair tensor [L, L, d_pair] float16
            shard_000000.h5
            shard_000001.h5
            ...

    Each shard HDF5 file contains:
        single    float16  [N, L, d_single]  — Chai-1 single-token reps
        frame_idx int64    [N]               — original MD frame index

    pair.pt is produced by extract_pairs.py (one tensor per protein, frame 0).
    It is loaded once into CPU RAM in __init__ and shared across all workers
    via copy-on-write fork semantics — no per-item IO.

    Args:
        data_dir:     Root scratch directory containing per-protein subdirs.
        protein_name: Subdirectory name (e.g. "16pk_A").
        load_pair:    If False, returns None for pair (saves IO when only single
                      is needed, e.g. during fast single-rep ablations).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        protein_name: str,
        load_pair: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.protein_name = protein_name
        self.load_pair = load_pair

        protein_dir = self.data_dir / protein_name
        manifest_path = protein_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.shards: list[dict] = manifest["shards"]

        # Flat index: list of (shard_path, local_frame_index)
        # Built once in __init__; workers read from it without re-loading json.
        self.index_map: list[tuple[Path, int]] = []
        for shard_info in self.shards:
            shard_path = protein_dir / shard_info["path"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard referenced in manifest missing: {shard_path}")
            for local_idx in range(shard_info["n_frames"]):
                self.index_map.append((shard_path, local_idx))

        # Static pair tensor — loaded once, shared across all workers via
        # copy-on-write fork semantics. Zero per-item IO cost.
        pair_path = protein_dir / "pair.pt"
        if load_pair and not pair_path.exists():
            raise FileNotFoundError(
                f"pair.pt not found at {pair_path}. "
                "Run extract_pairs.slurm first, or pass load_pair=False."
            )
        self.static_pair: torch.Tensor | None = (
            torch.load(pair_path, weights_only=True).float() if load_pair else None
        )

        # Populated lazily inside __getitem__, after DataLoader workers are
        # forked.  h5py.File objects cannot be pickled, so they must never be
        # created before the fork boundary.
        self.file_handles: dict[str, h5py.File] = {}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None, int]:
        """
        Returns consecutive frame pair for flow matching.

        Returns:
            (single_0, single_1): float32 [L, d_single] each — frames i and i+1
            pair:      float32 [L, L, d_pair] from frame i, or None if load_pair=False
            frame_idx: int — original MD trajectory frame index of z0
        """
        idx1 = min(idx + 1, len(self) - 1)

        shard_path_0, local_idx_0 = self.index_map[idx]
        shard_path_1, local_idx_1 = self.index_map[idx1]

        def _get_handle(shard_path: Path) -> h5py.File:
            key = str(shard_path)
            if key not in self.file_handles:
                self.file_handles[key] = h5py.File(key, "r")
            return self.file_handles[key]

        f0 = _get_handle(shard_path_0)
        f1 = _get_handle(shard_path_1)

        single_0  = torch.from_numpy(f0["single"][local_idx_0]).float()
        single_1  = torch.from_numpy(f1["single"][local_idx_1]).float()
        frame_idx = int(f0["frame_idx"][local_idx_0])

        # Static pair: already in RAM, return directly (no HDF5 read)
        return (single_0, single_1), self.static_pair, frame_idx

    def __del__(self) -> None:
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LatentDataset(protein={self.protein_name!r}, "
            f"frames={len(self)}, shards={len(self.shards)}, "
            f"load_pair={self.load_pair})"
        )
