"""
data_pipeline/dataset.py

Worker-safe, memory-efficient PyTorch Dataset over HDF5 latent shards produced
by extract_latents.py.  Opens each shard file per __getitem__ call so that file
handles are never shared across DataLoader workers (h5py handles are not fork-safe).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset


_REPLICA_RE   = re.compile(r"_R(\d+)_")
_SHARD_NUM_RE = re.compile(r"shard_(\d+)\.h5$")


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

    Replica-safe frame pairing
    ---------------------------
    ATLAS proteins have multiple independent MD replicas, extracted by
    concurrent SLURM tasks that each append their own shards to one shared
    manifest.json (guarded only by a file lock, not by write order). So shard
    entries in the manifest are interleaved by whichever job finished a shard
    first — not grouped by replica and not necessarily in shard-index order.
    Shard filenames encode the replica as "<protein>_R<n>_shard_NNNNNN.h5";
    this class groups shards by that tag and only pairs consecutive frames
    within the same replica group, so z0/z1 training pairs never splice
    together two unrelated trajectories. Shards without a replica tag (e.g.
    single-trajectory fixtures) fall into one implicit group.

    Args:
        data_dir:     Root scratch directory containing per-protein subdirs.
        protein_name: Subdirectory name (e.g. "16pk_A").
        load_pair:    If False, returns None for pair (saves IO when only single
                      is needed, e.g. during fast single-rep ablations).
        norm_stats:   Optional {"mean": [d_single], "std": [d_single]} (as
                      produced by data_pipeline/compute_norm_stats.py). If
                      given, z0/z1 are returned as (x - mean) / std instead of
                      raw Chai-1 latent scale.
        lag:          Frame offset between z0 and z1 (default 1 = adjacent
                      frames, i.e. one native ATLAS save interval, 10ps).
                      z1 is drawn from index i+lag within the SAME replica
                      group as z0. At native 10ps/frame stride, single-step
                      (lag=1) displacement was found to carry ~0% explainable
                      signal beyond thermal noise (position-mean baseline,
                      2026-07-03 pilot) — this parameter exists to let
                      training and diagnostics target a longer, more dynamics-
                      relevant timescale without re-extracting any data, since
                      every frame was already extracted independently.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        protein_name: str,
        load_pair: bool = True,
        norm_stats: dict[str, torch.Tensor] | None = None,
        lag: int = 1,
    ) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.data_dir = Path(data_dir)
        self.protein_name = protein_name
        self.load_pair = load_pair
        self.lag = lag
        self.norm_mean = norm_stats["mean"].float() if norm_stats is not None else None
        self.norm_std  = norm_stats["std"].float()  if norm_stats is not None else None

        protein_dir = self.data_dir / protein_name
        manifest_path = protein_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Deduplicate by path — concurrent replica jobs can write the same shard
        # filename and produce duplicate manifest entries for identical files.
        seen: set[str] = set()
        self.shards: list[dict] = [
            s for s in manifest[protein_name]["shards"]
            if not (s["path"] in seen or seen.add(s["path"]))  # type: ignore[func-returns-value]
        ]

        # Group shards by replica (see class docstring), then order each
        # group by shard index — never trust raw manifest append order.
        groups: dict[str, list[dict]] = {}
        for s in self.shards:
            name = Path(s["path"]).name
            m = _REPLICA_RE.search(name)
            groups.setdefault(m.group(1) if m else "", []).append(s)

        def _shard_num(shard_info: dict) -> int:
            m = _SHARD_NUM_RE.search(Path(shard_info["path"]).name)
            return int(m.group(1)) if m else 0

        # index_map: flat (shard_path, local_idx) for every frame, laid out
        # replica group by replica group, each group internally ordered by
        # shard index. _next_idx[i] gives the index_map position to use as
        # the z1 partner for position i — normally i+lag, but within `lag`
        # frames of the end of each replica group it wraps back to i itself
        # (zero target velocity) instead of reaching into the next, unrelated
        # group.
        self.index_map: list[tuple[Path, int]] = []
        self._next_idx: list[int] = []
        for rep_key in sorted(groups):
            rep_shards = sorted(groups[rep_key], key=_shard_num)
            group_start = len(self.index_map)
            for shard_info in rep_shards:
                shard_path = protein_dir / shard_info["path"]
                if not shard_path.exists():
                    raise FileNotFoundError(f"Shard referenced in manifest missing: {shard_path}")
                for local_idx in range(shard_info["n_frames"]):
                    self.index_map.append((shard_path, local_idx))
            group_end = len(self.index_map)
            for i in range(group_start, group_end):
                self._next_idx.append(i + lag if i + lag < group_end else i)

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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is None:
            return x
        return (x - self.norm_mean) / self.norm_std

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None, int]:
        """
        Returns a lagged frame pair for flow matching.

        Returns:
            (single_0, single_1): float32 [L, d_single] each — frames i and
                                   i+self.lag (normalized if norm_stats given)
            pair:      float32 [L, L, d_pair] from frame i, or None if load_pair=False
            frame_idx: int — original MD trajectory frame index of z0
        """
        idx1 = self._next_idx[idx]

        shard_path_0, local_idx_0 = self.index_map[idx]
        shard_path_1, local_idx_1 = self.index_map[idx1]

        def _get_handle(shard_path: Path) -> h5py.File:
            key = str(shard_path)
            if key not in self.file_handles:
                self.file_handles[key] = h5py.File(key, "r")
            return self.file_handles[key]

        f0 = _get_handle(shard_path_0)
        f1 = _get_handle(shard_path_1)

        single_0  = self._normalize(torch.from_numpy(f0["single"][local_idx_0]).float())
        single_1  = self._normalize(torch.from_numpy(f1["single"][local_idx_1]).float())
        frame_idx = int(f0["frame_idx"][local_idx_0])

        # Static pair: already in RAM, return directly (no HDF5 read)
        return (single_0, single_1), self.static_pair, frame_idx

    def __del__(self) -> None:
        for f in getattr(self, "file_handles", {}).values():
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
            f"load_pair={self.load_pair}, lag={self.lag})"
        )


# ---------------------------------------------------------------------------
# Multi-protein helpers
# ---------------------------------------------------------------------------

def build_multi_protein_dataset(
    latents_dir: Union[str, Path],
    protein_ids: list[str] | None = None,
    norm_stats_path: Union[str, Path, None] = None,
    lag: int = 1,
) -> ConcatDataset:
    """
    Discover all proteins in `latents_dir` that have both manifest.json and
    pair.pt, build a LatentDataset for each, and return a ConcatDataset.

    protein_ids: optional allowlist — if provided, only include those proteins.
    Proteins missing either file are skipped with a warning so a partially
    finished extraction run doesn't block training.

    norm_stats_path: optional explicit path to a norm_stats.pt file (as
    produced by data_pipeline/compute_norm_stats.py) with per-channel "mean"
    and "std" for the single-token latents. If not given, auto-detects
    "<latents_dir>/norm_stats.pt"; if that also doesn't exist, proceeds
    WITHOUT normalization (a warning is printed). Callers that require
    normalized data (e.g. training/train.py) must verify the file exists
    themselves before calling this, so training never silently runs
    unnormalized.

    lag: frame offset between z0/z1 for every protein — see LatentDataset.
    """
    latents_dir = Path(latents_dir)

    if norm_stats_path is None:
        norm_stats_path = latents_dir / "norm_stats.pt"
    norm_stats_path = Path(norm_stats_path)
    norm_stats: dict[str, torch.Tensor] | None = None
    if norm_stats_path.exists():
        norm_stats = torch.load(norm_stats_path, weights_only=True)
        print(f"[dataset] Loaded normalization stats from {norm_stats_path}", flush=True)
    else:
        print(f"[dataset] No normalization stats found at {norm_stats_path} — "
              f"proceeding WITHOUT latent normalization.", flush=True)

    datasets = []
    skipped = []
    for protein_dir in sorted(latents_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        if protein_ids is not None and protein_dir.name not in protein_ids:
            continue
        manifest = protein_dir / "manifest.json"
        pair     = protein_dir / "pair.pt"
        if not manifest.exists() or not pair.exists():
            skipped.append(protein_dir.name)
            continue
        datasets.append(LatentDataset(
            latents_dir, protein_dir.name, load_pair=True, norm_stats=norm_stats, lag=lag,
        ))

    if skipped:
        print(f"[dataset] Skipped {len(skipped)} proteins (missing manifest or pair.pt): "
              f"{skipped}", flush=True)
    if not datasets:
        raise RuntimeError(f"No valid protein datasets found in {latents_dir}")

    total_frames = sum(len(d) for d in datasets)
    print(f"[dataset] {len(datasets)} proteins, {total_frames} total frames", flush=True)
    return ConcatDataset(datasets)


def collate_fn(
    batch: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a mixed-length batch to the longest sequence in it.

    Each sample is ((z0 [L_i, d_s], z1 [L_i, d_s]), pair [L_i, L_i, d_p], frame_idx).
    Proteins with different sequence lengths L_i are zero-padded to L_max.

    Returns:
        (z0, z1)      float32 [B, L_max, d_s]
        pair          float32 [B, L_max, L_max, d_p]
        frame_idx     int64   [B]
        padding_mask  bool    [B, L_max]  — True = padded position (ignore in loss/attn)
    """
    L_max = max(item[0][0].shape[0] for item in batch)

    z0_list, z1_list, pair_list, mask_list, fidx_list = [], [], [], [], []
    for (z0, z1), pair, frame_idx in batch:
        L = z0.shape[0]
        pad = L_max - L

        z0_list.append(F.pad(z0,   (0, 0, 0, pad)))           # [L_max, d_s]
        z1_list.append(F.pad(z1,   (0, 0, 0, pad)))
        pair_list.append(F.pad(pair, (0, 0, 0, pad, 0, pad)))  # [L_max, L_max, d_p]

        mask = torch.zeros(L_max, dtype=torch.bool)
        mask[L:] = True
        mask_list.append(mask)
        fidx_list.append(frame_idx)

    return (
        (torch.stack(z0_list), torch.stack(z1_list)),
        torch.stack(pair_list),
        torch.tensor(fidx_list, dtype=torch.int64),
        torch.stack(mask_list),
    )
