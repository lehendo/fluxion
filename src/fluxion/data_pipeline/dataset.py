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
        velocity_lag: If > 0, __getitem__ additionally returns z_prev — the
                      frame `velocity_lag` steps BEFORE z0, within the same
                      replica group — as the first element of the returned
                      tuple, giving (z_prev, z0, z1) instead of (z0, z1).
                      Lets a model or diagnostic condition on an explicit
                      finite-difference velocity z0 - z_prev in addition to
                      the current state, rather than z0 alone. Default 0
                      disables this (identical return shape to before this
                      parameter existed). Mirrors `lag`'s tail self-pairing:
                      at the START of a replica, where no real previous frame
                      exists, z_prev self-pairs to z0 (velocity = 0) rather
                      than reaching into the previous, unrelated group.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        protein_name: str,
        load_pair: bool = True,
        norm_stats: dict[str, torch.Tensor] | None = None,
        lag: int = 1,
        velocity_lag: int = 0,
    ) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        if velocity_lag < 0:
            raise ValueError(f"velocity_lag must be >= 0, got {velocity_lag}")
        self.data_dir = Path(data_dir)
        self.protein_name = protein_name
        self.load_pair = load_pair
        self.lag = lag
        self.velocity_lag = velocity_lag
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
        # group. _prev_idx[i] is the symmetric lookup for velocity_lag: the
        # position i-velocity_lag, wrapping to i itself within velocity_lag
        # frames of the START of each group.
        self.index_map: list[tuple[Path, int]] = []
        self._next_idx: list[int] = []
        self._prev_idx: list[int] = []
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
                self._prev_idx.append(i - velocity_lag if i - velocity_lag >= group_start else i)

        # Static pair tensor — loaded once, shared across all workers via
        # copy-on-write fork semantics. Zero per-item IO cost.
        pair_path = protein_dir / "pair.pt"
        if load_pair and not pair_path.exists():
            raise FileNotFoundError(
                f"pair.pt not found at {pair_path}. "
                "Run slurm/extraction/extract_pairs.slurm first, or pass load_pair=False."
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

    def _get_handle(self, shard_path: Path) -> h5py.File:
        key = str(shard_path)
        if key not in self.file_handles:
            self.file_handles[key] = h5py.File(key, "r")
        return self.file_handles[key]

    def _load_frame(self, flat_idx: int) -> torch.Tensor:
        """Normalized single-token tensor for a flat index_map position."""
        shard_path, local_idx = self.index_map[flat_idx]
        f = self._get_handle(shard_path)
        return self._normalize(torch.from_numpy(f["single"][local_idx]).float())

    def __getitem__(self, idx: int):
        """
        Returns a lagged frame pair for flow matching (or a velocity-augmented
        triplet if velocity_lag > 0 — see class docstring).

        Returns (velocity_lag == 0, default):
            (single_0, single_1): float32 [L, d_single] each — frames i and
                                   i+self.lag (normalized if norm_stats given)
            pair:      float32 [L, L, d_pair] from frame i, or None if load_pair=False
            frame_idx: int — original MD trajectory frame index of z0

        Returns (velocity_lag > 0):
            (single_prev, single_0, single_1): as above, plus the frame
                                   i-self.velocity_lag
            pair, frame_idx: as above
        """
        idx1 = self._next_idx[idx]

        shard_path_0, local_idx_0 = self.index_map[idx]
        f0 = self._get_handle(shard_path_0)

        single_0  = self._load_frame(idx)
        single_1  = self._load_frame(idx1)
        frame_idx = int(f0["frame_idx"][local_idx_0])

        if self.velocity_lag > 0:
            single_prev = self._load_frame(self._prev_idx[idx])
            return (single_prev, single_0, single_1), self.static_pair, frame_idx

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
            f"load_pair={self.load_pair}, lag={self.lag}, "
            f"velocity_lag={self.velocity_lag})"
        )


class StaticFrameDataset(Dataset):
    """
    Yields INDIVIDUAL frames — (single, pair, frame_idx), no pairing — for
    noise-to-ensemble generation training (a formulation the paper does not use).

    Every MD frame across every replica is treated as an i.i.d. sample from
    the protein's equilibrium conformational distribution (the task is
    "learn to generate a plausible conformation," not "predict the next
    frame"), so unlike LatentDataset there is no lag/velocity_lag/replica-
    pairing concern at all — replica boundaries don't matter when frames
    are never paired with each other. The "noise" half of the OT-CFM
    training pair (z0) is generated fresh at training time ), not read from disk.

    Wraps a LatentDataset by composition to reuse its already-tested shard-
    reading, replica-grouping, deduplication, and normalization logic
    without duplicating any of it — only consumes its index_map and
    _load_frame, ignoring the pairing machinery entirely.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        protein_name: str,
        load_pair: bool = True,
        norm_stats: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._inner = LatentDataset(
            data_dir, protein_name, load_pair=load_pair, norm_stats=norm_stats,
        )
        self.protein_name = protein_name
        self.static_pair = self._inner.static_pair

    def __len__(self) -> int:
        return len(self._inner.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Returns:
            single:    float32 [L, d_single] — a single real conformational
                       sample (normalized if norm_stats given)
            pair:      float32 [L, L, d_pair], or None if load_pair=False
            frame_idx: int — original MD trajectory frame index
        """
        single = self._inner._load_frame(idx)
        shard_path, local_idx = self._inner.index_map[idx]
        f = self._inner._get_handle(shard_path)
        frame_idx = int(f["frame_idx"][local_idx])
        return single, self.static_pair, frame_idx

    def __repr__(self) -> str:
        return f"StaticFrameDataset(protein={self.protein_name!r}, frames={len(self)})"


# ---------------------------------------------------------------------------
# Multi-protein helpers
# ---------------------------------------------------------------------------

def build_multi_protein_dataset(
    latents_dir: Union[str, Path],
    protein_ids: list[str] | None = None,
    norm_stats_path: Union[str, Path, None] = None,
    lag: int = 1,
    velocity_lag: int = 0,
    load_pair: bool = True,
) -> ConcatDataset:
    """
    Discover all proteins in `latents_dir` that have manifest.json (and, if
    load_pair, pair.pt too), build a LatentDataset for each, and return a
    ConcatDataset.

    protein_ids: optional allowlist — if provided, only include those proteins.
    Proteins missing either required file are skipped with a warning so a
    partially finished extraction run doesn't block training.

    load_pair: defaults to True (existing behavior — training and checkpoint
    evaluation need the real pair conditioning). Pass False for callers that
    only need single-token latents (e.g. compute_trivial_baseline.py), so
    they work against extraction runs written with --skip-pair.

    norm_stats_path: optional explicit path to a norm_stats.pt file (as
    produced by data_pipeline/compute_norm_stats.py) with per-channel "mean"
    and "std" for the single-token latents. If not given, auto-detects
    "<latents_dir>/norm_stats.pt"; if that also doesn't exist, proceeds
    WITHOUT normalization (a warning is printed). Callers that require
    normalized data (e.g. training/train.py) must verify the file exists
    themselves before calling this, so training never silently runs
    unnormalized.

    lag: frame offset between z0/z1 for every protein — see LatentDataset.
    velocity_lag: optional explicit-velocity conditioning — see LatentDataset.
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
        if not manifest.exists() or (load_pair and not pair.exists()):
            skipped.append(protein_dir.name)
            continue
        datasets.append(LatentDataset(
            latents_dir, protein_dir.name, load_pair=load_pair, norm_stats=norm_stats,
            lag=lag, velocity_lag=velocity_lag,
        ))

    if skipped:
        reason = "missing manifest or pair.pt" if load_pair else "missing manifest"
        print(f"[dataset] Skipped {len(skipped)} proteins ({reason}): "
              f"{skipped}", flush=True)
    if not datasets:
        raise RuntimeError(f"No valid protein datasets found in {latents_dir}")

    total_frames = sum(len(d) for d in datasets)
    print(f"[dataset] {len(datasets)} proteins, {total_frames} total frames", flush=True)
    return ConcatDataset(datasets)


def build_multi_protein_static_dataset(
    latents_dir: Union[str, Path],
    protein_ids: list[str] | None = None,
    norm_stats_path: Union[str, Path, None] = None,
) -> ConcatDataset:
    """
    Same discovery/normalization behavior as build_multi_protein_dataset, but
    builds StaticFrameDataset instances (individual frames, no pairing) for
    noise-to-ensemble generation training instead of LatentDataset instances (a formulation the paper does not use).
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
        datasets.append(StaticFrameDataset(
            latents_dir, protein_dir.name, load_pair=True, norm_stats=norm_stats,
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
    pad_to: int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a mixed-length batch to the longest sequence in it (or to a fixed
    length — see pad_to below).

    Each sample is ((z0 [L_i, d_s], z1 [L_i, d_s]), pair [L_i, L_i, d_p], frame_idx).
    Proteins with different sequence lengths L_i are zero-padded to L_max.

    pad_to: if given, every batch pads to this fixed length instead of the
    per-batch max. Default (None) preserves the original per-batch-dynamic
    behavior. Use a fixed pad_to (e.g. the largest Chai-1 crop bucket, 768)
    for long-running DataLoaders with num_workers>0 and pin_memory=True —
    PyTorch's pinned-memory allocator caches a separate block per distinct
    tensor SIZE it sees and never releases them, so a per-batch-varying
    L_max (this dataset spans 256/384/512/768 depending on which proteins
    get sampled) causes host RAM to grow unboundedly over training as new
    sizes keep appearing. Confirmed 2026-08-27: a training run OOM'd via a
    Linux cgroup oom_kill (host RAM, not CUDA) at the identical step twice
    in a row regardless of MALLOC_ARENA_MAX (which only affects glibc
    malloc, not this allocator) -- a fixed pad_to eliminates the variable-
    size pattern that triggers it, at the cost of always paying the max
    bucket's memory/compute (worth it for training stability; a run-once
    eval script under different memory constraints might prefer the
    original dynamic behavior instead).

    Returns:
        (z0, z1)      float32 [B, L_max, d_s]
        pair          float32 [B, L_max, L_max, d_p]
        frame_idx     int64   [B]
        padding_mask  bool    [B, L_max]  — True = padded position (ignore in loss/attn)
    """
    L_max = pad_to if pad_to is not None else max(item[0][0].shape[0] for item in batch)

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


def collate_fn_no_pair(
    batch: list[tuple[tuple[torch.Tensor, torch.Tensor], None, int]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], None, torch.Tensor, torch.Tensor]:
    """
    Same padding behavior as collate_fn, for a multi-protein (mixed-length)
    batch, but for datasets built with load_pair=False — pair is None per
    sample, so it's never padded/stacked (saves the per-batch GB-scale pair
    tensor entirely for callers, like compute_trivial_baseline.py, that
    never touch it and may be run against extractions with no pair.pt at
    all, e.g. written with --skip-pair).

    Returns the same 4-tuple shape as collate_fn, with pair always None.
    """
    L_max = max(item[0][0].shape[0] for item in batch)

    z0_list, z1_list, mask_list, fidx_list = [], [], [], []
    for (z0, z1), _pair, frame_idx in batch:
        L = z0.shape[0]
        pad = L_max - L

        z0_list.append(F.pad(z0, (0, 0, 0, pad)))
        z1_list.append(F.pad(z1, (0, 0, 0, pad)))

        mask = torch.zeros(L_max, dtype=torch.bool)
        mask[L:] = True
        mask_list.append(mask)
        fidx_list.append(frame_idx)

    return (
        (torch.stack(z0_list), torch.stack(z1_list)),
        None,
        torch.tensor(fidx_list, dtype=torch.int64),
        torch.stack(mask_list),
    )


def collate_fn_static(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a mixed-length batch of individual frames to the longest sequence in it.

    Each sample is (single [L_i, d_s], pair [L_i, L_i, d_p], frame_idx) — see
    StaticFrameDataset. Unlike collate_fn there is only one latent tensor per
    sample (no z0/z1 pairing); z0 is generated fresh as noise at train time.

    Returns:
        single        float32 [B, L_max, d_s]
        pair          float32 [B, L_max, L_max, d_p]
        frame_idx     int64   [B]
        padding_mask  bool    [B, L_max]  — True = padded position (ignore in loss/attn)
    """
    L_max = max(item[0].shape[0] for item in batch)

    single_list, pair_list, mask_list, fidx_list = [], [], [], []
    for single, pair, frame_idx in batch:
        L = single.shape[0]
        pad = L_max - L

        single_list.append(F.pad(single, (0, 0, 0, pad)))      # [L_max, d_s]
        pair_list.append(F.pad(pair, (0, 0, 0, pad, 0, pad)))  # [L_max, L_max, d_p]

        mask = torch.zeros(L_max, dtype=torch.bool)
        mask[L:] = True
        mask_list.append(mask)
        fidx_list.append(frame_idx)

    return (
        torch.stack(single_list),
        torch.stack(pair_list),
        torch.tensor(fidx_list, dtype=torch.int64),
        torch.stack(mask_list),
    )
