#!/usr/bin/env python3
"""
data_pipeline/extract_latents.py

Swarm-style latent extraction: stream ATLAS MD trajectory frames through the
frozen Chai-1 trunk and write per-frame single/pair representations to HDF5
shards.  One instance per GPU; multiple instances update a shared manifest
safely via filelock.

Usage
─────
python data_pipeline/extract_latents.py \\
    --pdb  /scratch/arjunc4/fluxion_data/raw/1ABC/protein.pdb \\
    --dcd  /scratch/arjunc4/fluxion_data/raw/1ABC/traj.dcd \\
    --out-dir /scratch/arjunc4/fluxion_data/latents \\
    --shard-size 1000 \\
    --device cuda

For large proteins (L > 400) the pair buffer can exceed 10 GB RAM per shard.
Use --shard-size 100 or --skip-pair in those cases.

Output layout
─────────────
<out-dir>/
  manifest.json                  shared protein→shard index (filelock-guarded)
  <protein_stem>/
    shard_000000.h5              float16 single [N,L,d_s] + pair [N,L,L,d_p]
    shard_000001.h5
    ...

═══════════════════════════════════════════════════════════════════════════════
Chai-1 interception strategy — read before modifying
═══════════════════════════════════════════════════════════════════════════════

Chai-1's trunk (the Evoformer-style module that produces per-residue single
embeddings and residue-pair embeddings) is compiled as a TorchScript jit_module
wrapped in a thin Python ModuleWrapper.

  WHY hooks fail:
    register_forward_hook / register_forward_pre_hook are Python-level
    mechanisms.  They can only intercept calls to Python nn.Module.forward().
    TorchScript jit_modules are executed inside the C++ runtime; Python hooks
    are silently ignored for any op that lives inside the compiled graph.

  Our intercept:
    1. Load Chai-1 through its public Python API.
    2. Locate the trunk's Python-level ModuleWrapper via _locate_trunk().
       The ModuleWrapper IS a regular nn.Module; we can call it directly.
    3. Wrap it in TrunkOnlyWrapper — a thin nn.Module that calls the trunk
       and returns (single, pair) without ever invoking the structure /
       diffusion decoder.  ~70 % FLOPs saved per frame.
    4. Each frame's backbone coordinates are injected as template conditioning
       through the featurizer so the trunk output is conformation-specific,
       not just sequence-specific (ASSUMPTION C).

  ASSUMPTION A  Trunk attribute name
    The loaded model object has a Python-level attribute whose value is the
    ModuleWrapper over the TorchScript jit_module.  _TRUNK_CANDIDATES lists
    the names tried in order.  If none match, the script prints
    list(model.named_children()) and raises an AttributeError that tells you
    exactly what to change.

  ASSUMPTION B  Trunk output format
    trunk(batch) returns something with .single / .pair attributes, OR a plain
    2-tuple, OR a dict with those keys.  _unpack_trunk_output() handles all
    three forms.  Add a new branch if the actual format differs.

  ASSUMPTION C  Coordinate conditioning
    Each frame's backbone coordinates are injected via
    featurizer.featurize_with_coords(seq_features, coords) so the trunk
    produces conformation-specific embeddings (not constant sequence
    embeddings that would be identical across all MD frames).
    If that method does not exist, the fallback packs coordinates into
    "template_backbone_positions" (AlphaFold-style key) and logs a warning
    telling you which method to implement.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import MDAnalysis as mda
import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fluxion.extract")


# ─────────────────────────────────────────────────────────────────────────────
# Chai-1 trunk interception
# ─────────────────────────────────────────────────────────────────────────────

# Attribute names to probe on the loaded model object (ASSUMPTION A).
# Extend this list if a future chai-lab version renames the trunk.
_TRUNK_CANDIDATES: Tuple[str, ...] = ("trunk", "encoder", "evoformer")


def _locate_trunk(model: nn.Module) -> nn.Module:
    """Return the trunk ModuleWrapper from a loaded Chai-1 model.

    Probes _TRUNK_CANDIDATES in order.  Raises a diagnostic AttributeError
    with model.named_children() output if nothing matches so you know
    exactly which attribute to add to _TRUNK_CANDIDATES.
    """
    for attr in _TRUNK_CANDIDATES:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            # Must be a distinct nn.Module (not the model itself)
            if isinstance(candidate, nn.Module) and candidate is not model:
                log.info("Trunk located at model.%s  [%s]", attr, type(candidate).__name__)
                return candidate

    children = [(n, type(m).__name__) for n, m in model.named_children()]
    raise AttributeError(
        f"Trunk not found under any of: {_TRUNK_CANDIDATES}\n"
        f"model.named_children() = {children}\n"
        "Add the correct attribute name to _TRUNK_CANDIDATES and re-run."
    )


def _unpack_trunk_output(out) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (single, pair) tensors from the trunk's raw return value (ASSUMPTION B).

    Handles three common return formats:
      - Object with .single / .pair attributes  (namedtuple, dataclass)
      - Plain 2-tuple / list
      - Dict with 'single' / 'pair' keys
    """
    if hasattr(out, "single") and hasattr(out, "pair"):
        return out.single, out.pair
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    if isinstance(out, dict) and "single" in out and "pair" in out:
        return out["single"], out["pair"]

    available = dir(out) if not isinstance(out, dict) else list(out.keys())
    raise TypeError(
        f"Cannot unpack trunk output of type {type(out)!r}.\n"
        f"Available attributes / keys: {available}\n"
        "Add a new branch to _unpack_trunk_output() for this format."
    )


class TrunkOnlyWrapper(nn.Module):
    """Calls only the Chai-1 trunk and returns (single, pair); never calls the decoder.

    This is the core of the JIT-safe extraction strategy.  By stopping at the
    trunk boundary (which is a Python-level nn.Module call) we avoid the
    structure / diffusion decoder entirely, saving ~70 % of per-frame FLOPs.

    Returns
    -------
    single : Tensor [L, d_s]      per-residue representation
    pair   : Tensor [L, L, d_p]   residue-pair representation
    """

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk

    @torch.no_grad()
    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.trunk(batch)
        single, pair = _unpack_trunk_output(out)
        # Strip batch dim if the trunk returns [1, L, d] shaped tensors
        single = single.squeeze(0) if single.dim() == 3 else single  # → [L, d_s]
        pair   = pair.squeeze(0)   if pair.dim() == 4   else pair    # → [L, L, d_p]
        return single, pair


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_chai1_trunk(device: torch.device) -> Tuple[TrunkOnlyWrapper, object]:
    """Load Chai-1 weights and return (TrunkOnlyWrapper, featurizer)."""
    
    log.info("Attempting to initialize Chai-1...")
    
    # Use the verified import path for your version of chai-lab
    from chai_lab.chai1 import load_model
    
    # Load the model directly
    raw_model = load_model(device=device)
    featurizer = raw_model

    raw_model.eval().to(device)
    for p in raw_model.parameters():
        p.requires_grad_(False)

    wrapper = TrunkOnlyWrapper(_locate_trunk(raw_model)).to(device).eval()
    log.info("TrunkOnlyWrapper ready on %s", device)
    
    return wrapper, featurizer


# ─────────────────────────────────────────────────────────────────────────────
# Featurization  (MDAnalysis frame → Chai-1 input batch dict)
# ─────────────────────────────────────────────────────────────────────────────

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_sequence(u: mda.Universe) -> str:
    """Extract the protein's one-letter sequence from the MDAnalysis Universe."""
    seq = "".join(
        _THREE_TO_ONE.get(r.resname.upper(), "X")
        for r in u.select_atoms("protein").residues
    )
    log.info("Sequence: %d residues", len(seq))
    return seq


def build_seq_features(sequence: str, featurizer) -> Dict:
    """Compute sequence-constant features once; reused for every frame.

    Tries featurizer method names in order of likelihood.  Falls back to a
    minimal raw dict if none match, with a warning.
    """
    for method in ("encode_sequence", "tokenize", "featurize_sequence"):
        if hasattr(featurizer, method):
            log.info("Encoding sequence via featurizer.%s()", method)
            return getattr(featurizer, method)(sequence)

    log.warning(
        "featurizer has no encode_sequence / tokenize / featurize_sequence method. "
        "Using raw fallback dict.  Update build_seq_features() if trunk forward errors."
    )
    return {"sequence": sequence}


def frame_to_batch(
    positions: np.ndarray,    # [N_backbone_atoms, 3] in Å, already .copy()-ed
    seq_features: Dict,
    featurizer,
    device: torch.device,
) -> Dict:
    """Merge static sequence features with per-frame backbone coordinates.

    Coordinate injection strategy (ASSUMPTION C):
      backbone coordinates from the current MD frame are passed to the
      featurizer as structural / template conditioning.  This is what makes
      the trunk's output vary across conformations rather than being a constant
      sequence embedding.

    Backbone atom ordering expected by Chai-1 (standard PDB convention):
      N, CA, C, O  per residue, in residue order.
    MDAnalysis "protein and backbone" selection matches this ordering when the
    PDB was written with standard atom names.
    """
    coords = torch.from_numpy(positions).float()   # [N_bb, 3]

    # ── Try featurizer's native coordinate injection (preferred) ─────────────
    batch: Optional[Dict] = None
    for method in ("featurize_with_coords", "add_template_coords"):
        if hasattr(featurizer, method):
            batch = getattr(featurizer, method)(seq_features, coords)
            break

    # ── Fallback: inject under the AlphaFold-style template key ─────────────
    if batch is None:
        log.warning(
            "Featurizer has no featurize_with_coords / add_template_coords method. "
            "Packing coords under 'template_backbone_positions' (AlphaFold-style). "
            "If the trunk forward errors, implement one of those methods or map "
            "coords to the key your version of chai-lab actually expects."
        )
        batch = {k: v for k, v in seq_features.items()}
        batch["template_backbone_positions"] = coords.unsqueeze(0)   # [1, N_bb, 3]

    # Move tensors to device; leave non-tensor values (strings, ints) as-is
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 sharding
# ─────────────────────────────────────────────────────────────────────────────

def write_shard(
    protein_dir: Path,
    shard_idx: int,
    singles: List[torch.Tensor],    # each [L, d_s], CPU, float16
    pairs:   List[torch.Tensor],    # each [L, L, d_p], CPU, float16 — empty if skip_pair
    frame_indices: List[int],
    skip_pair: bool,
) -> Path:
    """Stack buffered tensors and flush one HDF5 shard to disk.

    Storage format: float16 with LZF compression (fast, lossless).
    LZF is bundled with h5py; no external plugin needed.

    Shard datasets
    ──────────────
    single     float16  [N, L, d_s]       per-residue representations
    pair       float16  [N, L, L, d_p]    pair representations (if not skip_pair)
    frame_idx  int32    [N]               original trajectory frame index
    """
    protein_dir.mkdir(parents=True, exist_ok=True)
    path = protein_dir / f"shard_{shard_idx:06d}.h5"

    # Stack on CPU (buffers already moved to CPU)
    single_arr: np.ndarray = torch.stack(singles).numpy()   # [N, L, d_s],    float16
    idx_arr:    np.ndarray = np.array(frame_indices, dtype=np.int32)

    with h5py.File(path, "w") as f:
        f.create_dataset("single",    data=single_arr, compression="lzf")
        f.create_dataset("frame_idx", data=idx_arr)
        f.attrs.update({
            "protein":   protein_dir.name,
            "shard_idx": shard_idx,
            "n_frames":  len(singles),
            "d_single":  int(single_arr.shape[-1]),
            "L":         int(single_arr.shape[1]),
            "has_pair":  not skip_pair,
        })
        if not skip_pair:
            pair_arr: np.ndarray = torch.stack(pairs).numpy()   # [N, L, L, d_p], float16
            f.create_dataset("pair", data=pair_arr, compression="lzf")
            f.attrs["d_pair"] = int(pair_arr.shape[-1])

    size_mb = path.stat().st_size / 1e6
    log.info(
        "Shard %06d written: %d frames, %.1f MB on disk → %s",
        shard_idx, len(singles), size_mb, path,
    )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Shared manifest  (multi-GPU safe via filelock)
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_FILE = "manifest.json"
_MANIFEST_LOCK = "manifest.json.lock"


def read_manifest(out_dir: Path) -> Dict:
    path = out_dir / _MANIFEST_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def update_manifest(
    out_dir: Path,
    protein: str,
    shard_path: Path,
    n_frames: int,
    first_idx: int,
    last_idx: int,
) -> None:
    """Append one shard record to the manifest under an exclusive filelock.

    Safe for concurrent writes from multiple GPU workers on the same NFS mount.
    The lock file persists on disk; the lock is released when the context exits.
    """
    lock = FileLock(str(out_dir / _MANIFEST_LOCK))
    with lock:
        manifest = read_manifest(out_dir)
        manifest.setdefault(protein, {"shards": []})
        manifest[protein]["shards"].append({
            "path":      str(shard_path.relative_to(out_dir)),
            "n_frames":  n_frames,
            "first_idx": first_idx,
            "last_idx":  last_idx,
        })
        with open(out_dir / _MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)


def frames_already_done(out_dir: Path, protein: str) -> int:
    """Total frames already extracted for this protein (used for resume)."""
    return sum(
        s["n_frames"]
        for s in read_manifest(out_dir).get(protein, {}).get("shards", [])
    )


def shards_already_done(out_dir: Path, protein: str) -> int:
    return len(read_manifest(out_dir).get(protein, {}).get("shards", []))


# ─────────────────────────────────────────────────────────────────────────────
# Flush helper
# ─────────────────────────────────────────────────────────────────────────────

def _flush_shard(
    protein_dir: Path,
    out_dir: Path,
    protein: str,
    shard_idx: int,
    singles: List[torch.Tensor],
    pairs: List[torch.Tensor],
    frame_indices: List[int],
    skip_pair: bool,
) -> None:
    """Write shard to disk and update the shared manifest atomically."""
    path = write_shard(
        protein_dir, shard_idx, singles, pairs, frame_indices, skip_pair
    )
    update_manifest(
        out_dir, protein, path,
        n_frames=len(singles),
        first_idx=frame_indices[0],
        last_idx=frame_indices[-1],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract(args: argparse.Namespace) -> None:
    device   = torch.device(args.device)
    out_dir  = Path(args.out_dir)
    protein  = Path(args.pdb).stem

    os.makedirs(out_dir, exist_ok=True)

    log.info("━━━ Fluxion latent extraction ━━━")
    log.info("  protein    : %s", protein)
    log.info("  pdb        : %s", args.pdb)
    log.info("  dcd        : %s", args.dcd)
    log.info("  out_dir    : %s", out_dir)
    log.info("  shard_size : %d frames", args.shard_size)
    log.info("  device     : %s", device)
    log.info("  skip_pair  : %s", args.skip_pair)

    # ── Load frozen model ────────────────────────────────────────────────────
    trunk, featurizer = load_chai1_trunk(device)

    # ── Open trajectory without loading into RAM ─────────────────────────────
    # MDAnalysis lazily reads DCD frames on each iteration; the full trajectory
    # never lives in system memory simultaneously.
    u       = mda.Universe(args.pdb, args.dcd)
    n_total = len(u.trajectory)
    log.info("Trajectory: %d frames total", n_total)

    # Select backbone atoms once; reuse the AtomGroup across all frames
    backbone = u.select_atoms("protein and backbone")
    log.info("Backbone atoms selected: %d", len(backbone))

    # ── Resume: skip frames that already have shards ─────────────────────────
    n_done     = frames_already_done(out_dir, protein)
    shard_idx  = shards_already_done(out_dir, protein)
    start_frame = n_done

    if start_frame >= n_total:
        log.info("All %d frames already extracted — nothing to do.", n_total)
        return
    if start_frame > 0:
        log.info("Resuming from frame %d / %d (shard index %d).", start_frame, n_total, shard_idx)

    # ── Sequence features (constant across all frames) ───────────────────────
    sequence     = get_sequence(u)
    seq_features = build_seq_features(sequence, featurizer)

    # ── Storage size estimate (pair reps can be large) ───────────────────────
    if not args.skip_pair:
        L              = len(sequence)
        d_p_est        = 128     # Chai-1 pair dim; typical value — adjust if needed
        n_shards_rem   = (n_total - start_frame + args.shard_size - 1) // args.shard_size
        gb_per_shard   = (L * L * d_p_est * 2 * args.shard_size) / 1e9  # float16 bytes
        gb_buf_peak    = (L * L * d_p_est * 2 * args.shard_size) / 1e9  # RAM buffer peak
        log.info(
            "Pair storage estimate: ~%.1f GB/shard × %d shards = ~%.0f GB total on disk.  "
            "Peak RAM buffer: ~%.1f GB.  Use --skip-pair or --shard-size 100 to reduce.",
            gb_per_shard, n_shards_rem, gb_per_shard * n_shards_rem, gb_buf_peak,
        )

    # ── Pre-flight validation on frame 0 ─────────────────────────────────────
    # Run one frame through the full pipeline before committing to the full
    # trajectory.  This catches wrong attribute names, bad output shapes, and
    # featurizer mismatches early.
    log.info("Pre-flight check: running trunk on frame 0 …")
    u.trajectory[0]
    pos0  = backbone.positions.copy()
    bat0  = frame_to_batch(pos0, seq_features, featurizer, device)
    s0, p0 = trunk(bat0)
    assert s0.dim() == 2, (
        f"Expected single [L, d_s] (2D), got shape {tuple(s0.shape)}. "
        "Check _unpack_trunk_output() and the squeeze logic in TrunkOnlyWrapper."
    )
    assert p0.dim() == 3, (
        f"Expected pair [L, L, d_p] (3D), got shape {tuple(p0.shape)}. "
        "Check _unpack_trunk_output() and the squeeze logic in TrunkOnlyWrapper."
    )
    d_s, d_p = s0.shape[-1], p0.shape[-1]
    log.info("  single: %s   pair: %s   ✓", tuple(s0.shape), tuple(p0.shape))
    del s0, p0, bat0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Shard accumulation buffers ────────────────────────────────────────────
    # Tensors are moved to CPU and cast to float16 immediately to keep the
    # GPU free for the next forward pass and to halve RAM usage vs float32.
    singles_buf:   List[torch.Tensor] = []
    pairs_buf:     List[torch.Tensor] = []
    frame_idx_buf: List[int]          = []

    protein_dir = out_dir / protein
    t0          = time.perf_counter()
    frames_done = 0

    pbar = tqdm(
        u.trajectory[start_frame:],
        total=n_total - start_frame,
        unit="frame",
        dynamic_ncols=True,
        desc=protein,
        smoothing=0.1,
    )

    for ts in pbar:
        # ── Extract this frame ───────────────────────────────────────────────
        # CRITICAL: .copy() the positions before the next timestep advances the
        # underlying MDAnalysis memory buffer and overwrites them in-place.
        positions = backbone.positions.copy()   # [N_bb, 3], float32, Å

        batch          = frame_to_batch(positions, seq_features, featurizer, device)
        single, pair   = trunk(batch)   # [L, d_s], [L, L, d_p]

        # Move to CPU and cast to float16 immediately to free VRAM
        singles_buf.append(single.detach().cpu().half())
        if not args.skip_pair:
            pairs_buf.append(pair.detach().cpu().half())
        frame_idx_buf.append(int(ts.frame))
        frames_done += 1

        # ── Flush shard when buffer is full ──────────────────────────────────
        if len(singles_buf) >= args.shard_size:
            _flush_shard(
                protein_dir, out_dir, protein, shard_idx,
                singles_buf, pairs_buf, frame_idx_buf, args.skip_pair,
            )
            shard_idx += 1
            singles_buf.clear()
            pairs_buf.clear()
            frame_idx_buf.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Update tqdm postfix with live throughput
        fps = frames_done / max(time.perf_counter() - t0, 1e-9)
        pbar.set_postfix(fps=f"{fps:.1f}", shard=shard_idx, d_s=d_s, d_p=d_p)

    # ── Flush final partial shard ─────────────────────────────────────────────
    if singles_buf:
        _flush_shard(
            protein_dir, out_dir, protein, shard_idx,
            singles_buf, pairs_buf, frame_idx_buf, args.skip_pair,
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "Extraction complete: %d frames in %.1f s (%.2f fps), %d shards written.",
        frames_done, elapsed, frames_done / max(elapsed, 1e-9), shard_idx + 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Chai-1 trunk latents from an ATLAS MD trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pdb", required=True,
        help="Reference topology file (.pdb).",
    )
    p.add_argument(
        "--dcd", required=True,
        help="Trajectory file (.dcd).",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Scratch directory root for HDF5 shards and manifest.",
    )
    p.add_argument(
        "--shard-size", type=int, default=1000,
        help=(
            "Number of frames per HDF5 shard.  "
            "Reduce to 100 for proteins with L > 400 when storing pair reps."
        ),
    )
    p.add_argument(
        "--device", default="cuda",
        help="PyTorch device string (cuda, cuda:1, cpu).",
    )
    p.add_argument(
        "--skip-pair", action="store_true",
        help=(
            "Do not store pair representations [L, L, d_p].  "
            "Pair reps are O(L²) larger than single reps and can consume "
            "tens of GB per shard for long sequences.  "
            "Use when only single reps are needed for the DiT."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(args)
