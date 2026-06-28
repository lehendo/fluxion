#!/usr/bin/env python3
"""
data_pipeline/extract_latents.py — Chai-lab 0.6.1

Swarm-style latent extraction: stream ATLAS MD trajectory frames through the
frozen Chai-1 trunk pipeline and write per-frame token representations to HDF5
shards.  One instance per GPU; multiple instances share a manifest safely via
filelock.

Usage
─────
python data_pipeline/extract_latents.py \
    --pdb     /scratch/arjunc4/fluxion_data/raw/1ABC/protein.pdb \
    --dcd     /scratch/arjunc4/fluxion_data/raw/1ABC/traj.dcd \
    --out-dir /scratch/arjunc4/fluxion_data/latents \
    --shard-size 1000 \
    --num-trunk-recycles 3 \
    --device cuda
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import MDAnalysis as mda
import numpy as np
import torch
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
# Chai-lab 0.6.1 imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from chai_lab.chai1 import load_exported  # type: ignore
except ImportError:
    raise ImportError("load_exported not found in chai_lab.chai1. Verify chai-lab==0.6.1 is installed.")

try:
    from chai_lab.data.collate.collate import Collate                  # type: ignore
    from chai_lab.chai1 import feature_factory                         # type: ignore
except ImportError:
    raise ImportError("Cannot import Collate or feature_factory.")

try:
    from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext  # type: ignore
except ImportError:
    raise ImportError("Cannot import AllAtomFeatureContext.")

try:
    from chai_lab.data.features.generators.token_bond import TokenBondRestraint  # type: ignore
except ImportError:
    raise ImportError("Cannot import TokenBondRestraint.")

try:
    from chai_lab.model.utils import move_data_to_device  # type: ignore
except ImportError:
    def move_data_to_device(data, device):  # type: ignore
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: move_data_to_device(v, device) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(move_data_to_device(v, device) for v in data)
        return data

try:
    from chai_lab.utils.tensor_utils import und_self  # type: ignore
except ImportError:
    def und_self(mask, pattern: str):  # type: ignore
        return mask.unsqueeze(-1) & mask.unsqueeze(-2)

# ─────────────────────────────────────────────────────────────────────────────
# Amino acid helpers
# ─────────────────────────────────────────────────────────────────────────────
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def _get_sequence(u: mda.Universe) -> str:
    return "".join(
        _THREE_TO_ONE.get(r.resname.upper(), "X")
        for r in u.select_atoms("protein").residues
    )

# ─────────────────────────────────────────────────────────────────────────────
# AllAtomFeatureContext construction
# ─────────────────────────────────────────────────────────────────────────────
def _write_fasta(sequence: str, name: str, path: Path) -> None:
    path.write_text(f">protein|{name}\n{sequence}\n")

def _build_feature_context(fasta_path: Path, esm_cache_dir: Path, msa_dir: Optional[Path] = None) -> "AllAtomFeatureContext":
    """Build AllAtomFeatureContext from a FASTA file using chai-lab 0.6.1 internals."""
    from chai_lab.chai1 import make_all_atom_feature_context  # type: ignore
    
    esm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    return make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=esm_cache_dir,
        use_esm_embeddings=True,
        msa_directory=msa_dir,
        esm_device=torch.device("cpu"), # Keep ESM on CPU to save VRAM
    )

# ─────────────────────────────────────────────────────────────────────────────
# Per-frame coordinate injection
# ─────────────────────────────────────────────────────────────────────────────
_COORD_FIELD_CANDIDATES = ("atom_gt_coords", "all_atom_positions", "positions", "coords")

def _patch_atom_coords(
    context: "AllAtomFeatureContext",
    positions: np.ndarray,   # [N_atoms, 3] float32, Å
) -> "AllAtomFeatureContext":
    struct_ctx = context.structure_context
    
    # Topology Alignment Patch
    expected_atoms = struct_ctx.atom_exists_mask.shape[0]
    if positions.shape[0] > expected_atoms:
        positions = positions[:expected_atoms]
    elif positions.shape[0] < expected_atoms:
        pad = np.zeros((expected_atoms - positions.shape[0], 3), dtype=positions.dtype)
        positions = np.vstack([positions, pad])

    pos_tensor = torch.from_numpy(positions).float()

    for field_name in _COORD_FIELD_CANDIDATES:
        if hasattr(struct_ctx, field_name):
            new_struct_ctx = dataclasses.replace(struct_ctx, **{field_name: pos_tensor})
            return dataclasses.replace(context, structure_context=new_struct_ctx)

    raise AttributeError(f"Cannot find atom coordinate field on {type(struct_ctx).__name__}.")

# ─────────────────────────────────────────────────────────────────────────────
# Core 0.6.1 trunk pipeline
# ─────────────────────────────────────────────────────────────────────────────
_COLLATOR: Optional["Collate"] = None

def _get_collator() -> "Collate":
    global _COLLATOR
    if _COLLATOR is None:
        _COLLATOR = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
    return _COLLATOR

def _extract_from_batch(batch: Dict, key: str, fallback_keys: Tuple[str, ...] = ()) -> torch.Tensor:
    for subdict_name in ("inputs", "features"):
        subdict = batch.get(subdict_name, {})
        if key in subdict:
            return subdict[key]
        for fk in fallback_keys:
            if fk in subdict:
                return subdict[fk]
    raise KeyError(f"Key '{key}' (or fallbacks {fallback_keys}) not found in batch.")

@torch.no_grad()
def run_trunk_pipeline(
    feature_context: "AllAtomFeatureContext",
    loaded_modules: Dict[str, torch.jit.ScriptModule],
    device: torch.device,
    num_trunk_recycles: int = 3,
    low_memory: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    feature_embedding    = loaded_modules["feature_embedding"]
    bond_loss_input_proj = loaded_modules["bond_loss_input_proj"]
    token_input_embedder = loaded_modules["token_embedder"]
    trunk                = loaded_modules["trunk"]

    # ── Strict Precision Airlock ──
    def _to_bf16(x):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.bfloat16()
        if isinstance(x, dict):
            return {k: _to_bf16(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_to_bf16(v) for v in x)
        return x

    # ── Stage 1: Collation ───────────────────────────────────────────────────
    batch = _get_collator()([feature_context])
    if not low_memory:
        batch = move_data_to_device(batch, device=device)

    features = {name: feat for name, feat in batch["features"].items()}
    inputs   = batch["inputs"]

    block_indices_h      = inputs["block_atom_pair_q_idces"]
    block_indices_w      = inputs["block_atom_pair_kv_idces"]
    atom_single_mask     = inputs["atom_exists_mask"]
    atom_token_indices   = inputs["atom_token_index"].long()
    token_single_mask    = inputs["token_exists_mask"]
    token_pair_mask      = und_self(token_single_mask, "b i, b j -> b i j")
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

    msa_mask = _extract_from_batch(batch, "msa_mask", fallback_keys=("msa_seq_mask",))
    _, _, model_size = msa_mask.shape

    raw_template_mask = _extract_from_batch(
        batch, "template_mask", fallback_keys=("template_exists_mask", "template_token_mask")
    )
    template_input_masks = und_self(raw_template_mask, "b t n1, b t n2 -> b t n1 n2")

    # ── Stage 2: Feature embedding ───────────────────────────────────────────
    embedded_features = feature_embedding.forward(
        crop_size=model_size,
        move_to_device=device,
        return_on_cpu=low_memory,
        **_to_bf16(features),
    )

    token_single_input_feats   = _to_bf16(embedded_features["TOKEN"])
    token_pair_input_feats, _  = _to_bf16(embedded_features["TOKEN_PAIR"].chunk(2, dim=-1))
    atom_single_input_feats, _ = _to_bf16(embedded_features["ATOM"].chunk(2, dim=-1))
    block_atom_pair_input_feats, _ = _to_bf16(embedded_features["ATOM_PAIR"].chunk(2, dim=-1))
    template_input_feats       = _to_bf16(embedded_features["TEMPLATES"])
    msa_input_feats            = _to_bf16(embedded_features["MSA"])

    del embedded_features

    # ── Stage 3: Bond features ───────────────────────────────────────────────
    bond_ft = TokenBondRestraint().generate(batch=batch).data
    
    bond_proj_out = bond_loss_input_proj.forward(
        return_on_cpu=low_memory,
        move_to_device=device,
        crop_size=model_size,
        input=bond_ft.float(), 
    )
    
    trunk_bond_feat, _ = _to_bf16(bond_proj_out.chunk(2, dim=-1))
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat
    del bond_ft, bond_proj_out, trunk_bond_feat

    # ── Stage 4: Token input embedder ────────────────────────────────────────
    token_input_embedder_outputs = token_input_embedder.forward(
        return_on_cpu=low_memory,
        move_to_device=device,
        token_single_input_feats=token_single_input_feats,
        token_pair_input_feats=token_pair_input_feats,
        atom_single_input_feats=atom_single_input_feats,
        block_atom_pair_feat=block_atom_pair_input_feats,
        block_atom_pair_mask=block_atom_pair_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_single_mask=atom_single_mask,
        atom_token_indices=atom_token_indices,
        crop_size=model_size,
    )
    token_single_initial_repr = _to_bf16(token_input_embedder_outputs[0])
    token_pair_initial_repr   = _to_bf16(token_input_embedder_outputs[2])
    del token_input_embedder_outputs

    del (
        token_single_input_feats, token_pair_input_feats,
        atom_single_input_feats, block_atom_pair_input_feats,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Stage 5: Trunk recycles ──────────────────────────────────────────────
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr   = token_pair_initial_repr

    for _ in range(num_trunk_recycles):
        (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
            move_to_device=device,
            token_single_trunk_initial_repr=token_single_initial_repr,
            token_pair_trunk_initial_repr=token_pair_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            msa_input_feats=msa_input_feats,
            msa_mask=msa_mask,
            template_input_feats=template_input_feats,
            template_input_masks=template_input_masks,
            token_single_mask=token_single_mask,
            token_pair_mask=token_pair_mask,
            crop_size=model_size,
        )
        token_single_trunk_repr = _to_bf16(token_single_trunk_repr)
        token_pair_trunk_repr   = _to_bf16(token_pair_trunk_repr)

    # ── Short-circuit ────────────────────────────────────────────────────────
    del (
        token_single_initial_repr, token_pair_initial_repr,
        msa_input_feats, msa_mask,
        template_input_feats, template_input_masks,
        token_single_mask, token_pair_mask,
        batch, features, inputs,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    single = token_single_trunk_repr.squeeze(0).cpu()
    pair   = token_pair_trunk_repr.squeeze(0).cpu()

    del token_single_trunk_repr, token_pair_trunk_repr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return single, pair

# ─────────────────────────────────────────────────────────────────────────────
# HDF5 sharding
# ─────────────────────────────────────────────────────────────────────────────
def write_shard(
    protein_dir: Path,
    shard_idx: int,
    singles: List[torch.Tensor],
    pairs:   List[torch.Tensor],
    frame_indices: List[int],
    skip_pair: bool,
    shard_prefix: str = "",
) -> Path:
    protein_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{shard_prefix}_" if shard_prefix else ""
    path = protein_dir / f"{prefix}shard_{shard_idx:06d}.h5"

    single_arr: np.ndarray = torch.stack(singles).numpy()
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
            pair_arr: np.ndarray = torch.stack(pairs).numpy()
            f.create_dataset("pair", data=pair_arr, compression="lzf")
            f.attrs["d_pair"] = int(pair_arr.shape[-1])

    log.info("Shard %06d written: %d frames, %.1f MB → %s", shard_idx, len(singles), path.stat().st_size / 1e6, path)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Shared manifest
# ─────────────────────────────────────────────────────────────────────────────
_MANIFEST_FILE = "manifest.json"
_MANIFEST_LOCK = "manifest.json.lock"

def read_manifest(out_dir: Path) -> Dict:
    path = out_dir / _MANIFEST_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def update_manifest(out_dir: Path, protein: str, shard_path: Path, n_frames: int, first_idx: int, last_idx: int) -> None:
    with FileLock(str(out_dir / _MANIFEST_LOCK)):
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

def frames_already_done(out_dir: Path, protein: str, shard_prefix: str = "") -> int:
    shards = read_manifest(out_dir).get(protein, {}).get("shards", [])
    if shard_prefix:
        shards = [s for s in shards if Path(s["path"]).name.startswith(shard_prefix + "_")]
    return sum(s["n_frames"] for s in shards)

def shards_already_done(out_dir: Path, protein: str, shard_prefix: str = "") -> int:
    shards = read_manifest(out_dir).get(protein, {}).get("shards", [])
    if shard_prefix:
        shards = [s for s in shards if Path(s["path"]).name.startswith(shard_prefix + "_")]
    return len(shards)

def _flush_shard(protein_dir: Path, out_dir: Path, protein: str, shard_idx: int, singles: List[torch.Tensor], pairs: List[torch.Tensor], frame_indices: List[int], skip_pair: bool, shard_prefix: str = "") -> None:
    path = write_shard(protein_dir, shard_idx, singles, pairs, frame_indices, skip_pair, shard_prefix=shard_prefix)
    update_manifest(out_dir, protein, path, n_frames=len(singles), first_idx=frame_indices[0], last_idx=frame_indices[-1])

# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def extract(args: argparse.Namespace) -> None:
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    esm_cache_dir = out_dir / "esm_cache"

    log.info("Loading Chai-1 compiled modules onto %s …", device)
    loaded_modules: Dict[str, torch.jit.ScriptModule] = {
        "feature_embedding":    load_exported("feature_embedding.pt",    device),
        "bond_loss_input_proj": load_exported("bond_loss_input_proj.pt", device),
        "token_embedder":       load_exported("token_embedder.pt",       device),
        "trunk":                load_exported("trunk.pt",                device),
    }

    if args.fasta:
        protein = Path(args.fasta).stem
        shard_prefix = args.shard_prefix if args.shard_prefix else ""
        log.info("FASTA mode: encoding single sequence for %s", protein)
        context = _build_feature_context(Path(args.fasta), esm_cache_dir)
        single, pair = run_trunk_pipeline(context, loaded_modules, device, num_trunk_recycles=args.num_trunk_recycles, low_memory=args.low_memory)
        protein_dir = out_dir / protein
        _flush_shard(protein_dir, out_dir, protein, 0, singles=[single.half()], pairs=[] if args.skip_pair else [pair.half()], frame_indices=[0], skip_pair=args.skip_pair, shard_prefix=shard_prefix)
        log.info("Done: single %s  pair %s", tuple(single.shape), tuple(pair.shape))
        return

    protein = Path(args.pdb).stem
    shard_prefix = args.shard_prefix if args.shard_prefix else ""
    log.info("MD mode: protein=%s  device=%s  recycles=%d  skip_pair=%s  shard_prefix=%r", protein, device, args.num_trunk_recycles, args.skip_pair, shard_prefix)

    u       = mda.Universe(args.pdb, args.traj)
    n_total = len(u.trajectory)
    log.info("Trajectory: %d frames", n_total)
    all_atoms = u.select_atoms("protein")

    n_done      = frames_already_done(out_dir, protein, shard_prefix)
    shard_idx   = shards_already_done(out_dir, protein, shard_prefix)
    start_frame = n_done

    if start_frame >= n_total:
        log.info("All %d frames already extracted — nothing to do.", n_total)
        return

    sequence = _get_sequence(u)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", prefix=f"{protein}_", delete=False) as fh:
        fasta_tmp = Path(fh.name)
        _write_fasta(sequence, protein, fasta_tmp)
    try:
        base_context = _build_feature_context(fasta_tmp, esm_cache_dir)
    finally:
        fasta_tmp.unlink(missing_ok=True)

    log.info("Pre-flight: validating trunk pipeline on frame 0 …")
    u.trajectory[0]
    test_ctx = _patch_atom_coords(base_context, all_atoms.positions.copy())
    s0, p0 = run_trunk_pipeline(test_ctx, loaded_modules, device, num_trunk_recycles=args.num_trunk_recycles, low_memory=args.low_memory)
    d_s, d_p = s0.shape[-1], p0.shape[-1]
    log.info("  single: %s   pair: %s   ✓", tuple(s0.shape), tuple(p0.shape))
    del s0, p0, test_ctx

    singles_buf, pairs_buf, frame_idx_buf = [], [], []
    protein_dir = out_dir / protein
    t0          = time.perf_counter()
    frames_done = 0

    pbar = tqdm(u.trajectory[start_frame:], total=n_total - start_frame, unit="frame", dynamic_ncols=True, desc=protein, smoothing=0.1)

    for ts in pbar:
        positions = all_atoms.positions.copy()
        frame_context = _patch_atom_coords(base_context, positions)
        single, pair = run_trunk_pipeline(frame_context, loaded_modules, device, num_trunk_recycles=args.num_trunk_recycles, low_memory=args.low_memory)

        singles_buf.append(single.half())
        if not args.skip_pair:
            pairs_buf.append(pair.half())
        frame_idx_buf.append(int(ts.frame))
        frames_done += 1

        if len(singles_buf) >= args.shard_size:
            _flush_shard(protein_dir, out_dir, protein, shard_idx, singles_buf, pairs_buf, frame_idx_buf, args.skip_pair, shard_prefix=shard_prefix)
            shard_idx += 1
            singles_buf.clear(); pairs_buf.clear(); frame_idx_buf.clear()

        fps = frames_done / max(time.perf_counter() - t0, 1e-9)
        pbar.set_postfix(fps=f"{fps:.1f}", shard=shard_idx, d_s=d_s, d_p=d_p)

    if singles_buf:
        _flush_shard(protein_dir, out_dir, protein, shard_idx, singles_buf, pairs_buf, frame_idx_buf, args.skip_pair, shard_prefix=shard_prefix)

    elapsed = time.perf_counter() - t0
    log.info("Extraction complete: %d frames in %.1fs (%.2f fps), %d shards.", frames_done, elapsed, frames_done / max(elapsed, 1e-9), shard_idx + 1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--pdb", help="Reference topology (.pdb).")
    src.add_argument("--fasta", help="FASTA file.")
    p.add_argument("--traj", help="Trajectory file (.xtc or .dcd — MDAnalysis autodetects).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-prefix", default=None,
                   help="Prefix for shard filenames. Defaults to protein name. "
                        "Set to e.g. '16pk_A_R1' per replica to avoid concurrent write collisions.")
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-trunk-recycles", type=int, default=3)
    p.add_argument("--low-memory", action="store_true")
    p.add_argument("--skip-pair", action="store_true")
    
    args = p.parse_args()
    
    # --- CLI Safety Validations ---
    if args.fasta is None and args.pdb is None:
        p.error("One of --pdb or --fasta is required.")
    if args.pdb and not args.traj:
        p.error("--traj is required when using --pdb.")
        
    return args

if __name__ == "__main__":
    args = parse_args()
    extract(args)