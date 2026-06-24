#!/usr/bin/env python3
"""
data_pipeline/extract_latents.py — Chai-lab 0.6.1

Swarm-style latent extraction: stream ATLAS MD trajectory frames through the
frozen Chai-1 trunk pipeline and write per-frame token representations to HDF5
shards.  One instance per GPU; multiple instances share a manifest safely via
filelock.

Usage
─────
python data_pipeline/extract_latents.py \\
    --pdb     /scratch/arjunc4/fluxion_data/raw/1ABC/protein.pdb \\
    --dcd     /scratch/arjunc4/fluxion_data/raw/1ABC/traj.dcd \\
    --out-dir /scratch/arjunc4/fluxion_data/latents \\
    --shard-size 1000 \\
    --num-trunk-recycles 3 \\
    --device cuda

Alternatively, to encode from a FASTA (no trajectory):
    python data_pipeline/extract_latents.py --fasta protein.fasta --out-dir ...

For large proteins (L > 400) the pair buffer can exceed 10 GB RAM per shard.
Use --shard-size 100 or --skip-pair in those cases.

Output layout
─────────────
<out-dir>/
  manifest.json                 shared protein→shard index (filelock-guarded)
  <protein_stem>/
    shard_000000.h5             float16  single [N,L,d_s]  +  pair [N,L,L,d_p]
    shard_000001.h5
    ...

═══════════════════════════════════════════════════════════════════════════════
Chai-lab 0.6.1 pipeline
═══════════════════════════════════════════════════════════════════════════════

In 0.6.1 the monolithic load_model() was replaced with isolated, pre-compiled
TorchScript modules loaded via load_exported().  The orchestration code below
faithfully replicates the sequence from run_folding_on_context() (chai1.py) up
to the final trunk recycle — then stops.  diffusion_module.pt and
confidence_head.pt are never loaded.

Stage 1  Collation
    Collate(feature_factory, num_key_atoms=128, num_query_atoms=32)([context])
    → batch dict with "features" and "inputs" sub-dicts.

Stage 2  Feature embedding
    load_exported("feature_embedding.pt") — embeds TOKEN, TOKEN_PAIR, ATOM,
    ATOM_PAIR, TEMPLATES, MSA feature groups.

Stage 3  Bond features
    TokenBondRestraint().generate(batch) → bond_ft
    load_exported("bond_loss_input_proj.pt") → trunk_bond_feat (added to
    token_pair_input_feats) + structure_bond_feat (discarded — only needed
    by the structure decoder we skip).

Stage 4  Token input embedder
    load_exported("token_embedder.pt") → token_single_initial_repr,
    token_single_structure_input (discarded), token_pair_initial_repr.

Stage 5  Trunk recycles  ← WE STOP HERE
    load_exported("trunk.pt") — run num_trunk_recycles times, recycling
    token_single_trunk_repr and token_pair_trunk_repr.
    FINAL OUTPUTS:  token_single_trunk_repr [B, L, d_s]
                    token_pair_trunk_repr   [B, L, L, d_p]

Per-frame coordinate injection
    Each MD frame's backbone coordinates are written into the
    AllAtomStructureContext before collation so that the ATOM features seen by
    the trunk vary per conformation.  Specifically, _patch_atom_coords() calls
    dataclasses.replace() on AllAtomStructureContext with an updated coordinate
    field — a shallow copy that shares all other fields with the base context.

    Field name probed in order: atom_gt_coords, all_atom_positions, positions.
    A diagnostic error with dataclasses.fields() output is raised if none match.
═══════════════════════════════════════════════════════════════════════════════
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
# Each block is isolated so a failing import tells you exactly what's wrong.
# Resolve paths with:
#   python -c "import chai_lab; print(chai_lab.__file__)"
#   grep -r "def load_exported" $(python -c "import chai_lab,os;print(os.path.dirname(chai_lab.__file__))")

try:
    from chai_lab.chai1 import load_exported  # type: ignore
except ImportError:
    raise ImportError(
        "load_exported not found in chai_lab.chai1. "
        "Verify chai-lab==0.6.1 is installed.  Try:\n"
        "  grep -r 'def load_exported' <chai_lab_src>/\n"
        "then update this import."
    )

try:
    from chai_lab.data.collate.collate import Collate                  # type: ignore
    from chai_lab.chai1 import feature_factory                         # type: ignore
except ImportError:
    raise ImportError(
        "Cannot import Collate or feature_factory. "
        "Check chai_lab/data/collate/ and chai_lab/chai1.py."
    )

try:
    from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext  # type: ignore
except ImportError:
    raise ImportError(
        "Cannot import AllAtomFeatureContext.  "
        "Search: grep -r 'class AllAtomFeatureContext' <chai_lab_src>/"
    )

try:
    from chai_lab.data.features.generators.token_bond import TokenBondRestraint  # type: ignore
except ImportError:
    raise ImportError(
        "Cannot import TokenBondRestraint.  "
        "Check chai_lab/data/features/generators/token_bond.py"
    )

# move_data_to_device: moves nested dicts/lists of tensors to a device.
try:
    from chai_lab.model.utils import move_data_to_device  # type: ignore
except ImportError:
    def move_data_to_device(data, device):  # type: ignore
        """Pure-Python fallback for chai_lab.model.utils.move_data_to_device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: move_data_to_device(v, device) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(move_data_to_device(v, device) for v in data)
        return data

# und_self: outer-product boolean mask, pattern "b i, b j -> b i j"
try:
    from chai_lab.utils.tensor_utils import und_self  # type: ignore
except ImportError:
    def und_self(mask, pattern: str):  # type: ignore
        """Fallback: broadcast mask into a square pair mask."""
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


def _build_feature_context(fasta_path: Path, msa_dir: Optional[Path] = None) -> "AllAtomFeatureContext":
    """Build AllAtomFeatureContext from a FASTA file using chai-lab 0.6.1 internals."""
    from chai_lab.chai1 import make_all_atom_feature_context  # type: ignore
    
    # We pass a dummy output_dir since we don't save embeddings here
    dummy_out = fasta_path.parent / "dummy_embeddings"
    dummy_out.mkdir(exist_ok=True)
    
    return make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=dummy_out,
        use_esm_embeddings=True,
        msa_directory=msa_dir,
        esm_device=torch.device("cpu"), # Keep ESM on CPU to save VRAM
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame coordinate injection
# ─────────────────────────────────────────────────────────────────────────────

# Field names probed on AllAtomStructureContext for atom coordinates.
# Add entries if your 0.6.1 build uses a different name.
_COORD_FIELD_CANDIDATES = ("atom_gt_coords", "all_atom_positions", "positions", "coords")


def _patch_atom_coords(
    context: "AllAtomFeatureContext",
    positions: np.ndarray,   # [N_atoms, 3] float32, Å
) -> "AllAtomFeatureContext":
    """Return a shallow copy of context with atom positions replaced for this frame.

    Uses dataclasses.replace() — O(1) allocation overhead since only the
    coordinate tensor is reallocated; all other structure_context fields are
    shared by reference.

    If this raises AttributeError, find the correct field name:
        import dataclasses
        from chai_lab... import AllAtomStructureContext
        print([f.name for f in dataclasses.fields(AllAtomStructureContext)])
    then add it to _COORD_FIELD_CANDIDATES above.
    """
    struct_ctx = context.structure_context
    pos_tensor = torch.from_numpy(positions).float()   # [N_atoms, 3]

    # Find the coordinate field
    for field_name in _COORD_FIELD_CANDIDATES:
        if hasattr(struct_ctx, field_name):
            new_struct_ctx = dataclasses.replace(struct_ctx, **{field_name: pos_tensor})
            return dataclasses.replace(context, structure_context=new_struct_ctx)

    # Diagnostic error
    try:
        fields = [f.name for f in dataclasses.fields(struct_ctx)]
    except TypeError:
        fields = list(vars(struct_ctx).keys())
    raise AttributeError(
        f"Cannot find atom coordinate field on {type(struct_ctx).__name__}.\n"
        f"Available fields: {fields}\n"
        f"Tried: {_COORD_FIELD_CANDIDATES}\n"
        "Add the correct name to _COORD_FIELD_CANDIDATES."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core 0.6.1 trunk pipeline
# ─────────────────────────────────────────────────────────────────────────────

_COLLATOR: Optional["Collate"] = None


def _get_collator() -> "Collate":
    """Return a module-level Collate instance (created once)."""
    global _COLLATOR
    if _COLLATOR is None:
        _COLLATOR = Collate(
            feature_factory=feature_factory,
            num_key_atoms=128,
            num_query_atoms=32,
        )
    return _COLLATOR


def _extract_from_batch(batch: Dict, key: str, fallback_keys: Tuple[str, ...] = ()) -> torch.Tensor:
    """Retrieve a tensor from batch['inputs'] or batch['features'] with a clear error."""
    for subdict_name in ("inputs", "features"):
        subdict = batch.get(subdict_name, {})
        if key in subdict:
            return subdict[key]
        for fk in fallback_keys:
            if fk in subdict:
                return subdict[fk]
    inputs_keys   = list(batch.get("inputs",   {}).keys())
    features_keys = list(batch.get("features", {}).keys())
    raise KeyError(
        f"Key '{key}' (or fallbacks {fallback_keys}) not found in batch.\n"
        f"  batch['inputs']   keys: {inputs_keys}\n"
        f"  batch['features'] keys: {features_keys}\n"
        "Update _extract_from_batch() call with the correct key."
    )


@torch.no_grad()
def run_trunk_pipeline(
    feature_context: "AllAtomFeatureContext",
    loaded_modules: Dict[str, torch.jit.ScriptModule],
    device: torch.device,
    num_trunk_recycles: int = 3,
    low_memory: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate run_folding_on_context() through the trunk, then stop.

    Faithfully mirrors the orchestration in chai_lab.chai1.run_folding_on_context,
    stopping after the trunk recycles and before the diffusion decoder.
    diffusion_module.pt and confidence_head.pt are never loaded or called.

    Parameters
    ----------
    feature_context   : AllAtomFeatureContext for this frame / sequence.
    loaded_modules    : dict with keys feature_embedding, bond_loss_input_proj,
                        token_embedder, trunk — loaded once in extract() and
                        reused across all frames.
    device            : target CUDA/CPU device.
    num_trunk_recycles: number of trunk recycle passes (default: 3).
    low_memory        : if True, tensors stay on CPU between modules (slower
                        but uses less VRAM).

    Returns
    -------
    single : Tensor [L, d_s]      token-level single representation (fp32, on CPU)
    pair   : Tensor [L, L, d_p]   token-level pair representation   (fp32, on CPU)
    """
    feature_embedding    = loaded_modules["feature_embedding"]
    bond_loss_input_proj = loaded_modules["bond_loss_input_proj"]
    token_input_embedder = loaded_modules["token_embedder"]
    trunk                = loaded_modules["trunk"]

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

    # MSA mask is used only to derive model_size (crop / sequence length).
    # Shape: [B, msa_depth, n_tokens] — last dim is the sequence crop size.
    msa_mask = _extract_from_batch(batch, "msa_mask", fallback_keys=("msa_seq_mask",))
    _, _, model_size = msa_mask.shape

    template_input_masks = _extract_from_batch(
        batch, "template_input_masks",
        fallback_keys=("template_mask", "template_exists_mask"),
    )

    # ── Stage 2: Feature embedding ───────────────────────────────────────────
    embedded_features = feature_embedding.forward(
        crop_size=model_size,
        move_to_device=device,
        return_on_cpu=low_memory,
        **features,
    )

    token_single_input_feats   = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = (
        embedded_features["TOKEN_PAIR"].chunk(2, dim=-1)
    )
    atom_single_input_feats, _ = embedded_features["ATOM"].chunk(2, dim=-1)
    block_atom_pair_input_feats, _ = embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats      = embedded_features["MSA"]

    # Free the full embedded_features dict to reclaim VRAM
    del embedded_features

    # ── Stage 3: Bond features ───────────────────────────────────────────────
    bond_ft = TokenBondRestraint().generate(batch=batch).data
    bond_proj_out = bond_loss_input_proj.forward(
        return_on_cpu=low_memory,
        move_to_device=device,
        crop_size=model_size,
        input=bond_ft,
    )
    trunk_bond_feat, _ = bond_proj_out.chunk(2, dim=-1)
    # structure_bond_feat (second chunk) is only needed by the structure decoder
    # which we explicitly skip.
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat
    del bond_ft, bond_proj_out, trunk_bond_feat

    # ── Stage 4: Token input embedder ────────────────────────────────────────
    token_input_embedder_outputs: Tuple[torch.Tensor, ...] = token_input_embedder.forward(
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
    # Outputs: (token_single_initial_repr, token_single_structure_input, token_pair_initial_repr)
    # token_single_structure_input is only used by the structure decoder — discard it.
    token_single_initial_repr = token_input_embedder_outputs[0]
    token_pair_initial_repr   = token_input_embedder_outputs[2]
    del token_input_embedder_outputs

    # Free intermediate embedding tensors before the trunk loop
    del (
        token_single_input_feats, token_pair_input_feats,
        atom_single_input_feats, block_atom_pair_input_feats,
        token_pair_structure_input_feats,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Stage 5: Trunk recycles ──────────────────────────────────────────────
    # This is the final stage we run.  After this loop, token_single_trunk_repr
    # and token_pair_trunk_repr are the latents we save.  We never load or call
    # diffusion_module.pt or confidence_head.pt.
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr   = token_pair_initial_repr

    for recycle_idx in range(num_trunk_recycles):
        (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
            move_to_device=device,
            token_single_trunk_initial_repr=token_single_initial_repr,
            token_pair_trunk_initial_repr=token_pair_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,   # recycled
            token_pair_trunk_repr=token_pair_trunk_repr,       # recycled
            msa_input_feats=msa_input_feats,
            msa_mask=msa_mask,
            template_input_feats=template_input_feats,
            template_input_masks=template_input_masks,
            token_single_mask=token_single_mask,
            token_pair_mask=token_pair_mask,
            crop_size=model_size,
        )

    # ── Short-circuit: discard everything except the trunk outputs ────────────
    del (
        token_single_initial_repr, token_pair_initial_repr,
        msa_input_feats, msa_mask,
        template_input_feats, template_input_masks,
        token_single_mask, token_pair_mask,
        batch, features, inputs,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Strip batch dim [1, L, ...] → [L, ...]
    single = token_single_trunk_repr.squeeze(0).cpu()   # [L, d_s]
    pair   = token_pair_trunk_repr.squeeze(0).cpu()     # [L, L, d_p]

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
    singles: List[torch.Tensor],   # each [L, d_s], CPU, float16
    pairs:   List[torch.Tensor],   # each [L, L, d_p], CPU, float16; empty if skip_pair
    frame_indices: List[int],
    skip_pair: bool,
) -> Path:
    """Stack buffered tensors and flush one HDF5 shard to disk.

    Shard datasets
    ──────────────
    single     float16  [N, L, d_s]       token single representations
    pair       float16  [N, L, L, d_p]    token pair representations (if not skip_pair)
    frame_idx  int32    [N]               original trajectory frame index
    """
    protein_dir.mkdir(parents=True, exist_ok=True)
    path = protein_dir / f"shard_{shard_idx:06d}.h5"

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

    log.info(
        "Shard %06d written: %d frames, %.1f MB → %s",
        shard_idx, len(singles), path.stat().st_size / 1e6, path,
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


def frames_already_done(out_dir: Path, protein: str) -> int:
    return sum(
        s["n_frames"]
        for s in read_manifest(out_dir).get(protein, {}).get("shards", [])
    )


def shards_already_done(out_dir: Path, protein: str) -> int:
    return len(read_manifest(out_dir).get(protein, {}).get("shards", []))


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
    path = write_shard(protein_dir, shard_idx, singles, pairs, frame_indices, skip_pair)
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
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load the four compiled modules ONCE — reused across all frames ────────
    # diffusion_module.pt and confidence_head.pt are intentionally NOT loaded.
    log.info("Loading Chai-1 compiled modules onto %s …", device)
    loaded_modules: Dict[str, torch.jit.ScriptModule] = {
        "feature_embedding":    load_exported("feature_embedding.pt",    device),
        "bond_loss_input_proj": load_exported("bond_loss_input_proj.pt", device),
        "token_embedder":       load_exported("token_embedder.pt",       device),
        "trunk":                load_exported("trunk.pt",                device),
    }
    log.info("Modules loaded: %s", list(loaded_modules.keys()))

    # ── FASTA-only mode (no trajectory) ──────────────────────────────────────
    if args.fasta:
        protein = Path(args.fasta).stem
        log.info("FASTA mode: encoding single sequence for %s", protein)
        context = _build_feature_context(Path(args.fasta))
        single, pair = run_trunk_pipeline(
            context, loaded_modules, device,
            num_trunk_recycles=args.num_trunk_recycles,
            low_memory=args.low_memory,
        )
        protein_dir = out_dir / protein
        _flush_shard(
            protein_dir, out_dir, protein, 0,
            singles=[single.half()],
            pairs=[] if args.skip_pair else [pair.half()],
            frame_indices=[0],
            skip_pair=args.skip_pair,
        )
        log.info("Done: single %s  pair %s", tuple(single.shape), tuple(pair.shape))
        return

    # ── MD trajectory mode ────────────────────────────────────────────────────
    protein = Path(args.pdb).stem
    log.info("MD mode: protein=%s  device=%s  recycles=%d  skip_pair=%s",
             protein, device, args.num_trunk_recycles, args.skip_pair)

    # Open trajectory lazily — DCD frames are never fully loaded into RAM
    u       = mda.Universe(args.pdb, args.dcd)
    n_total = len(u.trajectory)
    log.info("Trajectory: %d frames", n_total)

    # Select backbone atoms once; reuse AtomGroup across all frames
    backbone = u.select_atoms("protein and backbone")

    # ── Resume: skip frames already written ──────────────────────────────────
    n_done      = frames_already_done(out_dir, protein)
    shard_idx   = shards_already_done(out_dir, protein)
    start_frame = n_done

    if start_frame >= n_total:
        log.info("All %d frames already extracted — nothing to do.", n_total)
        return
    if start_frame > 0:
        log.info("Resuming from frame %d / %d (shard %d).", start_frame, n_total, shard_idx)

    # ── Build base AllAtomFeatureContext from PDB sequence ────────────────────
    # Written to a temp FASTA; chai-lab's context builder handles sequencing.
    sequence = _get_sequence(u)
    log.info("Sequence: %d residues", len(sequence))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", prefix=f"{protein}_", delete=False
    ) as fh:
        fasta_tmp = Path(fh.name)
        _write_fasta(sequence, protein, fasta_tmp)
        log.info("Temporary FASTA: %s", fasta_tmp)

    try:
        base_context = _build_feature_context(fasta_tmp)
    finally:
        fasta_tmp.unlink(missing_ok=True)

    # ── Storage estimate ──────────────────────────────────────────────────────
    if not args.skip_pair:
        L = len(sequence)
        d_p_est       = 128
        n_shards_rem  = (n_total - start_frame + args.shard_size - 1) // args.shard_size
        gb_per_shard  = (L * L * d_p_est * 2 * args.shard_size) / 1e9
        log.info(
            "Pair storage estimate: ~%.1f GB/shard × %d shards = ~%.0f GB.  "
            "Use --skip-pair or --shard-size 100 to reduce.",
            gb_per_shard, n_shards_rem, gb_per_shard * n_shards_rem,
        )

    # ── Pre-flight: validate pipeline on frame 0 ─────────────────────────────
    log.info("Pre-flight: validating trunk pipeline on frame 0 …")
    u.trajectory[0]
    test_ctx = _patch_atom_coords(base_context, backbone.positions.copy())
    s0, p0 = run_trunk_pipeline(
        test_ctx, loaded_modules, device,
        num_trunk_recycles=args.num_trunk_recycles,
        low_memory=args.low_memory,
    )
    assert s0.dim() == 2, f"Expected single [L, d_s], got {tuple(s0.shape)}"
    assert p0.dim() == 3, f"Expected pair [L, L, d_p], got {tuple(p0.shape)}"
    d_s, d_p = s0.shape[-1], p0.shape[-1]
    log.info("  single: %s   pair: %s   ✓", tuple(s0.shape), tuple(p0.shape))
    del s0, p0, test_ctx

    # ── Shard accumulation buffers ────────────────────────────────────────────
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
        # CRITICAL: .copy() before next ts.next() overwrites the MDAnalysis buffer
        positions = backbone.positions.copy()   # [N_bb, 3] Å

        # Shallow-copy context with this frame's coordinates injected
        frame_context = _patch_atom_coords(base_context, positions)

        single, pair = run_trunk_pipeline(
            frame_context, loaded_modules, device,
            num_trunk_recycles=args.num_trunk_recycles,
            low_memory=args.low_memory,
        )

        singles_buf.append(single.half())
        if not args.skip_pair:
            pairs_buf.append(pair.half())
        frame_idx_buf.append(int(ts.frame))
        frames_done += 1

        if len(singles_buf) >= args.shard_size:
            _flush_shard(
                protein_dir, out_dir, protein, shard_idx,
                singles_buf, pairs_buf, frame_idx_buf, args.skip_pair,
            )
            shard_idx += 1
            singles_buf.clear()
            pairs_buf.clear()
            frame_idx_buf.clear()

        fps = frames_done / max(time.perf_counter() - t0, 1e-9)
        pbar.set_postfix(fps=f"{fps:.1f}", shard=shard_idx, d_s=d_s, d_p=d_p)

    if singles_buf:
        _flush_shard(
            protein_dir, out_dir, protein, shard_idx,
            singles_buf, pairs_buf, frame_idx_buf, args.skip_pair,
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "Extraction complete: %d frames in %.1fs (%.2f fps), %d shards.",
        frames_done, elapsed, frames_done / max(elapsed, 1e-9), shard_idx + 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Chai-1 trunk latents from an ATLAS MD trajectory (chai-lab 0.6.1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--pdb",   help="Reference topology (.pdb).  Pair with --dcd.")
    src.add_argument("--fasta", help="FASTA file for single-sequence encoding (no trajectory).")

    p.add_argument("--dcd",        help="Trajectory file (.dcd).  Required when using --pdb.")
    p.add_argument("--out-dir",    required=True, help="Scratch root for HDF5 shards.")
    p.add_argument("--shard-size", type=int, default=1000,
                   help="Frames per HDF5 shard.  Reduce to 100 for L > 400 with pair reps.")
    p.add_argument("--device",     default="cuda",
                   help="PyTorch device string (cuda, cuda:1, cpu).")
    p.add_argument("--num-trunk-recycles", type=int, default=3,
                   help="Number of trunk recycle iterations (mirrors num_trunk_recycles in run_folding_on_context).")
    p.add_argument("--low-memory", action="store_true",
                   help="Keep tensors on CPU between modules; move to device per-call.  "
                        "Slower but uses less peak VRAM.")
    p.add_argument("--skip-pair", action="store_true",
                   help="Skip storing pair representations [L, L, d_p].  "
                        "Pair reps are O(L²) larger than single reps.")

    args = p.parse_args()

    if args.fasta is None and args.pdb is None:
        p.error("One of --pdb or --fasta is required.")
    if args.pdb and not args.dcd:
        p.error("--dcd is required when using --pdb.")

    return args


if __name__ == "__main__":
    args = parse_args()
    extract(args)
