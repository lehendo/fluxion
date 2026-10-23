"""
data_pipeline/chai1_conditioning.py

Fix for the encoder-inertness bug diagnosed in a code review of the extraction pipeline and confirmed both by source audit and empirically
(scripts/verify_latent_frame_sensitivity.py): extract_latents.py's per-frame
coordinate injection was inert. Only two of Chai-1's registered feature
generators read atom_gt_coords at all (TokenDistanceRestraint,
TokenPairPocketRestraint), and both fall through to a constant fill unless a
constraint_path is supplied, which extraction never does. TokenPairDistance,
the one generator that would genuinely use raw coordinates, isn't even
registered. The one live, registered, coordinate-dependent path into the
frozen trunk is the TEMPLATE feature set (TemplateMask/UnitVector/ResType/
DistogramGenerator) -- template_pseudo_beta_distances and the unit vectors
are computed directly from a template's atom_gt_coords
(chai_lab/data/dataset/templates/load.py).

This module has two responsibilities:

1. build_atom_correspondence(): a name-based mapping from Chai-1's tokenized
   heavy-atom order to the real structure's atom order (Sec 3 of the
   review). The previous code (_patch_atom_coords, removed) assumed
   positional truncate/pad correspondence, which is wrong whenever atom
   counts or orderings differ -- and they always do here, since ATLAS PDBs
   include hydrogens (confirmed: 1006 protein atoms for 1bxy_A) while
   Chai-1's tokenizer is heavy-atom only (476 for the same protein).
   Verified empirically on real data (2026-08-13): 476/476 atoms matched by
   (residue-sequence-rank, atom-name), 3 via a documented GROMACS-vs-PDB
   naming fallback (Ile's CD1/CD, C-terminal O|OXT vs OT1|OT2). Raises
   ValueError -- not a silent reshape -- on any unmatched atom.

2. build_self_template_context(): constructs a genuine per-frame
   TemplateContext by treating "this MD frame's own structure" as its own
   template, with a trivial identity alignment (template token i == query
   token i -- no cropping, no cross-structure indices needed). This
   reimplements the tensor math LoadedTemplate's properties use internally,
   but skips the entire homology-search / TemplateHit / cross-structure
   alignment machinery those properties exist for, since none of it applies
   to a self-template. Verified empirically: two frames of the same protein
   (frame 0 vs. frame 500 of 1bxy_A, 0.71 A mean / 7.39 A max real Cartesian
   displacement) produce template_distances differing by mean=0.62 A,
   max=4.66 A -- a physically sensible magnitude, confirming real
   conformational information now reaches the trunk through this feature.

Chai-1 imports are deferred to inside function bodies (not at module level)
so this module can still be imported where chai_lab isn't installed --
matching the lazy-import convention used for MDAnalysis elsewhere in this
project (see inference/visualize.py).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext
    from chai_lab.data.dataset.structure.all_atom_structure_context import (
        AllAtomStructureContext,
    )
    from chai_lab.data.dataset.templates.context import TemplateContext

# GROMACS (used to produce the ATLAS trajectories) names a small number of
# atoms differently from the PDB/CCD-standard names Chai-1's tokenizer
# expects. Tried only as a fallback AFTER the direct name match at that
# residue fails, so this cannot mask a genuinely wrong correspondence for
# the common case -- it only fires for these specific, documented atoms.
GROMACS_ATOM_NAME_FALLBACKS: dict[str, list[str]] = {
    "CD1": ["CD"],    # Ile's sole delta carbon: GROMACS "CD" vs PDB "CD1"
    "O": ["OT1"],     # C-terminal residue backbone carbonyl O
    "OXT": ["OT2"],   # C-terminal residue terminal carboxyl O
}


def decode_atom_name(row: torch.Tensor) -> str:
    """Inverse of chai_lab's _atom_names_to_tensor: ord(c)-32 per char, space-padded to 4."""
    return "".join(chr(int(v) + 32) for v in row).rstrip()


def build_atom_correspondence(
    resindices: np.ndarray,          # [n_mda_atoms] MDAnalysis AtomGroup.resindices
    atom_names: "np.ndarray | list[str]",  # [n_mda_atoms] MDAnalysis AtomGroup.names
    structure_context: "AllAtomStructureContext",
) -> torch.Tensor:
    """
    Build a LongTensor[n_chai_atoms] mapping each Chai-1 tokenized heavy atom
    to the corresponding real atom's local index within the MDAnalysis
    AtomGroup that `resindices`/`atom_names` came from (i.e. the same
    AtomGroup whose `.positions` is read per-frame).

    Joined on (residue sequence-rank, atom name) rather than absolute PDB
    residue numbers, so this is robust to numbering gaps or insertion
    codes -- both sides are derived from the same protein residue sequence,
    walked in the same order, since Chai-1's FASTA input is built from
    exactly this same MDAnalysis selection (see extract_latents.py's
    _get_sequence).

    Raises:
        ValueError: if any Chai-1 atom slot has no match, real or via
        GROMACS_ATOM_NAME_FALLBACKS. This is intentionally a hard failure,
        not a silent truncate/pad -- see this module's docstring.
    """
    resindices = np.asarray(resindices)
    unique_res = np.unique(resindices)
    rank_of_resindex = {int(r): i for i, r in enumerate(unique_res)}

    mda_lookup: dict[tuple[int, str], int] = {}
    for local_idx, (rix, name) in enumerate(zip(resindices, atom_names)):
        rank = rank_of_resindex[int(rix)]
        mda_lookup[(rank, str(name))] = local_idx

    atom_residue_rank = structure_context.token_residue_index[
        structure_context.atom_token_index
    ]
    n_atoms = structure_context.atom_gt_coords.shape[0]
    mapping = torch.empty(n_atoms, dtype=torch.long)
    missing: list[tuple[int, int, str]] = []

    for i in range(n_atoms):
        rank = int(atom_residue_rank[i])
        name = decode_atom_name(structure_context.atom_ref_name_chars[i])
        key = (rank, name)
        if key in mda_lookup:
            mapping[i] = mda_lookup[key]
            continue
        matched = False
        for alt in GROMACS_ATOM_NAME_FALLBACKS.get(name, []):
            alt_key = (rank, alt)
            if alt_key in mda_lookup:
                mapping[i] = mda_lookup[alt_key]
                matched = True
                break
        if not matched:
            missing.append((i, rank, name))

    if missing:
        raise ValueError(
            f"{len(missing)}/{n_atoms} Chai-1 tokenized heavy atoms have no "
            f"name-matched counterpart in the real structure -- refusing to "
            f"silently truncate/pad (that was the bug). Unmatched "
            f"(chai_atom_idx, residue_rank, atom_name): {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return mapping


def gather_frame_coords(positions: np.ndarray, chai_to_mda: torch.Tensor) -> torch.Tensor:
    """positions: [n_mda_atoms, 3] float32 (Angstrom) from one MD frame."""
    return torch.from_numpy(np.asarray(positions)[chai_to_mda.numpy()]).float()


def build_self_template_context(
    structure_context: "AllAtomStructureContext",
) -> "TemplateContext":
    """
    Build a single-template TemplateContext directly from a structure_context
    whose atom_gt_coords hold REAL per-frame coordinates: use "this frame's
    own structure" as its own template, with a trivial identity alignment
    (template token i == query token i). This is the fix for the inert-conditioning bug -- it routes real per-frame conformational
    information into Chai-1's frozen trunk via the one live feature path
    that actually reads coordinates.

    Reimplements the math chai_lab.data.dataset.templates.load.LoadedTemplate's
    properties compute internally, calling the SAME low-level helpers
    (get_token_reference_atom_positions_and_mask, cdist,
    Rigid.make_transform_from_reference) so the numerics match chai_lab's own
    real-template code path exactly -- but skips LoadedTemplate/TemplateHit
    entirely, since that machinery exists for cross-structure homology-search
    alignment, which is a no-op (identity) for a self-template.
    """
    from chai_lab.data.dataset.templates.context import TemplateContext
    from chai_lab.data.features.token_utils import (
        get_token_reference_atom_positions_and_mask,
    )
    from chai_lab.tools.rigid import Rigid
    from chai_lab.utils.tensor_utils import cdist

    atom_pos = structure_context.atom_gt_coords.unsqueeze(0)
    atom_mask = structure_context.atom_exists_mask.unsqueeze(0)
    token_exists = structure_context.token_exists_mask.unsqueeze(0)

    pb_pos, pb_mask = get_token_reference_atom_positions_and_mask(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        token_reference_atom_index=structure_context.token_ref_atom_index.unsqueeze(0),
        token_exists_mask=token_exists,
    )
    distances = cdist(pb_pos).squeeze(0)
    mask_block = pb_mask.squeeze(0)[:, None] & pb_mask.squeeze(0)[None, :]
    # Masked entries pushed to 100.0 (matches LoadedTemplate.template_pseudo_beta_distances):
    # above the one-hot encoding's 50.75 A cap, so they land in the "too far" bucket.
    distances = distances.masked_fill(~mask_block, 100.0)

    n_idx, ca_idx, c_idx = torch.split(
        structure_context.token_backbone_frame_index, 1, dim=-1
    )
    backbone_mask = structure_context.token_backbone_frame_mask.unsqueeze(0)
    n_pos, _ = get_token_reference_atom_positions_and_mask(
        atom_pos=atom_pos, atom_mask=atom_mask,
        token_reference_atom_index=n_idx.squeeze(-1).unsqueeze(0),
        token_exists_mask=backbone_mask,
    )
    ca_pos, _ = get_token_reference_atom_positions_and_mask(
        atom_pos=atom_pos, atom_mask=atom_mask,
        token_reference_atom_index=ca_idx.squeeze(-1).unsqueeze(0),
        token_exists_mask=backbone_mask,
    )
    c_pos, _ = get_token_reference_atom_positions_and_mask(
        atom_pos=atom_pos, atom_mask=atom_mask,
        token_reference_atom_index=c_idx.squeeze(-1).unsqueeze(0),
        token_exists_mask=backbone_mask,
    )
    rigids = Rigid.make_transform_from_reference(
        n_xyz=n_pos, ca_xyz=ca_pos, c_xyz=c_pos, eps=1e-12
    )
    points = rigids.get_trans()
    rigid_vec = rigids[..., None].invert_apply(points)
    inv_dist = torch.rsqrt(1e-12 + torch.sum(rigid_vec**2, dim=-1))
    bb_mask2d = backbone_mask.squeeze(0)[:, None] & backbone_mask.squeeze(0)[None, :]
    inv_dist = inv_dist.squeeze(0) * bb_mask2d
    unit_vector = rigid_vec.squeeze(0) * inv_dist.unsqueeze(-1)

    return TemplateContext(
        template_restype=structure_context.token_residue_type.unsqueeze(0),
        template_pseudo_beta_mask=pb_mask,
        template_backbone_frame_mask=backbone_mask,
        template_distances=distances.unsqueeze(0),
        template_unit_vector=unit_vector.unsqueeze(0),
    )


def inject_frame_conditioning(
    context: "AllAtomFeatureContext",
    positions: np.ndarray,
    chai_to_mda: torch.Tensor,
) -> "AllAtomFeatureContext":
    """
    Replaces both extract_latents.py's old _patch_atom_coords AND the
    previously-always-empty template context. For one MD frame:
      1. Gather this frame's real coordinates into Chai-1's tokenized atom
         order via `chai_to_mda` (see build_atom_correspondence).
      2. Write them into structure_context.atom_gt_coords.
      3. Build a genuine self-template from those coordinates and attach it
         as context.template_context, replacing TemplateContext.empty(...).

    Both steps are necessary together: (1) alone puts the right coordinates
    in atom_gt_coords, but nothing reads that field on the live inference
    path unless (3) also routes it through templates.
    """
    frame_coords = gather_frame_coords(positions, chai_to_mda)
    new_structure_context = dataclasses.replace(
        context.structure_context, atom_gt_coords=frame_coords
    )
    new_template_context = build_self_template_context(new_structure_context)
    return dataclasses.replace(
        context,
        structure_context=new_structure_context,
        template_context=new_template_context,
    )
