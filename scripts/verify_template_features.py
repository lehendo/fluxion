#!/usr/bin/env python3
"""Numeric check of the template-feature reimplementation against the trunk's inputs (paper Appendix A.5)."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT
import argparse, json, sys, tempfile
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import torch
from fluxion.analysis.template_features import template_features
from fluxion.data_pipeline.extract_latents import _build_feature_context, _get_collator, _get_sequence, _write_fasta
from fluxion.data_pipeline.chai1_conditioning import build_atom_correspondence, inject_frame_conditioning

PROTEINS = ["1ab1_A", "1bq8_A", "1aol_A", "1b2s_E", "1ah7_A"]       # fixed in advance: two small, three with different composition (glycine content varies)
FRAMES = (0, 5000, 9999)
BINS = torch.linspace(3.25, 50.75, 38)[1:]


def pseudo_beta_and_backbone(u):
    sel = {nm: u.select_atoms(f"protein and name {nm}") for nm in ("N", "CA", "C")}
    cb = [r.atoms.select_atoms("name CB")[0] if len(r.atoms.select_atoms("name CB")) else r.atoms.select_atoms("name CA")[0] for r in u.select_atoms("protein").residues]
    n = len(sel["CA"]); assert len(sel["N"]) == n == len(sel["C"]) == len(cb)
    return sel["N"], sel["CA"], sel["C"], mda.AtomGroup(cb)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", type=Path, default=ATLAS_10PS)
    ap.add_argument("--proteins", nargs="*", default=PROTEINS); ap.add_argument("--frames", nargs="*", type=int, default=list(FRAMES)); ap.add_argument("--esm-cache", type=Path, default=DATA_ROOT / "verify_tf_esm_cache"); ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/template_feature_verification.json")
    a = ap.parse_args(); res = {}
    for p in a.proteins:
        d = a.raw / p; u = mda.Universe(str(d / f"{p}.pdb"), str(d / f"{p}_prod_R1_fit.xtc")); all_atoms = u.select_atoms("protein")
        groups = pseudo_beta_and_backbone(u); n_res = len(groups[1])
        with tempfile.NamedTemporaryFile("w", suffix=".fasta", prefix=f"{p}_", delete=False) as fh:
            fasta = Path(fh.name)
        _write_fasta(_get_sequence(u), p, fasta); base = _build_feature_context(fasta, a.esm_cache / p); fasta.unlink(missing_ok=True)
        c2m = build_atom_correspondence(all_atoms.resindices, all_atoms.names, base.structure_context)
        rows = []
        for f in a.frames:
            u.trajectory[f]; pos = all_atoms.positions.copy()
            ctx = inject_frame_conditioning(base, pos, c2m); tc = ctx.template_context
            batch = _get_collator()([ctx]); inp, feats = batch["inputs"], batch["features"]
            mine_d, mine_u = template_features(*[g.positions.copy()[None].astype(np.float32) for g in groups]); mine_d = mine_d.reshape(n_res, n_res); mine_u = mine_u.reshape(n_res, n_res, 3)
            td = tc.template_distances[0]; tu = tc.template_unit_vector[0]
            assert td.shape[0] == n_res, (td.shape, n_res)
            r = {"frame": f, "n_tokens": int(td.shape[0])}
            r["masks_all_true"] = bool(tc.template_pseudo_beta_mask.all() and tc.template_backbone_frame_mask.all())
            r["template_slots_used"] = int(inp["template_mask"][0].any(dim=-1).sum()); r["num_templates"] = int(inp["num_templates"][0])        # the feature has 4 slots; only slot 0 carries the self-template
            r["dist_max_abs_diff_A"] = float((td - mine_d).abs().max())                      # (1) pipeline's template_distances vs cdist of my pseudo-beta coordinates
            off = ~torch.eye(n_res, dtype=torch.bool)
            r["unit_max_abs_diff"] = float((tu - mine_u)[off].abs().max())                      # (2) off-diagonal (the diagonal is zero by the masks/eps in both)
            r["unit_diag_max_abs_pipeline"] = float(tu.diagonal(dim1=0, dim2=1).abs().max()); r["unit_diag_max_abs_mine"] = float(mine_u.diagonal(dim1=0, dim2=1).abs().max())
            r["template_feature_shapes"] = {k: [list(v.shape), str(v.dtype)] for k, v in feats.items() if "emplate" in k or "istogram" in k or "nit" in k}
            r["template_input_shapes"] = {k: list(v.shape) for k, v in inp.items() if "emplate" in k}
            print(r["template_feature_shapes"], r["template_input_shapes"], flush=True)
            tdk = [k for k in feats if k == "TemplateDistogram"]; tuk = [k for k in feats if k == "TemplateUnitVector"]
            r["feature_keys"] = {"distogram": tdk, "unit": tuk}
            if tdk:
                F = feats[tdk[0]]
                # the feature is [1, templ, tokens, tokens, 1] integer bin ids (one-hot encoded inside the feature embedding): compare to the bins of the recomputed distances
                ids = F[0, 0, :n_res, :n_res, 0] if F.dim() == 5 else F.reshape(-1)
                binned = torch.searchsorted(BINS, mine_d)
                r["distogram_feature_shape"] = list(F.shape); r["distogram_bin_mismatches"] = int((ids.long() != binned.long()).sum())
                if r["distogram_bin_mismatches"]:
                    mm = (ids.long() != binned.long()); edges = BINS[(binned[mm]).clamp(max=len(BINS) - 1)]
                    r["distogram_mismatch_max_dist_to_edge_A"] = float((mine_d[mm] - edges).abs().max())
            if tuk:
                U = feats[tuk[0]]; r["unit_feature_shape"] = list(U.shape)
                Uv = U[0, 0, :n_res, :n_res] if U.dim() == 5 else U
                r["unit_feature_max_abs_diff_vs_mine"] = float((Uv - mine_u)[off].abs().max())
            rows.append(r); print(p, r, flush=True)
        res[p] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
