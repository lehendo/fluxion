#!/usr/bin/env python3
"""
scripts/verify_atlas_refetch.py

Is the re-fetched ATLAS data frame-identical to what produced the latents? Ground truth = the extraction-time logs
(extraction_metadata.json) and the latent shards. Pre-registered in experiments/EXPERIMENT_LOG.md (2026-10-10).
Layout: <analysis-root>/<protein>/{<p>.pdb, <p>_R{1,2,3}.xtc}; xtc = 1001 frames at 100 ps; analysis frame k <-> latent frame 10k.

  (i)   PDB residue count == n_residues AND protein atom count == real_atoms (exact)
  (ii)  every replica: n_frames == 1001, dt == 100 ps, n_atoms == PDB atoms
  (iii) Mantel alignment, replica 1, 250 frames (k = 0,4,..,996): Spearman(pairwise C-alpha RMSD, pairwise latent distance)
        aligned (R1 coords vs R1 latents at 10k); negative controls: (a) R1 coords vs R2 and R3 latents, (b) R1 coords vs R1
        latents circularly shifted by half the window.  PASS iff aligned >= 0.20 and aligned - max(control) >= 0.15.
Exit code 1 on any failure; per-protein verdicts are written to JSON.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MDAnalysis as mda
import numpy as np
import torch
from scipy.stats import spearmanr
from analysis.geometry import kabsch_pairwise_rmsd
from data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
K_FRAMES = list(range(0, 1000, 4))                                 # 250 analysis frames
MIN_ALIGNED, MIN_MARGIN = 0.20, 0.15


def condensed(M):
    return M[np.triu_indices_from(M, k=1)]


def replica_slices(ds):
    out = {}
    for i, (path, _) in enumerate(ds.index_map):
        c = int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1)))
        out.setdefault(c, [i, i + 1])[1] = i + 1
    return {k: tuple(v) for k, v in out.items()}


def latent_pairwise(ds, start, n_res):
    z = torch.stack([ds._load_frame(start + 10 * k)[:n_res].flatten() for k in K_FRAMES]).double()
    return torch.cdist(z, z).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-root", type=Path, default=Path("/scratch/arjunc4/atlas_analysis"))
    ap.add_argument("--metadata-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--fetch-meta", type=Path, default=Path("/scratch/arjunc4/fluxion_data/atlas_refetch_meta.json"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/atlas_verification.json"))
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    a = ap.parse_args()
    meta = json.loads(a.metadata_json.read_text())
    fetch_meta = json.loads(a.fetch_meta.read_text()) if a.fetch_meta.exists() else {}
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    report, failed = {}, []
    print(f"{'protein':8s} {'res':>11s} {'atoms':>13s} {'frames/dt':>10s} | {'rho aligned':>11s} {'R2':>7s} {'R3':>7s} {'shift':>7s} | verdict")
    for p in a.proteins:
        d = a.analysis_root / p
        r: dict = {"fetch": fetch_meta.get(p, {}).get("headers")}
        if not (d / ".complete").exists():
            report[p] = {**r, "status": "NOT RE-FETCHED", "pass": False}; failed.append(p); print(f"{p:8s} NOT RE-FETCHED"); continue
        e = meta[p]
        pdb = next(x for x in sorted(d.rglob("*.pdb")) if "_R" not in x.stem)
        prot = mda.Universe(str(pdb)).select_atoms("protein")
        real_atoms = {v["real_atoms"] for v in e["replicas"].values()}
        r["residues"] = [len(prot.residues), e["n_residues"]]
        r["protein_atoms"] = [prot.n_atoms, sorted(real_atoms)]
        ok_i = r["residues"][0] == r["residues"][1] and {prot.n_atoms} == real_atoms
        coords, ok_ii = {}, True
        for rep in (1, 2, 3):
            t = mda.Universe(str(pdb), str(d / f"{p}_R{rep}.xtc")).trajectory
            r[f"R{rep}"] = {"n_frames": t.n_frames, "dt_ps": t.dt, "n_atoms": t.n_atoms}
            ok_ii &= (t.n_frames == 1001 and abs(t.dt - 100.0) < 1e-3 and t.n_atoms == len(mda.Universe(str(pdb)).atoms))
            if rep == 1:
                u = mda.Universe(str(pdb), str(d / f"{p}_R1.xtc")); ca = u.select_atoms("name CA")
                coords = np.stack([ca.positions.copy() for ts in u.trajectory[K_FRAMES]])
        ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
        sl = replica_slices(ds); n_res = e["n_residues"]
        rm = condensed(kabsch_pairwise_rmsd(coords))
        rho = {}
        for rep in (1, 2, 3):
            rho[f"R{rep}"] = float(spearmanr(rm, condensed(latent_pairwise(ds, sl[rep][0], n_res)))[0])
        zs = torch.stack([ds._load_frame(sl[1][0] + 10 * k)[:n_res].flatten() for k in K_FRAMES]).double()
        zs = torch.roll(zs, shifts=len(K_FRAMES) // 2, dims=0)
        rho["shift"] = float(spearmanr(rm, condensed(torch.cdist(zs, zs).numpy()))[0])
        ctrl = max(rho["R2"], rho["R3"], rho["shift"])
        ok_iii = rho["R1"] >= MIN_ALIGNED and rho["R1"] - ctrl >= MIN_MARGIN
        r["mantel"] = rho; r["pass_i"], r["pass_ii"], r["pass_iii"] = bool(ok_i), bool(ok_ii), bool(ok_iii)
        r["pass"] = bool(ok_i and ok_ii and ok_iii); report[p] = r
        if not r["pass"]: failed.append(p)
        print(f"{p:8s} {r['residues'][0]:5d}/{r['residues'][1]:<5d} {r['protein_atoms'][0]:6d}/{sorted(real_atoms)[0]:<6d} {'ok' if ok_ii else 'BAD':>10s} | "
              f"{rho['R1']:11.3f} {rho['R2']:7.3f} {rho['R3']:7.3f} {rho['shift']:7.3f} | {'PASS' if r['pass'] else 'FAIL (i=%s ii=%s iii=%s)' % (ok_i, ok_ii, ok_iii)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"per_protein": report, "failed": failed, "criteria": {"min_aligned": MIN_ALIGNED, "min_margin": MIN_MARGIN}}, indent=1))
    print("\nATLAS RE-FETCH VERIFIED for all proteins" if not failed else f"\nNOT VERIFIED for: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
