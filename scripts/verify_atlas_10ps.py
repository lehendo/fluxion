#!/usr/bin/env python3
"""
scripts/verify_atlas_10ps.py

Verification of the fetched full-stride (10 ps) ATLAS trajectories, pre-registered in the experiment log (2026-10-14, T2). A protein is usable for the 10 ps analyses only if ALL pass:
  (i)   PDB residue count and protein atom count equal the extraction-log values (exact);
  (ii)  every replica has 10,001 frames at dt = 10 ps with the PDB atom count;
  (iii) COORDINATE IDENTITY with the already verified 100 ps analysis archives: for 50 sampled analysis frames per replica, analysis frame k and 10 ps frame 10k superpose (Calpha, optimal)
        to RMSD < 0.05 A. Exact, not statistical: the 10 ps frames therefore inherit the earlier alignment verification of the analysis archives.
Layout: <full-root>/<protein>/ holds the extracted protein archive; <analysis-root>/<protein>/{<p>.pdb,<p>_R{1,2,3}.xtc} the 100 ps analysis archive.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MDAnalysis as mda
import numpy as np
from analysis.geometry import superposed_rmsd

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
RMSD_MAX, N_SAMPLE, N_FRAMES, DT_PS = 0.05, 50, 10001, 10.0


def find_files(d: Path, protein: str):
    pdbs = sorted(x for x in d.rglob("*.pdb") if "_R" not in x.stem and "prod" not in x.stem) or sorted(d.rglob("*.pdb"))
    xtc = {r: sorted(d.rglob(f"*R{r}*.xtc")) for r in (1, 2, 3)}
    return pdbs[0], {r: v[0] for r, v in xtc.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-root", type=Path, default=Path("/scratch/arjunc4/atlas_10ps"))
    ap.add_argument("--analysis-root", type=Path, default=Path("/scratch/arjunc4/atlas_analysis"))
    ap.add_argument("--metadata-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/atlas_10ps_verification.json"))
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    a = ap.parse_args()
    meta = json.loads(a.metadata_json.read_text())
    rng = np.random.default_rng(0)
    report, failed = {}, []
    print(f"{'protein':8s} res atoms | replicas frames dt | max superposed RMSD over {N_SAMPLE} frames/replica (A) | verdict")
    for p in a.proteins:
        d = a.full_root / p
        if not (d / ".complete").exists():
            report[p] = {"status": "NOT FETCHED", "pass": False}; failed.append(p); print(f"{p:8s} NOT FETCHED"); continue
        e = meta[p]; real_atoms = {v["real_atoms"] for v in e["replicas"].values()}
        pdb, xtcs = find_files(d, p)
        pu = mda.Universe(str(pdb)); prot = pu.select_atoms("protein")
        ok_i = len(prot.residues) == e["n_residues"] and {prot.n_atoms} == real_atoms
        ok_ii, ok_iii, worst, rows = len(xtcs) == 3, True, 0.0, {}
        apdb = next(x for x in sorted((a.analysis_root / p).rglob("*.pdb")) if "_R" not in x.stem)
        for r, xt in xtcs.items():
            u = mda.Universe(str(pdb), str(xt)); t = u.trajectory
            ok_ii &= (t.n_frames == N_FRAMES and abs(t.dt - DT_PS) < 1e-3 and t.n_atoms == len(pu.atoms))
            ua = mda.Universe(str(apdb), str(a.analysis_root / p / f"{p}_R{r}.xtc"))
            ca_f, ca_a = u.select_atoms("name CA"), ua.select_atoms("name CA")
            ks = sorted(rng.choice(len(ua.trajectory), size=min(N_SAMPLE, len(ua.trajectory)), replace=False).tolist())
            rm = []
            for k in ks:
                ua.trajectory[k]; xa = ca_a.positions.copy()
                u.trajectory[min(10 * k, t.n_frames - 1)]; xf = ca_f.positions.copy()
                rm.append(superposed_rmsd(xa, xf) if xa.shape == xf.shape else float("inf"))
            rows[r] = {"n_frames": t.n_frames, "dt_ps": t.dt, "max_rmsd": float(max(rm)), "mean_rmsd": float(np.mean(rm))}
            ok_iii &= max(rm) < RMSD_MAX; worst = max(worst, max(rm))
        ok = bool(ok_i and ok_ii and ok_iii)
        report[p] = {"residues": [len(prot.residues), e["n_residues"]], "protein_atoms": [prot.n_atoms, sorted(real_atoms)], "replicas": rows, "pass_i": bool(ok_i), "pass_ii": bool(ok_ii), "pass_iii": bool(ok_iii), "pass": ok}
        if not ok:
            failed.append(p)
        print(f"{p:8s} {len(prot.residues):4d} {prot.n_atoms:5d} | {len(xtcs)} {rows[1]['n_frames'] if 1 in rows else '-'} {rows[1]['dt_ps'] if 1 in rows else '-'} | {worst:8.4f} | {'PASS' if ok else 'FAIL (i=%s ii=%s iii=%s)' % (ok_i, ok_ii, ok_iii)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"per_protein": report, "failed": failed, "criteria": {"rmsd_max_A": RMSD_MAX, "n_sample": N_SAMPLE}}, indent=1))
    print("\nALL 19 VERIFIED" if not failed else f"\nFAILED / UNFETCHED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
