#!/usr/bin/env python3
"""Global alignment check of the re-fetched archives against the latents (paper Appendix A.4)."""
from __future__ import annotations
from fluxion.paths import ATLAS_ANALYSIS, DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import torch
from scipy.stats import spearmanr
from fluxion.analysis.geometry import kabsch_pairwise_rmsd
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

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
    ap.add_argument("--analysis-root", type=Path, default=ATLAS_ANALYSIS)
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--fetch-meta", type=Path, default=DATA_ROOT / "atlas_refetch_meta.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/atlas_verification_revised.json")
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    a = ap.parse_args()
    meta = json.loads(a.metadata_json.read_text())
    fetch_meta = json.loads(a.fetch_meta.read_text()) if a.fetch_meta.exists() else {}
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    report, failed = {}, []
    print(f"{'protein':8s} {'res':>11s} {'atoms':>13s} {'frames/dt':>10s} | {'aligned':>7s} {'R2':>6s} {'R3':>6s} | {'sh1/4':>6s} {'sh1/2':>6s} {'sh3/4':>6s} | verdicts")
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
        zs0 = torch.stack([ds._load_frame(sl[1][0] + 10 * k)[:n_res].flatten() for k in K_FRAMES]).double()
        zs = torch.roll(zs0, shifts=len(K_FRAMES) // 2, dims=0)
        rho["shift"] = float(spearmanr(rm, condensed(torch.cdist(zs, zs).numpy()))[0])
        # POST HOC revised control: within-replica circular shifts of 1/4, 1/2, 3/4 of the window
        for sh, name in ((len(K_FRAMES) // 4, "shift_q1"), (3 * len(K_FRAMES) // 4, "shift_q3")):
            zq = torch.roll(zs0, shifts=sh, dims=0)
            rho[name] = float(spearmanr(rm, condensed(torch.cdist(zq, zq).numpy()))[0])
        ctrl = max(rho["R2"], rho["R3"], rho["shift"])                                   # ORIGINAL (pre-registered) criterion
        ok_iii = rho["R1"] >= MIN_ALIGNED and rho["R1"] - ctrl >= MIN_MARGIN
        ctrl_rev = max(rho["shift_q1"], rho["shift"], rho["shift_q3"])                    # REVISED, within-replica only
        ok_iii_rev = rho["R1"] >= MIN_ALIGNED and rho["R1"] - ctrl_rev >= MIN_MARGIN
        ok_iii_half = rho["R1"] >= MIN_ALIGNED and rho["R1"] - rho["shift"] >= MIN_MARGIN  # revised, half-window shift only
        r["mantel"] = rho; r["pass_i"], r["pass_ii"], r["pass_iii"] = bool(ok_i), bool(ok_ii), bool(ok_iii)
        r["pass_iii_revised"], r["pass_iii_revised_halfonly"] = bool(ok_iii_rev), bool(ok_iii_half)
        r["pass"] = bool(ok_i and ok_ii and ok_iii); r["pass_revised"] = bool(ok_i and ok_ii and ok_iii_rev); report[p] = r
        if not r["pass"]: failed.append(p)
        print(f"{p:8s} {r['residues'][0]:5d}/{r['residues'][1]:<5d} {r['protein_atoms'][0]:6d}/{sorted(real_atoms)[0]:<6d} {'ok' if ok_ii else 'BAD':>10s} | "
              f"{rho['R1']:7.3f} {rho['R2']:6.3f} {rho['R3']:6.3f} | {rho['shift_q1']:6.3f} {rho['shift']:6.3f} {rho['shift_q3']:6.3f} | "
              f"orig {'PASS' if r['pass'] else 'FAIL'}  revised {'PASS' if r['pass_revised'] else 'FAIL'} (half-only {'PASS' if (ok_i and ok_ii and ok_iii_half) else 'FAIL'})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"per_protein": report, "failed": failed, "failed_revised": [q for q, v in report.items() if not v.get("pass_revised")], "criteria": {"min_aligned": MIN_ALIGNED, "min_margin": MIN_MARGIN}}, indent=1))
    print("\nATLAS RE-FETCH VERIFIED for all proteins" if not failed else f"\nNOT VERIFIED for: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
