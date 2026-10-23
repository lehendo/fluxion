#!/usr/bin/env python3
"""Builds extraction_metadata.json (residue counts, buckets, replicas), the input table every analysis reads, from extraction job logs or directly from the latent shards."""
from __future__ import annotations
from fluxion.paths import ATLAS_10PS, DATA_ROOT, LOG_DIR
import argparse, json, re
from collections import defaultdict
from pathlib import Path

FLOAT_REL_TOL = 0.05   # resubmissions on different GPUs differ by ~1%; >5% would be a real disagreement

PAT = {
    "worker": re.compile(r"Worker \d+: (\S+) replica (\d+)"),
    "pdb": re.compile(r"^\s*PDB: (\S+)"),
    "xtc": re.compile(r"^\s*XTC: (\S+)"),
    "fasta": re.compile(r"\[fasta\] \[[^\]]*\] protein\|(\S+) (\d+)"),
    "frames": re.compile(r"Trajectory: (\d+) frames"),
    "atoms": re.compile(r"(\d+) Chai-1 heavy atoms matched to (\d+) real"),
    "shapes": re.compile(r"single: \((\d+), (\d+)\)\s+pair: \((\d+), (\d+), (\d+)\)"),
    "fvar": re.compile(r"frame-variance check: frame 0 vs frame (\d+), single mean\|diff\|=([\d.]+)"),
}


def parse(path: Path) -> dict | None:
    rec: dict = {}
    for line in path.read_text(errors="replace").splitlines():
        for k, p in PAT.items():
            m = p.search(line)
            if not m:
                continue
            g = m.groups()
            if k == "worker": rec["protein"], rec["replica"] = g[0], int(g[1])
            elif k == "pdb": rec["pdb"] = Path(g[0]).name
            elif k == "xtc": rec["xtc"] = Path(g[0]).name
            elif k == "fasta": rec["n_residues"] = int(g[1])
            elif k == "frames": rec["traj_frames"] = int(g[0])
            elif k == "atoms": rec["heavy_atoms"], rec["real_atoms"] = int(g[0]), int(g[1])
            elif k == "shapes": rec["L_bucket"], rec["d_single"], rec["pair_shape"] = int(g[0]), int(g[1]), [int(x) for x in g[2:]]
            elif k == "fvar": rec["frame_diff_frame"], rec["frame_diff_0_vs"] = int(g[0]), float(g[1])
    need = {"protein", "replica", "n_residues", "traj_frames", "L_bucket"}
    return rec if need <= rec.keys() else None


def from_latents(latents_dir: Path, pdb_root: Path) -> dict:
    """Metadata without job logs: replicas, frame counts and L_bucket from the shards, n_residues from the protein's PDB file (<pdb_root>/<protein>/<protein>.pdb)."""
    import h5py
    import MDAnalysis as mda
    out: dict = {}
    for pdir in sorted(d for d in latents_dir.iterdir() if d.is_dir()):
        p = pdir.name
        frames: dict[int, set] = defaultdict(set); L = d = None
        for f in sorted(pdir.rglob("*_shard_*.h5")):
            m = re.search(r"_R(\d+)_shard_", f.name)
            if not m: continue
            with h5py.File(f, "r") as h:
                frames[int(m.group(1))].update(int(i) for i in h["frame_idx"][:]); L, d = int(h.attrs["L"]), int(h.attrs["d_single"])
        if not frames: continue
        pdb = pdb_root / p / f"{p}.pdb"
        n_res = int(mda.Universe(str(pdb)).select_atoms("protein").residues.n_residues)
        out[p] = {"replicas": {str(r): {"n_residues": n_res, "traj_frames": len(fr), "L_bucket": L, "d_single": d} for r, fr in sorted(frames.items())}, "n_residues": n_res, "L_bucket": L}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=LOG_DIR)
    ap.add_argument("--glob", default="swarm_fixed_*.out")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--from-latents", type=Path, default=None, help="build from the shards in this latents directory instead of from job logs (needs --pdb-root)")
    ap.add_argument("--pdb-root", type=Path, default=ATLAS_10PS, help="directory with <protein>/<protein>.pdb, used by --from-latents")
    args = ap.parse_args()
    if args.from_latents is not None:
        meta = from_latents(args.from_latents, args.pdb_root)
        args.out.write_text(json.dumps({"_source": f"shards under {args.from_latents.name}", "_problems": [], **meta}, indent=1))
        print(f"wrote {args.out}: {len(meta)} proteins")
        return
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    files = sorted(args.logs.glob(args.glob))
    for f in files:
        r = parse(f)
        if r:
            r["log"] = f.name
            by_key[(r["protein"], r["replica"])].append(r)
    out: dict = {}
    problems: list[str] = []
    for (p, rep), recs in sorted(by_key.items()):
        merged: dict = {}
        for k in ("n_residues", "traj_frames", "L_bucket", "heavy_atoms", "real_atoms", "pdb", "xtc",
                  "pair_shape", "d_single"):
            vals = {json.dumps(r[k]) for r in recs if k in r}
            if len(vals) > 1:                       # exact-match fields: any disagreement is a problem
                problems.append(f"{p} R{rep}: conflicting {k}: {sorted(vals)}")
            if vals:
                merged[k] = json.loads(sorted(vals)[0])
        fv = [r["frame_diff_0_vs"] for r in recs if "frame_diff_0_vs" in r]
        if fv:                                      # float fingerprint: GPU nondeterminism => tolerance
            merged["frame_diff_0_vs"] = sum(fv) / len(fv)
            merged["frame_diff_0_vs_spread_rel"] = (max(fv) - min(fv)) / max(max(fv), 1e-9)
            if merged["frame_diff_0_vs_spread_rel"] > FLOAT_REL_TOL:
                problems.append(f"{p} R{rep}: frame_diff spread {merged['frame_diff_0_vs_spread_rel']:.3f} > {FLOAT_REL_TOL}")
        merged["n_logs"] = len(recs)
        out.setdefault(p, {"replicas": {}})["replicas"][str(rep)] = merged
    for p, e in out.items():
        for k in ("n_residues", "L_bucket"):
            vals = {r[k] for r in e["replicas"].values() if k in r}
            if len(vals) > 1:
                problems.append(f"{p}: {k} differs across replicas: {sorted(vals)}")
            e[k] = min(vals)
    args.out.write_text(json.dumps({"_source": f"{len(files)} logs matching {args.glob}", "_problems": problems, **out}, indent=1))
    print(f"parsed {len(files)} logs -> {sum(len(e['replicas']) for e in out.values())} (protein,replica) records, "
          f"{len(out)} proteins; conflicts: {len(problems)}")
    for p in problems: print("  PROBLEM:", p)
    print(f"{'protein':8s} {'n_res':>6s} {'L_bucket':>8s} {'replicas':>8s} {'frames':>14s} {'atoms heavy/real':>18s} {'|diff| f0 vs f500':>18s}")
    for p, e in sorted(out.items()):
        fr = sorted({r.get('traj_frames') for r in e['replicas'].values()})
        ha = sorted({(r.get('heavy_atoms'), r.get('real_atoms')) for r in e['replicas'].values()})
        fv = sorted({r.get('frame_diff_0_vs') for r in e['replicas'].values()})
        print(f"{p:8s} {e['n_residues']:6d} {e['L_bucket']:8d} {len(e['replicas']):8d} {str(fr):>14s} {str(ha):>18s} {str(fv):>18s}")


if __name__ == "__main__":
    main()
