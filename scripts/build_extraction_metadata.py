#!/usr/bin/env python3
"""
scripts/build_extraction_metadata.py

The raw ATLAS files used for latent extraction were purged from scratch and the ATLAS API is
unreachable, so the ground truth for "what did Chai-1 see" is the extraction-time logs
(logs/swarm_fixed_*.out). Parses them into one record per (protein, replica):

  n_residues        sequence length handed to Chai-1  ('[fasta] ... protein|<name> <len>')
  traj_frames       frames in the source trajectory   ('Trajectory: N frames')
  heavy_atoms / real_atoms   name-based atom correspondence  ('N Chai-1 heavy atoms matched to M real ...')
  single_shape / pair_shape  trunk output shapes
  frame_diff_0_500  frame-variance pre-flight value
  xtc / pdb         source file names

Asserts that every value is identical across all logs/resubmissions of the same (protein, replica),
and that n_residues is identical across replicas of a protein. Output: extraction_metadata.json
with the per-protein 'n_residues' that analysis/mask.py consumes.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("/u/arjunc4/fluxion/logs"))
    ap.add_argument("--glob", default="swarm_fixed_*.out")
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    args = ap.parse_args()
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
                  "frame_diff_0_vs", "pair_shape", "d_single"):
            vals = {json.dumps(r[k]) for r in recs if k in r}
            if len(vals) > 1:
                problems.append(f"{p} R{rep}: conflicting {k}: {sorted(vals)}")
            if vals:
                merged[k] = json.loads(sorted(vals)[0])
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
