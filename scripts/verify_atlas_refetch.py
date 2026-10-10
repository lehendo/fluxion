#!/usr/bin/env python3
"""
scripts/verify_atlas_refetch.py

Is the re-fetched ATLAS data frame-identical to what produced the latents? Run only when raw files exist.
Ground truth = the extraction-time records (scripts/build_extraction_metadata.py) and the shard manifests.

  (i)  residues and ATOM COUNT: re-fetched PDB protein residue count == n_residues and total protein atoms
       == the logged 'real (incl. H) atoms' (e.g. 9686 for 1b25_D)
  (ii) xtc file names, frame count per replica == logged traj_frames; shard manifests cover frames 0..N-1
       contiguously; trajectory dt == 10 ps (the stride the latents assume)
  (iii) Mantel rho: re-run scripts/verify_latent_pairwise_sensitivity.py on --mantel-proteins and compare with
       the rho logged when the latents were originally validated (deterministic frame selection => should
       reproduce to ~1e-3; a frame-misaligned re-fetch would collapse it)
Any failed or impossible check => Phase 1B is CONDITIONAL on unverified data. Exit code 1 on any failure.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import MDAnalysis as mda

MANTEL_TOL = 1e-3


def parse_logged_rho(log: Path) -> dict[str, float]:
    out, cur = {}, None
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"PROTEIN: (\S+)", line)
        if m: cur = m.group(1)
        m = re.search(r"Spearman\(pairwise RMSD, pairwise latent dist\) = ([-\d.]+)", line)
        if m and cur: out[cur] = float(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/scratch/arjunc4/atlas_raw"))
    ap.add_argument("--metadata-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--logged-mantel", type=Path, default=Path("/u/arjunc4/fluxion/logs/verify_pairwise_sweep_10112972.out"))
    ap.add_argument("--mantel-proteins", nargs="+", default=["1btk_A", "1a62_A", "1bgf_A"])
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase1_results/atlas_verification.json"))
    a = ap.parse_args()
    meta = json.loads(a.metadata_json.read_text())
    report, failed = {}, []
    for p, e in sorted((k, v) for k, v in meta.items() if not k.startswith("_")):
        d = a.raw_dir / p
        if not (d / ".complete").exists():
            report[p] = {"status": "NOT RE-FETCHED"}; failed.append(p); continue
        r = {}
        pdb = next((x for x in sorted(d.rglob("*.pdb")) if "_R" not in x.stem), None)
        u = mda.Universe(str(pdb)); prot = u.select_atoms("protein")
        r["n_residues"] = (len(prot.residues), e["n_residues"])
        real_atoms = {v.get("real_atoms") for v in e["replicas"].values()}
        r["real_atoms"] = (prot.n_atoms, sorted(real_atoms))
        for rep, v in sorted(e["replicas"].items()):
            xtc = next((x for x in d.rglob("*.xtc") if x.name == v["xtc"]), None)
            if xtc is None: r[f"R{rep}"] = {"xtc_missing": v["xtc"]}; continue
            t = mda.Universe(str(pdb), str(xtc)).trajectory
            shards = json.loads((a.latents_dir / p / "manifest.json").read_text())[p]["shards"]
            mine = [s for s in shards if f"_R{rep}_" in s["path"]]
            seen = sorted({(s["first_idx"], s["last_idx"]) for s in mine})
            contiguous = seen[0][0] == 0 and all(b[0] == a_[1] + 1 for a_, b in zip(seen, seen[1:])) and seen[-1][1] + 1 == v["traj_frames"]
            r[f"R{rep}"] = {"n_frames": (t.n_frames, v["traj_frames"]), "dt_ps": t.dt, "manifest_contiguous_0_to_N": contiguous}
        ok = (r["n_residues"][0] == r["n_residues"][1] and set([r["real_atoms"][0]]) == set(r["real_atoms"][1]) and
              all(isinstance(v, dict) and v.get("n_frames", (0, 1))[0] == v.get("n_frames", (0, 1))[1]
                  and abs(v.get("dt_ps", 0) - 10.0) < 1e-3 and v.get("manifest_contiguous_0_to_N") for k, v in r.items() if k.startswith("R")))
        r["pass"] = bool(ok); report[p] = r
        if not ok: failed.append(p)
    logged = parse_logged_rho(a.logged_mantel)
    mant = {}
    for p in a.mantel_proteins:
        if not (a.raw_dir / p / ".complete").exists(): mant[p] = {"status": "NOT RE-FETCHED"}; failed.append(f"mantel:{p}"); continue
        out = subprocess.run([sys.executable, "scripts/verify_latent_pairwise_sensitivity.py", "--latents-dir", str(a.latents_dir),
                              "--atlas-raw", str(a.raw_dir), "--protein-id", p, "--n-load", "1500", "--n-sample", "50"],
                             capture_output=True, text=True).stdout
        m = re.search(r"Spearman\(pairwise RMSD, pairwise latent dist\) = ([-\d.]+)", out)
        new = float(m.group(1)) if m else None
        old = logged.get(p)
        ok = new is not None and old is not None and abs(new - old) <= MANTEL_TOL
        mant[p] = {"logged": old, "re_fetched": new, "pass": ok}
        if not ok: failed.append(f"mantel:{p}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"per_protein": report, "mantel": mant, "failed": failed}, indent=1))
    print(json.dumps({"mantel": mant, "failed": failed}, indent=1))
    print("ATLAS RE-FETCH VERIFIED" if not failed else f"NOT VERIFIED -> Phase 1B conditional ({len(failed)} failures/gaps)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
