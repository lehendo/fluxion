#!/usr/bin/env python3
"""
scripts/fetch_atlas_sequences.py

Re-fetches the ATLAS archives for the pilot proteins (the raw files under
/scratch/arjunc4/atlas_raw were purged by the cluster's scratch policy; only empty
directories remained) and records each protein's residue count and sequence from the
ATLAS topology, using the SAME residue->letter rule as data_pipeline/extract_latents.py
(`select_atoms("protein").residues`). This is the ground truth for how many REAL tokens
Chai-1 saw, so the padding mask can be derived from sequence length instead of being
inferred from zero variance.

Also leaves the extracted trajectories in place for Phase 1B (Cartesian control).
Output: <out-root>/atlas_refetch_sequences.json
"""
from __future__ import annotations
import argparse, datetime, json, sys, time, zipfile
from pathlib import Path
import requests
import MDAnalysis as mda

BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein"
PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def server_up(timeout: int = 45) -> bool:
    """The ATLAS main page answered while every api/ and database/ endpoint hung (2026-10-10), so probe a
    real data endpoint, not the home page."""
    try:
        r = requests.get(f"{BASE_URL}/1btk_A", headers={"Range": "bytes=0-255"}, stream=True, timeout=(15, timeout))
        ok = r.status_code in (200, 206)
        r.close()
        return ok
    except requests.RequestException:
        return False


def fetch(protein: str, dest: Path, meta: dict, wait_s: int, probe_every: int = 300) -> Path:
    """Waits (up to wait_s in total) for the server, then downloads with a long read timeout. Records the
    HTTP headers and fetch time: the ATLAS version record the verification step needs."""
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"{protein}.zip"
    if (dest / ".complete").exists():
        return dest
    t0 = time.time()
    while True:
        if server_up():
            try:
                with requests.get(f"{BASE_URL}/{protein}", stream=True, timeout=(30, 1800)) as r:
                    r.raise_for_status()
                    meta[protein] = {"fetched_utc": datetime.datetime.utcnow().isoformat() + "Z",
                                     "url": f"{BASE_URL}/{protein}",
                                     "headers": {k: v for k, v in r.headers.items()
                                                 if k.lower() in ("last-modified", "etag", "content-length", "date", "content-type")}}
                    with open(zpath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 22):
                            f.write(chunk)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(dest)
                zpath.unlink()
                (dest / ".complete").touch()
                return dest
            except (requests.RequestException, zipfile.BadZipFile) as e:
                print(f"[fetch] {protein}: attempt failed ({type(e).__name__}: {e})", flush=True)
        if time.time() - t0 > wait_s:
            raise TimeoutError(f"ATLAS API unreachable for {wait_s}s while fetching {protein}")
        print(f"[fetch] {protein}: server not answering; retrying in {probe_every}s", flush=True)
        time.sleep(probe_every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/scratch/arjunc4/atlas_raw"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/atlas_refetch_sequences.json"))
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    ap.add_argument("--wait-s", type=int, default=3 * 3600, help="give up after this long without a server")
    ap.add_argument("--meta-out", type=Path, default=Path("/scratch/arjunc4/fluxion_data/atlas_refetch_meta.json"))
    args = ap.parse_args()
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    meta = json.loads(args.meta_out.read_text()) if args.meta_out.exists() else {}
    for p in args.proteins:
        d = fetch(p, args.raw_dir / p, meta, args.wait_s)
        args.meta_out.write_text(json.dumps(meta, indent=1))
        pdbs = sorted(d.rglob("*.pdb"))
        pdb = next((x for x in pdbs if "_R" not in x.stem), pdbs[0])
        u = mda.Universe(str(pdb))
        res = u.select_atoms("protein").residues
        seq = "".join(THREE.get(r.resname.upper(), "X") for r in res)
        xtcs = sorted(str(x.relative_to(d)) for x in d.rglob("*.xtc"))
        out[p] = {"n_residues": len(res), "sequence": seq, "pdb": str(pdb.relative_to(d)),
                  "n_X_nonstandard": seq.count("X"), "xtc": xtcs}
        print(f"[fetch] {p}: {len(res)} residues ({seq.count('X')} nonstandard), "
              f"{len(xtcs)} xtc, pdb={pdb.name}", flush=True)
        args.out.write_text(json.dumps(out, indent=1))
    print("[fetch] done", flush=True)


if __name__ == "__main__":
    main()
