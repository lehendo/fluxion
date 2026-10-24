#!/usr/bin/env python3
"""Re-fetches the ATLAS analysis archives (sequences) for the 19 proteins, the input to the alignment checks (paper Appendix A.4)."""
from __future__ import annotations
from fluxion.paths import ATLAS_ANALYSIS, DATA_ROOT
import os
import argparse, datetime, json, time, zipfile
from pathlib import Path
import requests
import MDAnalysis as mda

BASE = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS"
HEADERS = {"User-Agent": "Mozilla/5.0 (fluxion research; contact: " + os.environ.get("FLUXION_CONTACT", "unspecified") + ")"}
PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def fetch(protein: str, dest: Path, endpoint: str, meta: dict, tries: int = 4) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"{protein}_{endpoint}.zip"
    if (dest / ".complete").exists():
        return dest
    for attempt in range(1, tries + 1):
        try:
            with requests.get(f"{BASE}/{endpoint}/{protein}", headers=HEADERS, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                meta[protein] = {"fetched_utc": datetime.datetime.utcnow().isoformat() + "Z", "endpoint": endpoint,
                                 "headers": {k: v for k, v in r.headers.items()
                                             if k.lower() in ("last-modified", "etag", "content-length", "date", "content-disposition")}}
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        f.write(chunk)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dest)
            zpath.unlink()
            (dest / ".complete").touch()
            return dest
        except (requests.RequestException, zipfile.BadZipFile) as e:
            print(f"[fetch] {protein}: attempt {attempt}/{tries} failed ({type(e).__name__}: {e})", flush=True)
            time.sleep(10 * attempt)
    raise RuntimeError(f"could not fetch {endpoint}/{protein}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ATLAS_ANALYSIS)
    ap.add_argument("--endpoint", default="analysis", choices=["analysis", "protein", "total"])
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    ap.add_argument("--meta-out", type=Path, default=DATA_ROOT / "atlas_refetch_meta.json")
    ap.add_argument("--seq-out", type=Path, default=DATA_ROOT / "atlas_refetch_sequences.json")
    a = ap.parse_args()
    meta = json.loads(a.meta_out.read_text()) if a.meta_out.exists() else {}
    seqs = json.loads(a.seq_out.read_text()) if a.seq_out.exists() else {}
    for p in a.proteins:
        d = fetch(p, a.out_root / p, a.endpoint, meta)
        a.meta_out.write_text(json.dumps(meta, indent=1))
        pdb = next(x for x in sorted(d.rglob("*.pdb")) if "_R" not in x.stem)
        prot = mda.Universe(str(pdb)).select_atoms("protein")
        seqs[p] = {"n_residues": len(prot.residues), "n_protein_atoms": prot.n_atoms, "pdb": pdb.name}
        a.seq_out.write_text(json.dumps(seqs, indent=1))
        print(f"[fetch] {p}: {len(prot.residues)} residues, {prot.n_atoms} protein atoms, "
              f"last-modified {meta[p]['headers'].get('last-modified') if p in meta else 'cached'}", flush=True)
    print("[fetch] done", flush=True)


if __name__ == "__main__":
    main()
