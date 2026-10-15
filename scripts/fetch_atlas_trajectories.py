#!/usr/bin/env python3
"""
scripts/fetch_atlas_trajectories.py

Resumable, chunked download of the full-stride (10 ps) ATLAS protein trajectory archives, with integrity checks.

Endpoint: <base>/ATLAS/protein/<pdb_chain>  (a static zip holding the PDB and 3 replica XTCs; supports HTTP Range).
Why chunked: a single streaming request through the default client path timed out at the gateway earlier; HTTP/1.1 range requests return immediately
and a partial file can be resumed after any failure.

Per protein: ranged GET 0-0 gives the total size (Content-Range), 32 MiB chunks are appended to <name>.zip.part (resuming from its current size),
the finished size must equal the advertised total, zipfile CRC verification must pass, the SHA-256 and the response headers are recorded, and the archive
is extracted into its own new directory. Nothing is executed from the archive.

Data are ATLAS (CC BY-NC 4.0; Vander Meersche et al., Nucleic Acids Research 52(D1):D384-D392, 2024) and stay outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path

import requests

BASE = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein"
UA = {"User-Agent": "Mozilla/5.0"}
PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def total_size(url: str, session: requests.Session, timeout: float = 120) -> tuple[int, dict]:
    r = session.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=timeout)
    if r.status_code != 206:
        raise RuntimeError(f"{url}: expected 206 Partial Content, got {r.status_code} (range requests unsupported?)")
    m = re.match(r"bytes 0-0/(\d+)$", r.headers.get("Content-Range", ""))
    if not m:
        raise RuntimeError(f"{url}: unreadable Content-Range {r.headers.get('Content-Range')!r}")
    return int(m.group(1)), {k: v for k, v in r.headers.items() if k.lower() in ("last-modified", "date", "content-disposition", "etag", "server")}


def download(url: str, dest: Path, chunk: int = 32 * 2**20, retries: int = 8, session: requests.Session | None = None, timeout: float = 300) -> dict:
    """Download `url` to `dest` in ranged chunks, resuming from dest.part. Returns metadata. Raises on size mismatch."""
    session = session or requests.Session()
    total, hdrs = total_size(url, session)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > total:                                    # a stale/longer partial file cannot be a prefix of this resource
        part.unlink(); have = 0
    t0 = time.time()
    with open(part, "ab") as f:
        while have < total:
            end = min(have + chunk, total) - 1
            for attempt in range(retries):
                try:
                    r = session.get(url, headers={**UA, "Range": f"bytes={have}-{end}"}, timeout=timeout)
                    if r.status_code != 206 or len(r.content) != end - have + 1:
                        raise RuntimeError(f"bad chunk: status {r.status_code}, {len(r.content)} bytes for range {have}-{end}")
                    f.write(r.content); f.flush()
                    have = end + 1
                    break
                except Exception as e:                  # network error or short read: back off and retry the SAME range
                    if attempt == retries - 1:
                        raise
                    time.sleep(min(60, 2 ** attempt))
    if part.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: downloaded {part.stat().st_size} bytes, advertised {total}")
    part.rename(dest)
    return {"bytes": total, "seconds": round(time.time() - t0, 1), "headers": hdrs}


def verify_and_extract(zip_path: Path, out_dir: Path) -> dict:
    """CRC-check every member, then extract into a NEW directory. Raises on corruption."""
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"{zip_path.name}: corrupt member {bad}")
        names = z.namelist()
        out_dir.mkdir(parents=True, exist_ok=False)
        z.extractall(out_dir)
    return {"members": names}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("/scratch/arjunc4/atlas_10ps"))
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--chunk-mb", type=int, default=32)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = a.out_dir / "fetch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    s = requests.Session()
    for p in a.proteins:
        d = a.out_dir / p
        if (d / ".complete").exists():
            print(f"[fetch] {p}: already complete", flush=True); continue
        info = download(f"{a.base}/{p}", a.out_dir / f"{p}_protein.zip", chunk=a.chunk_mb * 2**20, session=s)
        zp = a.out_dir / f"{p}_protein.zip"
        info["sha256"] = sha256(zp)
        if d.exists():                                  # an interrupted extraction: start the directory over
            import shutil; shutil.rmtree(d)
        info.update(verify_and_extract(zp, d))
        (d / ".complete").write_text("ok")
        meta[p] = info
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"[fetch] {p}: {info['bytes'] / 1e6:.0f} MB in {info['seconds']} s, members {info['members']}", flush=True)
    print("[fetch] all done", flush=True)


if __name__ == "__main__":
    main()
