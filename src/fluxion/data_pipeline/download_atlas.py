"""
data_pipeline/download_atlas.py

Downloads ATLAS MD trajectory archives from the DSIMB REST API using aria2c.
Each protein is downloaded as a ZIP, extracted, and organized under OUT_DIR.

Usage
-----
# Show all available ATLAS protein IDs
python data_pipeline/download_atlas.py --list

# Download the curated 20-protein pilot set
python data_pipeline/download_atlas.py --pilot

# Download specific proteins
python data_pipeline/download_atlas.py --ids 1AKE_A 4LZT_A 1VJG_A

# Download everything (large — check space first)
python data_pipeline/download_atlas.py --all

Output layout
-------------
$FLUXION_ATLAS_RAW/
    1AKE_A/
        <extracted ZIP contents>
    4LZT_A/
        <extracted ZIP contents>
    ...
"""

from __future__ import annotations
from fluxion.paths import ATLAS_RAW

import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL  = "https://www.dsimb.inserm.fr/ATLAS/api"
OUT_DIR   = ATLAS_RAW

# Curated pilot set: well-characterized proteins with documented dynamics
# and published experimental validation data (NMR, HDX-MS, smFRET).
# These are strong candidates for the hero experiments.
PILOT_SET = [
    "16pk_A",
    "1a62_A",
    "1ab1_A",
    "1af7_A",
    "1ah7_A",
    "1ail_A",
    "1aol_A",
    "1b0n_A",
    "1b25_D",
    "1b2s_E",
    "1b2s_F",
    "1b8d_K",
    "1bcp_F",
    "1bgf_A",
    "1bkp_A",
    "1bq8_A",
    "1btk_A",
    "1bx7_A",
    "1bxy_A",
    "1bxy_B",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_protein_list() -> list[str]:
    """Download and parse the ATLAS parsable ZIP to get all protein chain IDs."""
    print("Fetching ATLAS protein list...", flush=True)
    r = requests.get(f"{BASE_URL}/parsable", timeout=60)
    r.raise_for_status()

    protein_ids: list[str] = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            # Files named like 2023_03_09_ATLAS_pdb.txt contain chain IDs
            if "ATLAS" in name and name.endswith("_pdb.txt"):
                with zf.open(name) as f:
                    for line in f.read().decode().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            protein_ids.append(line)

    print(f"  Found {len(protein_ids)} proteins in ATLAS.", flush=True)
    return sorted(set(protein_ids))


def already_downloaded(protein_id: str) -> bool:
    """Return True if this protein has already been extracted."""
    marker = OUT_DIR / protein_id / ".complete"
    return marker.exists()


def download_one(protein_id: str, aria2c_bin: str = "aria2c") -> bool:
    """
    Download and extract one protein's trajectory archive.
    Returns True on success.
    """
    dest_dir = OUT_DIR / protein_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / f"{protein_id}.zip"
    url = f"{BASE_URL}/ATLAS/protein/{protein_id}"

    # Download with aria2c: resumable, 4 parallel connections per server
    cmd = [
        aria2c_bin,
        "--continue=true",
        "--max-connection-per-server=4",
        "--split=4",
        "--max-tries=5",
        "--retry-wait=3",
        f"--dir={dest_dir}",
        f"--out={protein_id}.zip",
        url,
    ]

    print(f"  [{protein_id}] downloading...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [{protein_id}] aria2c failed:\n{result.stderr}", flush=True)
        return False

    # Extract
    print(f"  [{protein_id}] extracting...", flush=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()  # remove ZIP after successful extraction
        (dest_dir / ".complete").touch()
        print(f"  [{protein_id}] done.", flush=True)
        return True
    except zipfile.BadZipFile as e:
        print(f"  [{protein_id}] bad ZIP: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Download ATLAS MD trajectories")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",  action="store_true", help="Print all available protein IDs and exit")
    group.add_argument("--pilot", action="store_true", help="Download the curated 20-protein pilot set")
    group.add_argument("--ids",   nargs="+", metavar="ID", help="Download specific protein chain IDs")
    group.add_argument("--all",   action="store_true", help="Download all of ATLAS (check space first)")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory (default: %(default)s)")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)

    # --list
    if args.list:
        ids = fetch_protein_list()
        for pid in ids:
            print(pid)
        return

    # Resolve which proteins to download
    if args.pilot:
        targets = PILOT_SET
        print(f"Pilot set: {len(targets)} proteins", flush=True)
    elif args.ids:
        targets = args.ids
    else:  # --all
        targets = fetch_protein_list()
        print(f"Downloading all {len(targets)} proteins. This will take a while.", flush=True)

    # Download
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, skipped, failed = 0, 0, []

    for i, pid in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {pid}", flush=True)
        if already_downloaded(pid):
            print(f"  [{pid}] already complete, skipping.", flush=True)
            skipped += 1
            continue
        if download_one(pid):
            ok += 1
        else:
            failed.append(pid)

    # Summary
    print(f"\n{'='*50}")
    print(f"Done.  Success: {ok}  Skipped: {skipped}  Failed: {len(failed)}")
    if failed:
        print(f"Failed proteins: {failed}")
        print("Re-run the same command to retry failed downloads (aria2c resumes automatically).")
    else:
        print(f"All trajectories in: {OUT_DIR}")


if __name__ == "__main__":
    main()
