#!/usr/bin/env python3
"""Writes the 200-frame trajectory subset used for the per-protein extraction-noise measurement (paper Section 3.2)."""
from __future__ import annotations
import argparse
from pathlib import Path
import MDAnalysis as mda
import numpy as np

FRAMES = list(range(0, 10000, 50))   # 200 frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True); ap.add_argument("--xtc", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    u = mda.Universe(a.pdb, a.xtc)
    assert len(u.trajectory) >= FRAMES[-1] + 1, f"trajectory has {len(u.trajectory)} frames"
    ref = []
    with mda.Writer(a.out, n_atoms=u.atoms.n_atoms) as w:
        for f in FRAMES:
            u.trajectory[f]; w.write(u.atoms); ref.append(u.atoms.positions.copy())
    v = mda.Universe(a.pdb, a.out)
    assert len(v.trajectory) == len(FRAMES), (len(v.trajectory), len(FRAMES))
    worst = 0.0
    for k, ts in enumerate(v.trajectory):
        worst = max(worst, float(np.abs(v.atoms.positions - ref[k]).max()))
    print(f"{Path(a.out).name}: {len(FRAMES)} frames, max |coordinate difference| after the DCD round trip = {worst:.2e} Angstrom")
    assert worst < 1e-4, "DCD round trip changed coordinates"


if __name__ == "__main__":
    main()
