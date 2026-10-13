#!/usr/bin/env python3
"""
scripts/make_figures.py : regenerate every figure from the saved results (no cluster access needed).

Run from the repository root:  python scripts/make_figures.py

Inputs (produced by the analysis and evaluation jobs and copied from the cluster) are read from results/phase1/ and results/phase3/;
figures are written next to them (results/phase1/figs, results/phase3/figs). A stage whose inputs are missing is skipped with a message.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGES = [
    ("Phase 1 figures 1-4 (attenuation, lag sweep, shrinkage, Cartesian control)", "scripts/plot_phase1.py",
     ["results/phase1/headline.json", "results/phase1/sweep/across_proteins.json", "results/phase1/b1/b1_report.json"]),
    ("Figure 5 (gradient SNR) and the E3 table", "scripts/e3_table.py", ["results/phase1/gradsnr.json"]),
    ("Phase 3 figures 6-10 (flow curves, trajectories, step sweep, diagnostic, per-entry)", "scripts/plot_phase3.py",
     ["results/phase3/eval_100.json", "results/phase3/predictions.json"]),
]


def main() -> int:
    failed = 0
    for title, script, needs in STAGES:
        missing = [n for n in needs if not (ROOT / n).exists()]
        if missing:
            print(f"[skip] {title}: missing {missing}")
            continue
        print(f"[run ] {title}")
        try:
            runpy.run_path(str(ROOT / script), run_name="__main__")
        except Exception as e:                                           # keep going so one broken stage does not hide the others
            failed += 1
            print(f"[FAIL] {script}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    sys.exit(main())
