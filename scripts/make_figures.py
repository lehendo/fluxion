#!/usr/bin/env python3
"""Regenerates every paper figure and report from the saved results in paper_results/ (no GPU or cluster data needed)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGES = [
    ("Probe-study figures 1-4 (attenuation, lag sweep, shrinkage, Cartesian control)", "scripts/plot_phase1.py",
     ["paper_results/phase1/headline.json", "paper_results/phase1/sweep/across_proteins.json", "paper_results/phase1/b1/b1_report.json"]),
    ("Tables of Sections 3.3, 4.1, 7.1 and Appendices A.1, A.3, A.7", "scripts/make_tables.py",
     ["paper_results/phase1/noise_attribution_perprotein.json", "paper_results/phase3/eval_100.json", "paper_results/phase3/matched_linear_curve.json"]),
    ("Figure 5 (gradient SNR) and the E3 table", "scripts/e3_table.py", ["paper_results/phase1/gradsnr.json"]),
    ("Flow-model figures 6-10 (flow curves, trajectories, step sweep, diagnostic, per-entry)", "scripts/plot_phase3.py",
     ["paper_results/phase3/eval_100.json", "paper_results/phase3/predictions.json"]),
]


REPORTS = [   # (title, script, output file): text reports whose numbers the paper quotes; captured so that every quoted number is reachable from this driver
    ("B1 at 10 ps report", "scripts/phase1b_report_10ps.py", "paper_results/phase1/b1_10ps_report.txt", ["paper_results/phase1/b1_10ps"]),
    ("B1-10ps controls C1-C3", "scripts/phase1b_controls_report.py", "paper_results/phase1/b1_10ps_controls_report.txt", ["paper_results/phase1/b1_10ps_controls"]),
    ("B1-10ps top-m (post hoc)", "scripts/phase1b_controls_topm.py", "paper_results/phase1/b1_10ps_controls_topm.txt", ["paper_results/phase1/b1_10ps_controls"]),
    ("Stability diagnostics, registered statistic", "scripts/phase1b_stability_report.py", "paper_results/phase1/b1_10ps_stability_report.txt", ["paper_results/phase1/b1_10ps_stability"]),
    ("Stability diagnostics, chance-corrected", "scripts/phase1b_stability_null.py", "paper_results/phase1/b1_10ps_stability_null.txt", ["paper_results/phase1/b1_10ps_stability"]),
    ("Seed replication reading", "scripts/phase3_seed_report.py", "paper_results/phase3/seed_report.txt", ["paper_results/phase3/eval_s1_100.json", "paper_results/phase3/eval_s2_100.json"]),
]


def run_reports() -> int:
    import contextlib, io
    failed = 0
    for title, script, out, needs in REPORTS:
        if any(not (ROOT / n).exists() for n in needs):
            print(f"[skip] {title}: missing inputs"); continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                runpy.run_path(str(ROOT / script), run_name="__main__")
            (ROOT / out).write_text(buf.getvalue()); print(f"[run ] {title} -> {out}")
        except SystemExit as e:
            (ROOT / out).write_text(buf.getvalue()); print(f"[run ] {title} -> {out} (exit {e.code})")
        except Exception as e:
            failed += 1; print(f"[FAIL] {script}: {type(e).__name__}: {e}")
    return failed


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))          # sibling scripts import each other by module name
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
    failed += run_reports()
    return 1 if failed else 0


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    sys.exit(main())
