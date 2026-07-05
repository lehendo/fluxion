#!/usr/bin/env python3
"""
scripts/summarize_lag_sweep.py

Parses the output logs from scripts/sweep_lag_baseline.sh (one file per
SLURM array task, one lag value each) into a single table sorted by lag, so
the crossover point — where real conformational-dynamics signal starts to
exceed thermal noise — is visible at a glance instead of scattered across
8 separate log files.

Usage:
    python scripts/summarize_lag_sweep.py --logs-dir logs --job-id 123456
    # or point it at an explicit glob:
    python scripts/summarize_lag_sweep.py --glob "logs/lagsweep_123456_*.out"
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

_LAG_RE       = re.compile(r"\[posmean\] lag=(\d+) frames")
_ZERO_MSE_RE  = re.compile(r"\[summary\] Aggregate zero-baseline MSE:\s+([\d.eE+-]+)")
_MEAN_MSE_RE  = re.compile(r"\[summary\] Aggregate position-mean baseline MSE:\s+([\d.eE+-]+)")
_REDUCTION_RE = re.compile(r"\[summary\] Reduction from static position mean:\s+([\d.eE+-]+)%")


def parse_sweep_output(text: str) -> dict[str, float] | None:
    """
    Extracts lag (frames) and the three aggregate summary numbers from one
    sweep-array-task's stdout. Returns None if the file doesn't contain a
    complete summary block (e.g. the job was killed before finishing, or
    isn't a lag-sweep output at all).
    """
    lag_m       = _LAG_RE.search(text)
    zero_m      = _ZERO_MSE_RE.search(text)
    mean_m      = _MEAN_MSE_RE.search(text)
    reduction_m = _REDUCTION_RE.search(text)

    if not (lag_m and zero_m and mean_m and reduction_m):
        return None

    return {
        "lag_frames":         int(lag_m.group(1)),
        "zero_baseline_mse":  float(zero_m.group(1)),
        "position_mean_mse":  float(mean_m.group(1)),
        "reduction_pct":      float(reduction_m.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--job-id", type=str, default=None,
                         help="SLURM array job id — globs logs/lagsweep_<job-id>_*.out")
    parser.add_argument("--glob", type=str, default=None,
                         help="Explicit glob pattern, overrides --logs-dir/--job-id")
    parser.add_argument("--ps-per-frame", type=float, default=10.0,
                         help="Native ATLAS save interval in ps (default 10, per project data)")
    args = parser.parse_args()

    if args.glob:
        pattern = args.glob
    elif args.job_id:
        pattern = str(args.logs_dir / f"lagsweep_{args.job_id}_*.out")
    else:
        pattern = str(args.logs_dir / "lagsweep_*_*.out")

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    results: list[dict[str, float]] = []
    incomplete: list[str] = []
    for path in paths:
        text = Path(path).read_text()
        parsed = parse_sweep_output(text)
        if parsed is None:
            incomplete.append(path)
        else:
            results.append(parsed)

    if incomplete:
        print(f"[sweep-summary] {len(incomplete)} file(s) had no complete summary "
              f"(job may still be running or failed): {incomplete}", flush=True)

    if not results:
        raise SystemExit("No complete sweep results found across matched files.")

    results.sort(key=lambda r: r["lag_frames"])

    header = f"{'Lag (frames)':>14} {'Lag (time)':>12} {'ZeroMSE':>12} {'PosMeanMSE':>12} {'%Reduction':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        lag_ps = r["lag_frames"] * args.ps_per_frame
        time_str = f"{lag_ps/1000:.1f}ns" if lag_ps >= 1000 else f"{lag_ps:.0f}ps"
        print(f"{r['lag_frames']:>14} {time_str:>12} {r['zero_baseline_mse']:>12.6f} "
              f"{r['position_mean_mse']:>12.6f} {r['reduction_pct']:>11.2f}%")

    crossover = next((r for r in results if r["reduction_pct"] > 1.0), None)
    print()
    if crossover:
        lag_ps = crossover["lag_frames"] * args.ps_per_frame
        print(f"[sweep-summary] First lag with >1% explainable static structure: "
              f"{crossover['lag_frames']} frames ({lag_ps:.0f}ps) — "
              f"{crossover['reduction_pct']:.2f}% reduction.")
    else:
        print("[sweep-summary] No tested lag exceeded 1% explainable static structure — "
              "consider testing longer lags, or this may indicate the signal requires "
              "genuine temporal/velocity conditioning rather than just a longer static lag.")


if __name__ == "__main__":
    main()
