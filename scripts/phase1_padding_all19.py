#!/usr/bin/env python3
"""Padded-versus-masked MLP evaluation on all 19 proteins (paper Section 3.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from fluxion.analysis.stats import entry_bootstrap_mean, entry_groups, pct_improvement
from phase1a_ladder import PILOT, fit_predict, load, maybe_mask, sub

KEYS = ("identity", "mean", "linear", "mlp")


def run_arm(cells, groups, mask_pad, device, seeds):
    cs = {p: (sub(cells[p], ~cells[p]["pad"]) if mask_pad else cells[p]) for p in PILOT}
    per = {}
    for entry, members in groups.items():
        others = [e for e in groups if e != entry]
        stop_entries = [others[0], others[len(others) // 2]]
        tr = [p for e in others if e not in stop_entries for p in groups[e]]; st = [p for e in stop_entries for p in groups[e]]
        eo = fit_predict([cs[p] for p in tr + st], [cs[p] for p in tr], [cs[p] for p in st], [cs[p] for p in members], device, seeds)
        for p in members:
            per[p] = {k: float(eo[p][k].mean()) for k in KEYS}
        print(f"  [{'masked' if mask_pad else 'counted'}] fold {entry}: " + ", ".join(f"{p} mlp/mean={per[p]['mlp'] / per[p]['mean']:.3f} id/mean={per[p]['identity'] / per[p]['mean']:.3f}" for p in members), flush=True)
    return per


def summarize(per, pids):
    f = lambda a, b: {p: pct_improvement(per[p][a], per[p][b]) for p in pids}
    return {n: dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean(f(a, b), n_boot=10000))) for n, (a, b) in
            {"mlp_vs_mean": ("mlp", "mean"), "linear_vs_mean": ("linear", "mean"), "identity_vs_mean": ("identity", "mean"), "mlp_vs_identity": ("mlp", "identity")}.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DATA_ROOT / "phase1_cache_s0all")
    ap.add_argument("--acf", type=Path, default=DATA_ROOT / "phase1_results/acf.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/padding_all19.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    acf = json.loads(a.acf.read_text())
    cells = {p: load(a.cache, p, acf) for p in PILOT}
    pad_frac = {p: float(cells[p]["pad"].float().mean()) for p in PILOT}
    groups = entry_groups(PILOT)
    res = {"pad_frac": pad_frac, "n_rows": {p: int(len(cells[p]["z0"])) for p in PILOT}}
    for name, mask in (("masked", True), ("counted", False)):
        res[name] = {"per_protein": run_arm(cells, groups, mask, a.device, a.seeds)}
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    subsets = {"all19": PILOT, "without_1bxy_A": [p for p in PILOT if p != "1bxy_A"], "without_entry_1bxy": [p for p in PILOT if not p.startswith("1bxy")]}
    for arm in ("masked", "counted"):
        res[arm]["summary"] = {k: summarize(res[arm]["per_protein"], v) for k, v in subsets.items()}
    a.out.write_text(json.dumps(res, indent=1))
    print("\nimprovement over the mean baseline (%), mean over proteins [95% CI over entries]:")
    for k in subsets:
        for arm in ("counted", "masked"):
            s = res[arm]["summary"][k]
            print(f"  {k:20s} {arm:8s} MLP/mean {s['mlp_vs_mean']['mean']:6.2f} [{s['mlp_vs_mean']['lo']:6.2f},{s['mlp_vs_mean']['hi']:6.2f}]   identity/mean {s['identity_vs_mean']['mean']:6.2f} [{s['identity_vs_mean']['lo']:6.2f},{s['identity_vs_mean']['hi']:6.2f}]")


if __name__ == "__main__":
    main()
