#!/usr/bin/env python3
"""
scripts/phase1a_acf.py

Measures, PER PROTEIN, the autocorrelation of the latent trajectory and turns it into the
block length for the time-block bootstrap and an effective sample size per lag.

Method: structure function S(tau) = E||z(t+tau) - z(t)||^2 over real tokens, from
  * a contiguous 256-frame window at the start of each replica (tau = 1..128 frames), and
  * a stride-10 pass over each full replica (tau = 10 .. 5000 frames),
averaged across replicas. With pooled variance V (over all strided frames), the
autocorrelation is rho(tau) = 1 - S(tau) / (2V). tau_int = integral of rho (truncated where
rho < 0.05). Block length = lag + 2*tau_int frames; n_eff = valid start span / block length.

Outputs: <out>/acf.json and a printed table of nominal pairs vs. n_eff per lag.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from analysis.mask import n_residues, real_mask, verify_mask
from analysis.stats import (integrated_autocorr_time, block_len, n_eff_blocks,
                            n_independent_windows, lag_frames_to_label)
from data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,"
         "1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
SHORT_TAUS = [1, 2, 4, 8, 16, 32, 64, 128]
STRIDE = 10
LONG_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]       # x STRIDE -> 10..5000 frames
LAGS = [1, 10, 100, 1000, 5000]


def replica_slices(ds: LatentDataset) -> dict[int, tuple[int, int]]:
    out: dict[int, list[int]] = {}
    for i, (path, _) in enumerate(ds.index_map):
        m = _REPLICA_RE.search(path.name)
        code = int(re.sub(r"\D", "", m.group(1))) if m else 0
        out.setdefault(code, [i, i + 1])[1] = i + 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def structure(x: torch.Tensor, steps: list[int]) -> dict[int, float]:
    return {k: ((x[k:] - x[:-k]) ** 2).mean().item() for k in steps if k < x.shape[0]}


def measure(protein: str, latents_dir: Path, ns, seq_json: Path, seed: int, P: int = 64) -> dict:
    ds = LatentDataset(latents_dir, protein, load_pair=False, norm_stats=ns, lag=1)
    reps = replica_slices(ds)
    L = ds._load_frame(0).shape[0]
    mask = real_mask(L, n_residues(seq_json, protein))
    g = torch.Generator().manual_seed(seed)
    real_idx = mask.nonzero().squeeze(1)
    pos = real_idx[torch.randperm(len(real_idx), generator=g)[:P]]

    def frames(idxs):
        return torch.stack([ds._load_frame(i)[pos] for i in idxs])      # [T, P, d]

    verify_mask(mask, torch.stack([ds._load_frame(i) for i in range(24)]))   # one shard: padding is run-dependent

    S_short, S_long, strided = [], [], []
    for r, (a, b) in sorted(reps.items()):
        S_short.append(structure(frames(range(a, min(a + 256, b))), SHORT_TAUS))
        x = frames(range(a, b, STRIDE))
        strided.append(x)
        S_long.append(structure(x, LONG_STEPS))
    allx = torch.cat(strided, 0)
    V = allx.var(dim=0).mean().item()                                   # pooled variance per coordinate
    S = {}
    for tau in SHORT_TAUS:
        S[tau] = float(np.mean([s[tau] for s in S_short if tau in s]))
    for k in LONG_STEPS:
        vals = [s[k] for s in S_long if k in s]
        if vals:
            S[k * STRIDE] = float(np.mean(vals))
    taus = sorted(S)
    rho = [1.0 - S[t] / (2.0 * V) for t in taus]
    tau_int = integrated_autocorr_time(taus, rho)
    frames_per_rep = [b - a for (a, b) in reps.values()]
    return {"protein": protein, "L_bucket": L, "n_res": int(mask.sum()), "var": V,
            "taus": taus, "S": [S[t] for t in taus], "rho": rho, "tau_int_frames": tau_int,
            "frames_per_replica": frames_per_rep,
            "per_lag": {str(l): {"block_len": block_len(l, tau_int),
                                 "n_eff_blocks": n_eff_blocks(frames_per_rep, l, tau_int),
                                 "n_independent_windows": n_independent_windows(frames_per_rep, l),
                                 "nominal_valid_pairs": int(sum(max(n - l, 0) for n in frames_per_rep))}
                        for l in LAGS}}


def print_table(res: dict) -> None:
    print("\n=== effective sample size per lag (n_eff = valid start span / (lag + 2 tau_int)) ===")
    print(f"{'lag':>7s} {'nominal pairs':>14s} {'indep. windows':>15s} {'n_eff total':>12s} "
          f"{'n_eff/protein median [min,max]':>34s}")
    for l in LAGS:
        pl = [r["per_lag"][str(l)] for r in res.values()]
        ne = np.array([x["n_eff_blocks"] for x in pl])
        print(f"{lag_frames_to_label(l):>7s} {sum(x['nominal_valid_pairs'] for x in pl):14d} "
              f"{sum(x['n_independent_windows'] for x in pl):15d} {ne.sum():12.1f} "
              f"{np.median(ne):10.1f} [{ne.min():5.1f},{ne.max():6.1f}]")
    ti = np.array([r["tau_int_frames"] for r in res.values()])
    print(f"\ntau_int (frames; x10 ps): median {np.median(ti):.0f}, min {ti.min():.0f}, max {ti.max():.0f}")
    for r in res.values():
        print(f"  {r['protein']:8s} tau_int={r['tau_int_frames']:8.1f} frames "
              f"({r['tau_int_frames'] * 0.01:7.2f} ns)  rho(10ns)={r['rho'][r['taus'].index(1000)] if 1000 in r['taus'] else float('nan'):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--sequences-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--proteins", nargs="+", default=PILOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    if not args.summarize_only:
        ns = torch.load(args.latents_dir / "norm_stats.pt", weights_only=True)
        for p in args.proteins:
            if p in res:
                continue
            res[p] = measure(p, args.latents_dir, ns, args.sequences_json, args.seed)
            print(f"[acf] {p}: tau_int = {res[p]['tau_int_frames']:.1f} frames", flush=True)
            args.out.write_text(json.dumps(res, indent=1))
    print_table(res)


if __name__ == "__main__":
    main()
