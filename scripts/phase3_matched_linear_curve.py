#!/usr/bin/env python3
"""Matched linear reference curve fit on the training entries (paper Section 6.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import phase3_eval as E
from fluxion.training.phase3_data import build_datasets, n_real_table

R = Path("paper_results/phase3")


def entry_mean(curves_by_protein):
    ents = {}
    for p, c in curves_by_protein.items():
        ents.setdefault(p.split("_")[0], []).append(np.array(c))
    return np.mean([np.mean(v, 0) for v in ents.values()], 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents"); ap.add_argument("--meta-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--mean-dir", type=Path, default=DATA_ROOT / "phase3/mean_pos"); ap.add_argument("--config", type=Path, default=DATA_ROOT / "phase3/uniform/config.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase3/matched_linear_curve.json"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    cfg = json.loads(a.config.read_text()); train_p, held_p = cfg["train"], cfg["held_out"]; all_p = train_p + held_p; index_of = {p: i for i, p in enumerate(all_p)}
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    tr = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, train_p, index_of, ns); ho = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, held_p, index_of, ns)
    E.REAL_N = n_real_table(tr, ho)
    T = E.T_GRID; nT = len(T); d = 384
    S = {k: torch.zeros(nT, d, dtype=torch.float64, device=dev) for k in ("sx", "sy", "sxx", "sxy")}; n = torch.zeros(nT, dtype=torch.float64, device=dev)
    for p, ds in tr.items():
        for z0, z1, pi, mask in E.sample_batches(ds, E.N_FRAMES, a.seed, 16):
            z0, z1, pi, mask = z0.to(dev), z1.to(dev), pi.to(dev), mask.to(dev); valid = E.score_valid(pi, mask)
            u = (z1 - z0).double(); vv = valid.double()
            for ti, t in enumerate(T):
                x = ((1 - float(t)) * z0 + float(t) * z1).double() * vv; y = u * vv
                S["sx"][ti] += x.sum((0, 1)); S["sy"][ti] += y.sum((0, 1)); S["sxx"][ti] += (x * x).sum((0, 1)); S["sxy"][ti] += (x * y).sum((0, 1)); n[ti] += vv.sum()
        E.close_handles(ds); print(f"[matched] fit stats from {p}", flush=True)
    mx, my = S["sx"] / n[:, None], S["sy"] / n[:, None]
    k = (S["sxy"] / n[:, None] - mx * my) / (S["sxx"] / n[:, None] - mx * mx); b = my - k * mx
    curves = {}
    for p, ds in ho.items():
        num = np.zeros(nT); den = 0.0
        for z0, z1, pi, mask in E.sample_batches(ds, E.N_FRAMES, a.seed, 16):
            z0, z1, pi, mask = z0.to(dev), z1.to(dev), pi.to(dev), mask.to(dev); valid = E.score_valid(pi, mask).double(); u = (z1 - z0).double()
            den += float(((u ** 2) * valid).sum())
            for ti, t in enumerate(T):
                zt = ((1 - float(t)) * z0 + float(t) * z1).double()
                pred = b[ti][None, None, :] + k[ti][None, None, :] * zt
                num[ti] += float((((pred - u) ** 2) * valid).sum())
        curves[p] = (1 - num / den).tolist(); E.close_handles(ds); print(f"[matched] held-out {p}: f_matched(0.01) = {100 * curves[p][0]:.2f}%  f_matched(0.5) = {100 * curves[p][20]:.2f}%", flush=True)
    res = {"t_grid": T.tolist(), "per_protein": curves, "entry_mean": entry_mean(curves).tolist()}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res))
    print("entry-mean matched linear curve at t = 0.01, 0.23, 0.5, 0.79, 0.99:", [round(100 * res["entry_mean"][i], 2) for i in (0, 9, 20, 32, 40)])


if __name__ == "__main__":
    main()
