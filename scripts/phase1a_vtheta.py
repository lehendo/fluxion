#!/usr/bin/env python3
"""Velocity output of the original run's checkpoint on its training latents (paper Section 7.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fluxion.analysis.mask import n_residues, real_mask, verify_mask
from fluxion.data_pipeline.dataset import LatentDataset

T_GRID = [0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]     # 21 points
N_TOK_SUBSET = 4096


def collate(batch):
    return torch.stack([torch.stack(list(item)) for item, _p, _f in batch])   # [B, 2, L, d]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--cache", type=Path, default=DATA_ROOT / "phase1_cache")
    ap.add_argument("--checkpoint", type=Path, default=DATA_ROOT / "checkpoints/checkpoint.pt")
    ap.add_argument("--protein", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from fluxion.model.dit import FluxionDiT
    dev = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    models = {}
    for name, key in (("raw", "model"), ("ema", "ema")):
        m = FluxionDiT(); m.load_state_dict(state[key]); models[name] = m.to(dev).eval()

    ns = torch.load(args.latents_dir / "norm_stats.pt", weights_only=True)
    ds = LatentDataset(args.latents_dir, args.protein, load_pair=True, norm_stats=ns, lag=1)
    pair = ds.static_pair.to(dev)
    L = ds._load_frame(0).shape[0]
    n_res = n_residues(args.metadata_json, args.protein)
    mask = real_mask(L, n_res)
    g = torch.Generator().manual_seed(args.seed + zlib.crc32(args.protein.encode()) % 100000)
    verify_mask(mask, torch.stack([ds._load_frame(i) for i in range(24)]))   # one shard: padding is run-dependent
    real_idx = mask.nonzero().squeeze(1)
    mean_pos = torch.load(args.cache / f"{args.protein}__meta.pt", weights_only=False)["mean_pos"].float()
    d = mean_pos.shape[1]

    valid = [i for i in range(len(ds)) if ds._next_idx[i] != i]
    flat = sorted(valid[i] for i in torch.randperm(len(valid), generator=g)[: args.frames].tolist())
    loader = DataLoader(Subset(ds, flat), batch_size=8, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    # fixed token subset: (frame-in-run index, real-position index)
    n_frames = len(flat)
    tok_f = torch.randint(0, n_frames, (N_TOK_SUBSET,), generator=g)
    tok_p = real_idx[torch.randint(0, n_res, (N_TOK_SUBSET,), generator=g)]
    nT = len(T_GRID)
    acc = {m: {k: torch.zeros(nT, dtype=torch.float64) for k in ("v2", "u2", "vu", "e2")} for m in models}
    tok_v = {m: torch.zeros(nT, N_TOK_SUBSET) for m in models}
    tok_u = torch.zeros(N_TOK_SUBSET)
    cs = {k: torch.zeros(nT, d, dtype=torch.float64) for k in ("sx", "sy", "sxx", "sxy", "syy")}
    n_tokens = 0
    mp = mean_pos.to(dev)
    ridx = real_idx.to(dev)
    base = 0
    with torch.no_grad():
        for zs in loader:
            B = zs.shape[0]
            z0, z1 = zs[:, 0].to(dev), zs[:, 1].to(dev)
            u = z1 - z0
            n_tokens += B * n_res
            sel = (tok_f >= base) & (tok_f < base + B)
            if sel.any():
                bi = tok_f[sel] - base
                tok_u[sel] = u[bi.to(dev), tok_p[sel].to(dev)].pow(2).mean(-1).sqrt().cpu()
            for ti, t in enumerate(T_GRID):
                zt = (1.0 - t) * z0 + t * z1
                tv = torch.full((B,), t, device=dev)
                pair_b = pair.unsqueeze(0).expand(B, -1, -1, -1)
                ur = u[:, ridx]
                x = (zt - mp)[:, ridx].double()                  # centered z_t, real tokens
                y = ur.double()
                cs["sx"][ti] += x.sum((0, 1)).cpu(); cs["sy"][ti] += y.sum((0, 1)).cpu()
                cs["sxx"][ti] += (x * x).sum((0, 1)).cpu(); cs["sxy"][ti] += (x * y).sum((0, 1)).cpu()
                cs["syy"][ti] += (y * y).sum((0, 1)).cpu()
                for name, model in models.items():
                    v = model(zt, pair_b, tv, None)
                    vr = v[:, ridx]
                    acc[name]["v2"][ti] += vr.double().pow(2).sum().item()
                    acc[name]["u2"][ti] += ur.double().pow(2).sum().item()
                    acc[name]["vu"][ti] += (vr * ur).double().sum().item()
                    acc[name]["e2"][ti] += (vr - ur).double().pow(2).sum().item()
                    if sel.any():
                        tok_v[name][ti, sel] = v[bi.to(dev), tok_p[sel].to(dev)].pow(2).mean(-1).sqrt().cpu()
            base += B
    torch.save({"protein": args.protein, "t_grid": T_GRID, "n_tokens": n_tokens, "d": d, "n_res": n_res,
                "L_bucket": L, "n_frames": n_frames, "ckpt_step": int(state["step"]),
                "acc": {m: {k: v.clone() for k, v in a.items()} for m, a in acc.items()},
                "tok_rms_v": tok_v, "tok_rms_u": tok_u, "condmean_stats": cs},
               args.out_dir / f"{args.protein}__vtheta.pt")
    rms = lambda m, ti: (acc[m]["v2"][ti] / (n_tokens * d)).sqrt().item()
    u_rms = (acc["ema"]["u2"][0] / (n_tokens * d)).sqrt().item()
    print(f"[vtheta] {args.protein}: RMS|u|={u_rms:.5f}  EMA RMS|v| at t=0.01/0.5/0.99 = "
          f"{rms('ema', 0):.6f}/{rms('ema', T_GRID.index(0.5)):.6f}/{rms('ema', nT - 1):.6f}", flush=True)


if __name__ == "__main__":
    main()
