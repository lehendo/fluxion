#!/usr/bin/env python3
"""Centered padded-token norms against real tokens (paper Section 3.1, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from fluxion.model.dit import FluxionDiT
from fluxion.training.phase3_data import build_datasets, collate, load_pair_store

P3 = DATA_ROOT / "phase3"


def token_stats(ds, n_frames, seed=0):
    g = torch.Generator().manual_seed(seed); idx = torch.randperm(len(ds), generator=g)[:n_frames].tolist()
    pad, real = [], []
    for i in idx:
        z0, z1, pi, n_mask = ds[i]; n_res = ds.n_res
        pad.append(z0[n_res:]); real.append(z0[:n_res])
    pad, real = torch.cat(pad), torch.cat(real)
    nz = lambda x: float((x.abs().sum(1) == 0).float().mean())
    return {"pad_n": len(pad), "pad_max_abs": float(pad.abs().max()) if len(pad) else 0.0, "pad_exact_zero_frac": nz(pad) if len(pad) else float("nan"), "pad_mean_norm": float(pad.norm(dim=1).mean()) if len(pad) else 0.0,
            "real_exact_zero_frac": nz(real), "real_mean_norm": float(real.norm(dim=1).mean())}


@torch.no_grad()
def model_out(ckpt, ds, pairs, n_frames, t_list=(0.1, 0.5), seed=0):
    st = torch.load(ckpt, map_location="cpu", weights_only=False); m = FluxionDiT(**st["arch"]); m.load_state_dict(st["ema"]); m.eval()
    g = torch.Generator().manual_seed(seed); idx = torch.randperm(len(ds), generator=g)[:n_frames].tolist()
    batch = [ds[i] for i in idx]; z0, z1, pi, mask = collate(batch); n_res = ds.n_res
    out = {}
    for t in t_list:
        zt = (1 - t) * z0 + t * z1; tt = torch.full((z0.shape[0],), t)
        v = m(zt, pairs[pi], tt, mask).float()
        out[f"t{t}"] = {"v_norm_pad": float(v[:, n_res:ds.L_bucket].norm(dim=-1).mean()) if ds.L_bucket > n_res else float("nan"), "v_norm_real": float(v[:, :n_res].norm(dim=-1).mean()),
                        "target_norm_real": float((z1 - z0)[:, :n_res].norm(dim=-1).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--proteins", nargs="+", default=["1bx7_A", "1ab1_A", "1ail_A", "1b2s_E", "1bgf_A", "1a62_A"]); ap.add_argument("--n-frames", type=int, default=64)
    a = ap.parse_args()
    lat, meta, mean = DATA_ROOT / "latents", DATA_ROOT / "extraction_metadata.json", P3 / "mean_pos"
    cfg = json.loads((P3 / "t3a/config.json").read_text()); all_p = cfg["train"] + cfg["held_out"]; index_of = {p: i for i, p in enumerate(all_p)}
    ns = torch.load(lat / "norm_stats.pt", weights_only=True); pairs = load_pair_store(lat, all_p, torch.device("cpu"))
    res = {}
    for center in (True, False):
        ds = build_datasets(lat, meta, mean, a.proteins, index_of, ns, center=center, mask_real=False)
        for p, d in ds.items():
            d.L_bucket = int(torch.load(lat / "norm_stats.pt", weights_only=True) and d.ds._load_frame(0).shape[0])
            res.setdefault(p, {})["centered" if center else "uncentered"] = token_stats(d, a.n_frames)
            res[p]["n_res"], res[p]["L_bucket"] = d.n_res, d.L_bucket
        if center or True:
            for arm, ck in (("t3b", P3 / "t3b/milestone_100.pt"), ("t3b_s1", P3 / "t3b_s1/milestone_100.pt")) if center else (("t3a", P3 / "t3a/milestone_100.pt"),):
                for p, d in ds.items():
                    res[p].setdefault("model", {})[arm] = model_out(ck, d, pairs, 16)
    print(json.dumps(res, indent=1))
    print(f"\n{'protein':8s} {'n_res':>5s} {'bucket':>6s} | centered: pad max|v| / exact-zero frac / mean norm | uncentered: pad norm / real norm | real exact-zero frac")
    for p, r in res.items():
        c, u = r["centered"], r["uncentered"]
        print(f"{p:8s} {r['n_res']:5d} {r['L_bucket']:6d} | {c['pad_max_abs']:.2e} / {c['pad_exact_zero_frac']:.3f} / {c['pad_mean_norm']:.2e} | {u['pad_mean_norm']:.2f} / {u['real_mean_norm']:.2f} | {c['real_exact_zero_frac']:.3f}")


if __name__ == "__main__":
    main()
