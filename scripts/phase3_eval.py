#!/usr/bin/env python3
"""
scripts/phase3_eval.py   (TMLR Phase 3a evaluation; protocol pre-registered in experiments/EXPERIMENT_LOG.md)

Lag-1 (10 ps) pairs in centered coordinates, REAL tokens only, N sampled valid start frames per protein (fixed seed, identical for every
method and both arms). Methods, all scored on the same frames:
  identity            z1_hat = z0
  shrink              z1_hat = alpha * z0, alpha = sum(z0 z1)/sum(z0 z0) fitted on the TRAINING entries only
  linear_diag         per-channel OLS fitted on the TRAINING entries only
  dit_<arm>_<ema|raw>_n<1|8>   z1_hat = Euler integration of the trained velocity field from t=0 with 1 or 8 steps
Reported as % improvement over identity: 100 * (1 - MSE_method / MSE_identity), per protein, then mean over the held-out entries
(proteins of one entry averaged first) with a percentile bootstrap CI over entries (4 held-out entries: coarse, stated as such).
The same is reported on the TRAINING proteins as an in-sample column (labelled in-sample).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.dit_model import FluxionDiT
from training.phase3_data import PAD_TO, build_datasets, collate, entry_of, load_pair_store

N_FRAMES = 400


def load_models(ckpt_path: Path, dev):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {}
    for name, key in (("ema", "ema"), ("raw", "model")):
        m = FluxionDiT(**st["arch"]); m.load_state_dict(st[key]); out[name] = m.to(dev).eval()
    return out, st["step"]


@torch.no_grad()
def integrate(model, z0, pairs_b, mask, steps: int):
    z = z0
    for k in range(steps):
        t = torch.full((z.shape[0],), k / steps, device=z.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            z = z + model(z, pairs_b, t, mask).float() / steps
    return z


def sample_batches(ds, n: int, seed: int, bs: int):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return DataLoader([ds[i] for i in idx], batch_size=bs, shuffle=False, collate_fn=collate)


def fit_baselines(train_ds: dict, dev, seed: int):
    """alpha and per-channel (a_c, k_c) from streaming sums over sampled TRAINING frames (real tokens)."""
    S = None
    for p, ds in train_ds.items():
        for z0, z1, pi, mask in sample_batches(ds, N_FRAMES, seed, 16):
            v = (~mask).unsqueeze(-1).double()
            x, y = z0.double() * v, z1.double() * v
            n = v.sum()
            cur = dict(n=n, sx=x.sum((0, 1)), sy=y.sum((0, 1)), sxx=(x * x).sum((0, 1)), sxy=(x * y).sum((0, 1)))
            S = cur if S is None else {k: S[k] + cur[k] for k in S}
    alpha = float(S["sxy"].sum() / S["sxx"].sum())
    mx, my = S["sx"] / S["n"], S["sy"] / S["n"]
    k = (S["sxy"] / S["n"] - mx * my) / (S["sxx"] / S["n"] - mx * mx)
    a = my - k * mx
    return alpha, a.float().to(dev), k.float().to(dev)


@torch.no_grad()
def score_protein(ds, pairs, models, alpha, a, k, dev, seed: int, arm: str):
    sums = {}
    for z0, z1, pi, mask in sample_batches(ds, N_FRAMES, seed, 8):
        z0, z1, pi, mask = z0.to(dev), z1.to(dev), pi.to(dev), mask.to(dev)
        valid = (~mask).float().unsqueeze(-1)
        pb = pairs[pi]
        preds = {"identity": z0, "shrink": alpha * z0, "linear_diag": a + k * z0}
        for mname, m in models.items():
            for steps in (1, 8):
                preds[f"dit_{arm}_{mname}_n{steps}"] = integrate(m, z0, pb, mask, steps)
        for name, p in preds.items():
            sums[name] = sums.get(name, 0.0) + float((((p - z1) ** 2) * valid).sum())
        sums["_tokens"] = sums.get("_tokens", 0.0) + float(valid.sum() * z0.shape[-1])
    return {n: s / sums["_tokens"] for n, s in sums.items() if n != "_tokens"}


def boot(x, n=20000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/latents"))
    ap.add_argument("--meta-json", type=Path, default=Path("/scratch/arjunc4/fluxion_data/extraction_metadata.json"))
    ap.add_argument("--mean-dir", type=Path, default=Path("/scratch/arjunc4/fluxion_data/phase3/mean_pos"))
    ap.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "CKPT"), required=True, help="repeatable: --arm uniform path --arm beta path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = [json.loads((Path(c).parent / "config.json").read_text()) for _, c in a.arm]
    train_p, held_p = cfgs[0]["train"], cfgs[0]["held_out"]
    assert all(c["train"] == train_p and c["held_out"] == held_p for c in cfgs), "arms must share the split"
    all_p = train_p + held_p; index_of = {p: i for i, p in enumerate(all_p)}
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True)
    pairs = load_pair_store(a.latents_dir, all_p, dev)
    tr = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, train_p, index_of, ns)
    ho = build_datasets(a.latents_dir, a.meta_json, a.mean_dir, held_p, index_of, ns)
    alpha, ba, bk = fit_baselines(tr, dev, a.seed)
    print(f"[eval] alpha (training entries) = {alpha:.4f}", flush=True)
    arms = {}
    for name, ck in a.arm:
        arms[name] = load_models(Path(ck), dev); print(f"[eval] arm {name}: step {arms[name][1]}", flush=True)
    res = {"alpha": alpha, "arms_steps": {n: v[1] for n, v in arms.items()}, "per_protein": {}}
    for group, dss in (("held_out", ho), ("in_sample", tr)):
        for p, ds in dss.items():
            row = {}
            first = True
            for name, (models, _) in arms.items():
                r = score_protein(ds, pairs, models, alpha, ba, bk, dev, a.seed, name)
                if first:
                    row.update({k: v for k, v in r.items() if not k.startswith("dit_")}); first = False
                row.update({k: v for k, v in r.items() if k.startswith("dit_")})
            res["per_protein"][f"{group}:{p}"] = {"group": group, "mse": row, "pct_vs_identity": {k: 100 * (1 - v / row["identity"]) for k, v in row.items() if k != "identity"}}
            print(f"[eval] {group} {p}: " + "  ".join(f"{k}={100 * (1 - v / row['identity']):+.2f}%" for k, v in row.items() if k in ("shrink", "linear_diag") or k.endswith("_ema_n1")), flush=True)
    summary = {}
    for group in ("held_out", "in_sample"):
        ents = {}
        for key, v in res["per_protein"].items():
            if v["group"] == group:
                ents.setdefault(entry_of(key.split(":")[1]), []).append(v["pct_vs_identity"])
        methods = next(iter(ents.values()))[0].keys()
        summary[group] = {m: boot([np.mean([r[m] for r in rs]) for rs in ents.values()]) for m in methods}
        summary[group]["_n_entries"] = len(ents)
    res["summary"] = summary
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for group in summary:
        print(f"\n=== {group}: % improvement over identity, mean over {summary[group]['_n_entries']} entries [95% bootstrap CI over entries] ===")
        for m, v in summary[group].items():
            if m != "_n_entries":
                print(f"  {m:28s} {v[0]:+7.2f} [{v[1]:+7.2f}, {v[2]:+7.2f}]")


if __name__ == "__main__":
    main()
