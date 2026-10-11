#!/usr/bin/env python3
"""
scripts/phase3_eval.py   (TMLR Phase 3a evaluation; protocol pre-registered in experiments/EXPERIMENT_LOG.md)

Lag-1 (10 ps) pairs in centered coordinates, REAL tokens only, N sampled valid start frames per protein (fixed seed, identical for every
method and both arms). Methods, all scored on the same frames:
  identity            z1_hat = z0
  shrink              z1_hat = alpha * z0, alpha = sum(z0 z1)/sum(z0 z0) fitted on the TRAINING entries only
  linear_diag         per-channel OLS fitted on the TRAINING entries only
  dit_<arm>_<ema|raw>_n<1|8>   z1_hat = Euler integration of the trained velocity field from t=0 with 1 or 8 steps
PRIMARY (pre-registered) TEST, flow-loss space: on a fixed grid of 41 t values in [0.01, 0.99] the flow-loss reduction of each model,
R(t) = 1 - L_model(t) / L(v = 0), pooled over the sampled held-out frames of a protein; each arm's ATTAINED reduction is the integral of R against that arm's own
t-sampling distribution (Beta draws clamped to [0.01, 0.99], point masses included); the measured treatment/control ratio is compared with the closed-form
prediction in results/phase3/predictions.json (scripts/phase3_predict.py). DOWNSTREAM readout: Reported as % improvement over identity: 100 * (1 - MSE_method / MSE_identity), per protein, then mean over the held-out entries
(proteins of one entry averaged first) with a percentile bootstrap CI over entries (4 held-out entries: coarse, stated as such).
The same is reported on the TRAINING proteins as an in-sample column (labelled in-sample).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from training.dit_model import FluxionDiT
from training.phase3_data import PAD_TO, build_datasets, collate, entry_of, load_pair_store

N_FRAMES = 400
N_FRAMES_IN_SAMPLE = 100
T_LO, T_HI = 0.01, 0.99
T_GRID = np.linspace(T_LO, T_HI, 41)


def arm_masses(t_cfg: dict, grid: np.ndarray = T_GRID) -> np.ndarray:
    """Probability mass of the CLAMPED sampling distribution assigned to each grid point (cell between midpoints; the clamp point masses go to the end cells)."""
    lo, hi = t_cfg["clip"]
    d = stats.uniform() if t_cfg["dist"] == "uniform" else stats.beta(t_cfg["beta_a"], t_cfg["beta_b"])
    edges = np.concatenate([[lo], (grid[1:] + grid[:-1]) / 2, [hi]])
    m = np.diff(d.cdf(edges))
    m[0] += d.cdf(lo); m[-1] += 1 - d.cdf(hi)
    return m / m.sum()


def load_models(ckpt_path: Path, dev):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {}
    for name, key in (("ema", "ema"), ("raw", "model")):
        m = FluxionDiT(**st["arch"]); m.load_state_dict(st[key]); out[name] = m.to(dev).eval()
    return out, st["step"], st.get("t_cfg")


@torch.no_grad()
def integrate(model, z0, pairs_b, mask, steps: int):
    z = z0
    for k in range(steps):
        t = torch.full((z.shape[0],), max(k / steps, T_LO), device=z.device)         # never below the smallest t the model was trained at
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            z = z + model(z, pairs_b, t, mask).float() / steps
    return z


def close_handles(ds) -> None:
    """Release the HDF5 file handles a CenteredLagPairs dataset keeps open (63 shards/protein would exhaust the descriptor limit over 18 proteins)."""
    for f in ds.ds.file_handles.values():
        try:
            f.close()
        except Exception:
            pass
    ds.ds.file_handles.clear()


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
        close_handles(ds)
    alpha = float(S["sxy"].sum() / S["sxx"].sum())
    mx, my = S["sx"] / S["n"], S["sy"] / S["n"]
    k = (S["sxy"] / S["n"] - mx * my) / (S["sxx"] / S["n"] - mx * mx)
    a = my - k * mx
    return alpha, a.float().to(dev), k.float().to(dev)


@torch.no_grad()
def flow_loss_curves(ds, pairs, models, dev, seed: int, n_frames: int):
    """Per model: sum of squared error at each grid t, and the v=0 sum, over real tokens. Returns {model: R(t) array}, using ONE fixed set of frames and fixed noise-free t grid."""
    num = {m: np.zeros(len(T_GRID)) for m in models}; den = 0.0
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_frames].tolist()
    for batch in DataLoader([ds[i] for i in idx], batch_size=16, shuffle=False, collate_fn=collate):
        z0, z1, pi, mask = (x.to(dev) for x in batch)
        valid = (~mask).float().unsqueeze(-1); pb = pairs[pi]; u = z1 - z0
        den += float(((u ** 2) * valid).sum())
        for ti, t in enumerate(T_GRID):
            tt = torch.full((z0.shape[0],), float(t), device=dev)
            zt = (1 - float(t)) * z0 + float(t) * z1
            for mname, m in models.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    v = m(zt, pb, tt, mask).float()
                num[mname][ti] += float((((v - u) ** 2) * valid).sum())
    return {m: (1 - n / den).tolist() for m, n in num.items()}


@torch.no_grad()
def score_protein(ds, pairs, models, alpha, a, k, dev, seed: int, arm: str):
    sums = {}
    for z0, z1, pi, mask in sample_batches(ds, N_FRAMES, seed, 16):
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
        arms[name] = load_models(Path(ck), dev); print(f"[eval] arm {name}: step {arms[name][1]} t_cfg {arms[name][2]}", flush=True)
    res = {"alpha": alpha, "arms_steps": {n: v[1] for n, v in arms.items()}, "t_cfg": {n: v[2] for n, v in arms.items()}, "t_grid": T_GRID.tolist(), "per_protein": {}, "flow": {}}
    for group, dss in (("held_out", ho), ("in_sample", tr)):
        for p, ds in dss.items():
            row = {}
            first = True
            for name, (models, _, _) in arms.items():
                r = score_protein(ds, pairs, models, alpha, ba, bk, dev, a.seed, name)
                fl = flow_loss_curves(ds, pairs, models, dev, a.seed, N_FRAMES if group == "held_out" else N_FRAMES_IN_SAMPLE)
                for mname, curve in fl.items():
                    res["flow"].setdefault(f"{group}:{p}", {})[f"{name}_{mname}"] = curve
                if first:
                    row.update({k: v for k, v in r.items() if not k.startswith("dit_")}); first = False
                row.update({k: v for k, v in r.items() if k.startswith("dit_")})
            close_handles(ds)
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
    # ---- primary test: attained flow-loss reduction per arm and the treatment/control ratio
    flow_summary = {}
    for group in ("held_out", "in_sample"):
        ents = {}
        for key in res["flow"]:
            if key.split(":")[0] == group:
                ents.setdefault(entry_of(key.split(":")[1]), []).append(res["flow"][key])
        fs = {}
        for name, (_, _, tcfg) in arms.items():
            if tcfg is None:
                continue
            w = arm_masses(tcfg)
            for mname in ("ema", "raw"):
                per_entry = [float(np.mean([w @ np.array(r[f"{name}_{mname}"]) for r in rs])) for rs in ents.values()]
                fs[f"{name}_{mname}"] = {"attained_reduction": boot(per_entry), "per_entry": per_entry}
        for mname in ("ema", "raw"):
            for other in [n for n in arms if n != "uniform"]:
                if "uniform_" + mname in fs and f"{other}_" + mname in fs:
                    tu, tb = np.array(fs["uniform_" + mname]["per_entry"]), np.array(fs[f"{other}_" + mname]["per_entry"])
                    rng = np.random.default_rng(0); ix = rng.integers(0, len(tu), (20000, len(tu)))
                    rat = tb[ix].mean(1) / np.maximum(tu[ix].mean(1), 1e-12)
                    fs[f"ratio_{other}_over_uniform_" + mname] = {"point": float(tb.mean() / tu.mean()), "ci95": [float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))]}
        flow_summary[group] = fs
    res["flow_summary"] = flow_summary
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for group, fs in flow_summary.items():
        print(f"\n=== PRIMARY: attained flow-loss reduction (fraction of E||delta||^2), {group}, each arm integrated against its own t distribution ===")
        for k, v in fs.items():
            print(f"  {k:24s} " + (f"{v['attained_reduction'][0]:.4f} [{v['attained_reduction'][1]:.4f}, {v['attained_reduction'][2]:.4f}]" if 'attained_reduction' in v else f"{v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"))
    for group in summary:
        print(f"\n=== {group}: % improvement over identity, mean over {summary[group]['_n_entries']} entries [95% bootstrap CI over entries] ===")
        for m, v in summary[group].items():
            if m != "_n_entries":
                print(f"  {m:28s} {v[0]:+7.2f} [{v[1]:+7.2f}, {v[2]:+7.2f}]")


if __name__ == "__main__":
    main()
