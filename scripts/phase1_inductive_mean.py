#!/usr/bin/env python3
"""Probes re-fit with the mean of other replicas and of the training proteins (paper Sections 3.3 and 4.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
from fluxion.analysis.probes import fit_linear_diag, fit_linear_full, predict_linear_diag, predict_linear_full, row_sqerr, train_mlp
from fluxion.analysis.stats import entry_bootstrap_mean, entry_groups
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
LAGS = {"10ps": 1, "100ps": 10, "1ns": 100, "10ns": 1000}
MLP_LAGS = ("10ps", "1ns")


def replica_means(latents_dir, p, ns, cache_dir):
    f = cache_dir / f"{p}__replica_means.pt"
    if f.exists():
        return torch.load(f, weights_only=True)
    ds = LatentDataset(latents_dir, p, load_pair=False, norm_stats=ns, lag=1)
    rep = {}
    for i, (path, _) in enumerate(ds.index_map):
        rep.setdefault(int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))), []).append(i)
    out = {}
    for r, idx in rep.items():
        sel = idx[::10]; acc = None
        for i in sel:
            z = ds._load_frame(i).float(); acc = z.clone() if acc is None else acc + z
        out[r] = acc / len(sel)
    torch.save(out, f); return out


def centered(cell, mode, rmeans, global_mean):
    pos, rep = cell["pos"].long(), cell["replica"].long()
    if mode == "V0":
        raise RuntimeError
    if mode == "V1":
        mu = torch.stack([torch.stack([rmeans[r] for r in rmeans if r != rr]).mean(0) for rr in (1, 2, 3)])   # [3(replica 1..3), L, d]: mean of the other replicas
        m = mu[rep - 1, pos]
    else:
        m = global_mean.unsqueeze(0)
    return cell["z0"] - m, cell["z1"] - m


def run_fold(rows, held, groups, device, seed, with_mlp):
    train_entries = [e for e in groups if e != held]; rng = np.random.default_rng(seed)
    inner = set(rng.choice(train_entries, size=2, replace=False).tolist())
    tr = [p for e in train_entries for p in groups[e]]; fit = [p for e in train_entries if e not in inner for p in groups[e]]; inn = [p for e in train_entries if e in inner for p in groups[e]]
    cat = lambda ps, k: torch.cat([rows[p][k] for p in ps])
    z0, z1 = cat(tr, "z0c"), cat(tr, "z1c"); mean_vec = z1.mean(0, keepdim=True)
    a, k, _ = fit_linear_diag(z0, z1); A, b = fit_linear_full(z0, z1); alpha = float((z0 * z1).sum() / (z0 * z0).sum())
    mlp = train_mlp(cat(fit, "z0c"), cat(fit, "z1c"), cat(inn, "z0c"), cat(inn, "z1c"), device=device, seed=seed)[0] if with_mlp else None
    out = {}
    for p in groups[held]:
        x, y = rows[p]["z0c"], rows[p]["z1c"]
        with torch.no_grad():
            pr = {"identity": x, "mean": mean_vec.expand_as(y), "shrink": alpha * x, "linear_diag": predict_linear_diag(x, a, k), "linear_full": predict_linear_full(x, A, b)}
            if mlp is not None: pr["mlp"] = mlp(x)
        out[p] = {n: float(row_sqerr(v, y).mean()) for n, v in pr.items()}; out[p]["_alpha"] = alpha
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DATA_ROOT / "phase1_cache"); ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/inductive_mean.json"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True); groups = entry_groups(PILOT)
    cdir = a.out.parent / "replica_means"; cdir.mkdir(parents=True, exist_ok=True)
    rm = {p: replica_means(a.latents_dir, p, ns, cdir) for p in PILOT}; print("replica means ready", flush=True)
    res = {}
    for lname, lag in LAGS.items():
        cells = {p: torch.load(a.cache / f"{p}__lag{lag}.pt", weights_only=False) for p in PILOT}
        meta = {p: torch.load(a.cache / f"{p}__meta.pt", weights_only=False) for p in PILOT}
        for p in PILOT:                                              # real tokens only
            c = cells[p]; keep = ~c["is_pad"].bool() if "is_pad" in c else torch.ones(len(c["pos"]), dtype=torch.bool)
            for k in ("z0", "z1", "pos", "replica"): c[k] = c[k][keep]
        for mode in ("V0", "V1", "V2"):
            per = {}
            for held in groups:
                if mode == "V2":
                    tr = [p for e in groups if e != held for p in groups[e]]
                    gm = torch.cat([cells[p]["z0"] for p in tr]).mean(0)
                rows = {}
                for p in PILOT:
                    c = cells[p]
                    if mode == "V0":
                        mp, pos = meta[p]["mean_pos"].float(), c["pos"].long(); rows[p] = {"z0c": c["z0"] - mp[pos], "z1c": c["z1"] - mp[pos]}
                    else:
                        z0c, z1c = centered(c, mode, rm[p], gm if mode == "V2" else None); rows[p] = {"z0c": z0c, "z1c": z1c}
                fold = run_fold(rows, held, groups, a.device, a.seed + lag, lname in MLP_LAGS)
                per.update(fold)
                print(f"[{lname} {mode}] held-out {held}: alpha {next(iter(fold.values()))['_alpha']:.3f}", flush=True)
            res.setdefault(lname, {})[mode] = per
            a.out.write_text(json.dumps(res))
    # summary
    print("\nacross proteins: % over identity (mean [95% CI over entries]) and identity/mean error ratio")
    summ = {}
    for lname in LAGS:
        for mode in ("V0", "V1", "V2"):
            per = res[lname][mode]; row = {}
            for n in ("shrink", "linear_diag", "linear_full", "mlp", "mean"):
                if n in next(iter(per.values())):
                    v = {p: 100 * (1 - per[p][n] / per[p]["identity"]) for p in PILOT}; row[n] = dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean(v, n_boot=5000)))
            ratio = {p: per[p]["identity"] / per[p]["mean"] for p in PILOT}; row["id_over_mean"] = dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean(ratio, n_boot=5000)))
            row["alpha"] = float(np.mean([per[p]["_alpha"] for p in PILOT])); summ.setdefault(lname, {})[mode] = row
            print(f"  {lname:6s} {mode}: shrink {row['shrink']['mean']:+6.2f} [{row['shrink']['lo']:+6.2f},{row['shrink']['hi']:+6.2f}]  linear_full {row['linear_full']['mean']:+6.2f}"
                  + (f"  mlp {row['mlp']['mean']:+6.2f}" if "mlp" in row else "") + f"  id/mean {row['id_over_mean']['mean']:.3f} [{row['id_over_mean']['lo']:.3f},{row['id_over_mean']['hi']:.3f}]  alpha {row['alpha']:.3f}")
    res["_summary"] = summ; a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
