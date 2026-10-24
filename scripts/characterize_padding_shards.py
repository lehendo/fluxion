#!/usr/bin/env python3
"""Padding constants per shard and protein (paper Section 3.1, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import numpy as np, torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents")
    ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json"); ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/padding_shards.json")
    a = ap.parse_args()
    ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True); meta = json.loads(a.metadata_json.read_text()); res = {}; first_pad = {}
    for p in PILOT:
        ds = LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1); n_res = meta[p]["n_residues"]; L = ds._load_frame(0).shape[0]
        shards = {}                                   # (replica, shard_file) -> flat indices
        for i, (path, loc) in enumerate(ds.index_map):
            rep = int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))); shards.setdefault((rep, path.name), []).append(i)
        rows = []
        for (rep, name), idx in sorted(shards.items()):
            pick = [idx[0], idx[len(idx) // 2], idx[-1]]
            pad = torch.stack([ds._load_frame(i)[n_res:] for i in pick]) if L > n_res else None
            real_step = float(((ds._load_frame(idx[1])[:n_res] - ds._load_frame(idx[0])[:n_res]) ** 2).mean()) if len(idx) > 1 else float('nan')
            if pad is None: rows.append({"replica": rep, "shard": name, "n": len(idx)}); continue
            rows.append({"replica": rep, "shard": name, "n": len(idx), "pad_mean": pad.mean((0, 1)), "within_shard_spread": float((pad - pad.mean(0, keepdim=True)).abs().max()), "real_step_mse": real_step})
        if L > n_res:
            # cluster shard means (per-element max abs difference < 1e-3)
            centers, lab = [], []
            for r in rows:
                for ci, c in enumerate(centers):
                    if float((r["pad_mean"] - c).abs().max()) < 1e-3: lab.append(ci); break
                else: centers.append(r["pad_mean"]); lab.append(len(centers) - 1)
            dmat = [float(((a - b) ** 2).mean()) for i, a in enumerate(centers) for b in centers[i + 1:]]
            first_pad[p] = (L, centers[0][0].clone())
            res[p] = {"n_res": n_res, "L": L, "n_shards": len(rows), "n_distinct_pad_constants": len(centers), "labels_by_shard_order": lab, "replicas": [r["replica"] for r in rows],
                      "max_within_shard_spread": max(r["within_shard_spread"] for r in rows), "between_constant_mse_mean": float(np.mean(dmat)) if dmat else 0.0, "real_adjacent_step_mse": float(np.nanmean([r["real_step_mse"] for r in rows]))}
            print(f"{p:8s} n_res {n_res:3d} L {L} shards {len(rows):3d} distinct pad constants {len(centers):2d}  within-shard spread {res[p]['max_within_shard_spread']:.2e}  between-constant mse {res[p]['between_constant_mse_mean']:.3f} (real adjacent step {res[p]['real_adjacent_step_mse']:.4f})  labels {lab}", flush=True)
        else:
            res[p] = {"n_res": n_res, "L": L, "no_padding": True}
    # are constants shared across proteins of the same bucket? compare the first padded token (first shard) pairwise
    byL = {}
    for p, (L, v) in first_pad.items(): byL.setdefault(L, []).append((p, v))
    res["_cross_protein"] = {str(L): {"proteins": [p for p, _ in vs], "pairwise_mse_first_pad_token": [[float(((a - b) ** 2).mean()) for _, b in vs] for _, a in vs]} for L, vs in byL.items() if len(vs) > 1}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    for L, d in res["_cross_protein"].items():
        m = np.array(d["pairwise_mse_first_pad_token"]); print(f"bucket {L}: {len(d['proteins'])} proteins, pairwise mse of the first shard's first padded token: median {np.median(m[np.triu_indices_from(m, 1)]):.4f}, min {m[np.triu_indices_from(m, 1)].min():.4f}")


if __name__ == "__main__":
    main()
