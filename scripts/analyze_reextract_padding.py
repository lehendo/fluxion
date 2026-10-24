#!/usr/bin/env python3
"""Compares padded values across the GPU re-extraction conditions (paper Section 3.1, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import argparse, json, re, sys
from pathlib import Path
import torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

TOL = 1e-3
PROTEINS = ["1ab1_A", "1bq8_A", "1bx7_A"]


def rep1_frames(ds):
    idx = [i for i, (path, _) in enumerate(ds.index_map) if int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))) == 1]
    return idx


def probes(ds, n_res, n_frames=600):
    """-> {frame: padded token [pad, d]} for first/mid/last frame of each shard among frames < n_frames, and the shard each frame belongs to."""
    idx = rep1_frames(ds)[:n_frames]
    shards = {}
    for k, i in enumerate(idx):
        shards.setdefault(ds.index_map[i][0].name, []).append(k)
    out = {}
    for name, ks in shards.items():
        for k in (ks[0], ks[len(ks) // 2], ks[-1]):
            z = ds._load_frame(idx[k]); out[k] = (name, z[n_res:].clone(), z[:n_res].clone())
    return out


def same(a, b):
    return float((a - b).abs().max()) < TOL


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=DATA_ROOT / "reextract_padding")
    ap.add_argument("--latents-dir", type=Path, default=DATA_ROOT / "latents"); ap.add_argument("--metadata-json", type=Path, default=DATA_ROOT / "extraction_metadata.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "phase1_results/reextract_padding.json")
    a = ap.parse_args(); ns = torch.load(a.latents_dir / "norm_stats.pt", weights_only=True); meta = json.loads(a.metadata_json.read_text())
    conds = sorted(d.name for d in a.root.iterdir() if d.is_dir()); res = {}
    for p in PROTEINS:
        n_res = meta[p]["n_residues"]
        orig = probes(LatentDataset(a.latents_dir, p, load_pair=False, norm_stats=ns, lag=1), n_res)
        P = {"orig": orig}
        for c in conds:
            if (a.root / c / p).exists():
                P[c] = probes(LatentDataset(a.root / c, p, load_pair=False, norm_stats=ns, lag=1), n_res)
        info = {"within": {}, "vs_orig": {}, "real_token_max_abs_diff_vs_orig": {}, "cross": {}}
        for c, pr in P.items():
            fr = sorted(pr); ref = pr[fr[0]][1]
            info["within"][c] = bool(all(same(pr[k][1], ref) for k in fr)); info.setdefault("n_shards", {})[c] = len({v[0] for v in pr.values()})
            if c != "orig":
                common = [k for k in fr if k in orig]
                info["vs_orig"][c] = {str(k): same(pr[k][1], orig[k][1]) for k in common}
                info["real_token_max_abs_diff_vs_orig"][c] = float(max((pr[k][2] - orig[k][2]).abs().max() for k in common)) if common else None
        cs = [c for c in P if c != "orig"]
        for i, c1 in enumerate(cs):
            for c2 in cs[i + 1:]:
                info["cross"][f"{c1}~{c2}"] = bool(same(P[c1][min(P[c1])][1], P[c2][min(P[c2])][1]))
        w = info["within"]; x = info["cross"]
        if all(w.get(c, False) for c in cs) and x.get("C1~C2") is False:
            reading = "R-PROCESS"
        elif x.get("C1~C2") is True and x.get("C1~C3") is False:
            reading = "R-HARDWARE"
        elif cs and all(x.values()) and not all(w[c] for c in cs):
            reading = "R-ORDER"
        elif cs and all(x.values()) and all(w[c] for c in cs):
            reading = "R-NOT-REPRODUCED"
        else:
            reading = "INTERMEDIATE"
        info["reading"] = reading; res[p] = info
        print(f"{p}: conditions {cs}; within-condition constant {w}; cross-condition same {x}; real-token max|diff| vs original {info['real_token_max_abs_diff_vs_orig']}; READING {reading}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    from collections import Counter
    print("readings:", Counter(v["reading"] for v in res.values()))


if __name__ == "__main__":
    main()
