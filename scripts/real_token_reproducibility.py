#!/usr/bin/env python3
"""Real-token differences between re-extractions of identical frames (paper Section 3.2, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import json, re, sys
from pathlib import Path
import torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

ROOT, LAT = DATA_ROOT / "reextract_padding", DATA_ROOT
FR = (0, 100, 300, 599)
ns = torch.load(LAT / "latents/norm_stats.pt", weights_only=True); meta = json.loads((LAT / "extraction_metadata.json").read_text())


def frames(root, p):
    ds = LatentDataset(root, p, load_pair=False, norm_stats=ns, lag=1)
    idx = [i for i, (path, _) in enumerate(ds.index_map) if int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))) == 1][:600]
    return ds, idx


out = {}
for p in ("1ab1_A", "1bq8_A", "1bx7_A"):
    n = meta[p]["n_residues"]; D = {}
    for name, root in (("orig", LAT / "latents"), ("C1", ROOT / "C1"), ("C2", ROOT / "C2"), ("C3", ROOT / "C3")):
        ds, idx = frames(root, p); D[name] = {f: ds._load_frame(idx[f])[:n] for f in FR}
    ds, idx = frames(LAT / "latents", p)
    step = float(torch.stack([((ds._load_frame(idx[f + 1])[:n] - ds._load_frame(idx[f])[:n]) ** 2).mean() for f in range(0, 590, 10)]).mean())
    mse = lambda a, b: float(torch.stack([((D[a][f] - D[b][f]) ** 2).mean() for f in FR]).mean())
    out[p] = {"adjacent_step_mse": step, **{f"{a}~{b}": mse(a, b) for a, b in (("orig", "C1"), ("orig", "C2"), ("C1", "C2"), ("orig", "C3"), ("C1", "C3"))}}
    print(p, {k: round(v, 5) for k, v in out[p].items()}, flush=True)
DATA_ROOT / "phase1_results/real_token_reproducibility.json".write_text(json.dumps(out, indent=1))
