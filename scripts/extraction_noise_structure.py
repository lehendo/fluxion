#!/usr/bin/env python3
"""Structure of the run-to-run difference of re-extracted latents: offset share and cosines (paper Section 3.2, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import json, re, sys
from pathlib import Path
import torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

ROOT, LAT = DATA_ROOT / "reextract_padding", DATA_ROOT
ns = torch.load(LAT / "latents/norm_stats.pt", weights_only=True); meta = json.loads((LAT / "extraction_metadata.json").read_text())


def get(root, p, n, frames):
    ds = LatentDataset(root, p, load_pair=False, norm_stats=ns, lag=1)
    idx = [i for i, (path, _) in enumerate(ds.index_map) if int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))) == 1][:600]
    return torch.stack([ds._load_frame(idx[f])[:n].flatten() for f in frames])


out = {}
for p in ("1ab1_A", "1bq8_A", "1bx7_A"):
    n = meta[p]["n_residues"]; fr = list(range(0, 600, 20)) + [1, 21, 101, 301]
    a, b = get(ROOT / "C1", p, n, fr), get(ROOT / "C2", p, n, fr); d = a - b
    main = d[:30]; mse = float((main ** 2).mean()); off = main.mean(0, keepdim=True); off_share = float((off ** 2).mean() / mse)
    cos = torch.nn.functional.cosine_similarity(main[:, None, :], main[None, :, :], dim=-1); cosm = float((cos.sum() - cos.diag().sum()) / (30 * 29))
    adj = float(torch.nn.functional.cosine_similarity(d[0], d[30], dim=0).item())          # frame 0 vs frame 1 (adjacent frames)
    out[p] = {"mse_C1_minus_C2": mse, "offset_share_of_mse": off_share, "mean_pairwise_cosine_between_frames": cosm, "cosine_frame0_vs_frame1": adj}
    print(p, {k: round(v, 4) for k, v in out[p].items()}, flush=True)
DATA_ROOT / "phase1_results/extraction_noise_structure.json".write_text(json.dumps(out, indent=1))
