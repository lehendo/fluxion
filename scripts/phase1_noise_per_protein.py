#!/usr/bin/env python3
"""Per-protein extraction-noise variance from two re-extractions (paper Section 3.2, Appendix A.1)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import json, re, sys
from pathlib import Path
import numpy as np, torch
from fluxion.data_pipeline.dataset import LatentDataset, _REPLICA_RE

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
LAT, ROOT = DATA_ROOT, DATA_ROOT / "noise_reextract"
FRAMES = list(range(0, 10000, 50)); N_BOOT = 2000
ns = torch.load(LAT / "latents/norm_stats.pt", weights_only=True); meta = json.loads((LAT / "extraction_metadata.json").read_text())


def rep1(root, p):
    ds = LatentDataset(root, p, load_pair=False, norm_stats=ns, lag=1)
    idx = [i for i, (path, _) in enumerate(ds.index_map) if int(re.sub(r"\D", "", _REPLICA_RE.search(path.name).group(1))) == 1]
    return ds, idx


def real_frames(root, p, n, which):
    ds, idx = rep1(root, p); sel = which(idx)
    return torch.stack([ds._load_frame(i)[:n].float() for i in sel])


def main() -> None:
    out = {}
    for p in PILOT:
        n = meta[p]["n_residues"]
        o = real_frames(LAT / "latents", p, n, lambda idx: [idx[f] for f in FRAMES])
        A = real_frames(ROOT / "A", p, n, lambda idx: idx); B = real_frames(ROOT / "B", p, n, lambda idx: idx)
        assert A.shape == B.shape == o.shape == (len(FRAMES), n, 384), (p, A.shape, B.shape, o.shape)
        mse = lambda x, y: ((x - y) ** 2).mean(dim=(1, 2)).double().numpy()
        oA, oB, AB = mse(o, A), mse(o, B), mse(A, B)
        sig_o = (oA + oB - AB) / 2; pooled = (oA + oB + AB) / 6
        rng = np.random.default_rng(0); draws_o, draws_pool = [], []
        for _ in range(N_BOOT):
            j = rng.integers(0, len(FRAMES), len(FRAMES)); draws_o.append(float(sig_o[j].mean())); draws_pool.append(float(pooled[j].mean()))
        half = len(FRAMES) // 2
        out[p] = {"n_res": n, "mse_oA": float(oA.mean()), "mse_oB": float(oB.mean()), "mse_AB": float(AB.mean()),
                  "sigma2_original": float(sig_o.mean()), "sigma2_pooled": float(pooled.mean()),
                  "sigma2_original_ci": [float(np.percentile(draws_o, 2.5)), float(np.percentile(draws_o, 97.5))], "sigma2_pooled_ci": [float(np.percentile(draws_pool, 2.5)), float(np.percentile(draws_pool, 97.5))],
                  "sigma2_original_first_half": float(sig_o[:half].mean()), "sigma2_original_second_half": float(sig_o[half:].mean()),
                  "draws_original": draws_o, "draws_pooled": draws_pool,
                  "pairwise_max_over_min": float(max(oA.mean(), oB.mean(), AB.mean()) / min(oA.mean(), oB.mean(), AB.mean()))}
        r = out[p]; print(f"{p:7s} n_res {n:3d}  MSE oA {r['mse_oA']:.5f} oB {r['mse_oB']:.5f} AB {r['mse_AB']:.5f}  sigma2_orig {r['sigma2_original']:.6f} {r['sigma2_original_ci']}  pooled {r['sigma2_pooled']:.6f}  max/min {r['pairwise_max_over_min']:.3f}", flush=True)
    d = LAT / "phase1_results"; d.mkdir(parents=True, exist_ok=True); (d / "noise_per_protein.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
