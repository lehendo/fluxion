#!/usr/bin/env python3
"""
scripts/padding_snr.py   (analytical; no GPU)

How much does including Chai-1 bucket padding in the loss change the gradient SNR of the ORIGINAL configuration (batch 16, 19 proteins sampled uniformly over frames)?

IMPORTANT: E3's measured SNR (0.39 at batch 16, k* = 25.8 microbatches) was computed with the ORIGINAL loss (collate pad_to=None, batch size 1 per sample: the mask covers only positions beyond the
bucket, so bucket padding counts as real tokens). The measurement therefore ALREADY includes any padding effect. The question is what it would be with padding masked.

Model (assumptions stated, this is an ESTIMATE, not a measurement):
  A1  padded tokens within a shard have z0 = z1 (constant), target 0, and the model fits them: they contribute ~zero gradient but are counted in the per-sample normaliser (L_bucket tokens).
  A2  real-token per-sample mean gradient has the same mean s for every protein (homogeneous signal); per-sample noise variance sigma^2 (variant 1) or sigma^2 / N_real (variant 2: token-independent noise).
  A3  samples are drawn uniformly over frames; every protein has 3 x 10,001 frames (equal weight).
  A4  padding-constant changes at shard boundaries (every 500 frames) give rare pairs whose padded tokens carry a large, irreducible target: modelled as an extra variance term proportional to the
      loss mass (gradient is linear in the residual, so its variance ~ |residual|^2 ~ loss mass), with the pooled rate and size measured on 6 proteins (results/phase1/shard_boundaries_b.json).
Facts used: a common rescaling of the gradient (dilution by the real-token fraction f) multiplies the mean AND the noise by f, so it leaves SNR unchanged; only UNEQUAL per-sample weights and extra noise change SNR.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path("results/phase1")
meta = json.loads((R / "extraction_metadata.json").read_text())
sb = json.loads((R / "shard_boundaries_b.json").read_text())
prot = [p for p in meta if not p.startswith("_")]
n_res = np.array([meta[p]["replicas"]["1"]["n_residues"] for p in prot], float)
L = np.array([meta[p]["replicas"]["1"]["L_bucket"] for p in prot], float)
frames = np.array([sum(r["traj_frames"] for r in meta[p]["replicas"].values()) for p in prot], float)
pi = frames / frames.sum()
f = n_res / L

# ---- weighting effect: loss normalised by L_bucket (padding counted) vs by n_res (masked)
w_pad, w_mask = f, np.ones_like(f)
v1 = (pi @ w_pad) / np.sqrt(pi @ w_pad**2)                                            # homogeneous noise
v2 = (pi @ w_pad) / np.sqrt(pi @ (w_pad**2 / n_res)) * np.sqrt(pi @ (1 / n_res))        # noise variance ~ 1/N_real, relative to the masked case
# ---- boundary-shift term (A4)
chg = sum(v["n_boundaries_with_pad_change"] for v in sb.values()); nb = sum(v["n_boundaries"] for v in sb.values())
rate = chg / nb                                                                          # pad-change events per shard boundary
pad_mse_changed = sum(v["boundary_pad_mse_mean"] * v["n_boundaries"] for v in sb.values()) / chg
real_mse = float(np.mean([v["inside_step_mse"] for v in sb.values()]))
q = rate / 500.0                                                                         # per lag-1 pair
extra = float(pi @ (q * (L - n_res) * pad_mse_changed / (n_res * real_mse)))             # extra variance as a fraction of baseline variance
extra_max = float(np.max(q * (L - n_res) * pad_mse_changed / (n_res * real_mse)))
snr_measured_b16, k_star_samples = 0.394, 103
sqrt_f = float(np.sqrt(pi @ f))
out = {
    "n_proteins": len(prot), "real_token_fraction_f": {p: float(x) for p, x in zip(prot, f)}, "f_range": [float(f.min()), float(f.max())], "f_mean": float(pi @ f),
    "padding_fraction_range": [float(1 - f.max()), float(1 - f.min())],
    "claimed_sqrt_f_factor": sqrt_f,
    "weighting_factor_SNR_padded_over_masked": {"homogeneous_noise": float(v1), "token_independent_noise": float(v2)},
    "boundary_term": {"pad_change_rate_per_boundary": rate, "pad_mse_at_changed_boundary": pad_mse_changed, "real_step_mse": real_mse, "per_pair_probability": q,
                      "extra_variance_fraction_frame_weighted": extra, "extra_variance_fraction_max_protein": extra_max, "n_boundaries_measured": nb, "n_changes_measured": chg},
}
var_factor = 1 + extra
for name, fac in (("weighting_only_homogeneous", v1), ("weighting_only_token_independent", v2), ("weighting+boundary_homogeneous", v1 / np.sqrt(var_factor)), ("weighting+boundary_token_indep", v2 / np.sqrt(var_factor)), ("ARJUN_sqrt_f_formula", sqrt_f)):
    masked = snr_measured_b16 / fac                    # SNR_measured(padded) = fac * SNR_masked
    out.setdefault("implied_masked_snr_at_batch_16", {})[name] = {"factor_padded_over_masked": float(fac), "masked_snr_b16": float(masked), "k_star_samples_masked": float(k_star_samples * fac**2)}
(R / "padding_snr.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "real_token_fraction_f"}, indent=1))
