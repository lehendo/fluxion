#!/usr/bin/env python3
"""Pilot-based estimate of the noise share of the 10 ps shrinkage gain, superseded by noise_attribution_perprotein.py (paper Section 3.2)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import json, sys
from pathlib import Path
import numpy as np, torch
from fluxion.analysis.stats import entry_bootstrap_mean

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
CACHE = DATA_ROOT / "phase1_cache"; RES = DATA_ROOT / "phase1_results"
rt = json.loads((RES / "real_token_reproducibility.json").read_text())
sigma_n2 = float(np.mean([0.5 * np.mean([v["C1~C2"], v["orig~C1"], v["orig~C2"]]) for v in rt.values()]))
IND = json.loads((RES / "inductive_mean.json").read_text())["10ps"]
ind = IND["V0"]
RM = RES / "replica_means"
rows = {}
rows1 = {}
for p in PILOT:
    c = torch.load(CACHE / f"{p}__lag1.pt", weights_only=False); m = torch.load(CACHE / f"{p}__meta.pt", weights_only=False)
    keep = ~c["is_pad"].bool() if "is_pad" in c else torch.ones(len(c["pos"]), dtype=torch.bool)
    z = (c["z0"] - m["mean_pos"].float()[c["pos"].long()])[keep]; var = float((z ** 2).mean())
    g = 100 * (1 - ind[p]["shrink"] / ind[p]["identity"]) / 100       # fraction
    nu = sigma_n2 / var; alpha = 1 - 2 * g; rho = alpha / (1 - nu); g0 = (1 - rho) / 2
    rows[p] = {"var_per_element": var, "nu": nu, "improvement_observed_pct": 100 * g, "improvement_noise_free_pct": 100 * g0, "noise_attributable_points": 100 * (g - g0)}
    # V1: the protein's mean estimated from the OTHER replicas only (the same noise variance; the variance of the V1-centered rows includes the mean-estimation error)
    rm = torch.load(RM / f"{p}__replica_means.pt", weights_only=True); rep = c["replica"].long()
    mu = torch.stack([torch.stack([rm[r] for r in rm if r != rr]).mean(0) for rr in (1, 2, 3)])
    z1v = (c["z0"] - mu[rep - 1, c["pos"].long()])[keep]; var1 = float((z1v ** 2).mean())
    g1 = 1 - IND["V1"][p]["shrink"] / IND["V1"][p]["identity"]; nu1 = sigma_n2 / var1; alpha1 = 1 - 2 * g1; rho1 = alpha1 / (1 - nu1); g01 = (1 - rho1) / 2
    rows1[p] = {"var_per_element": var1, "nu": nu1, "improvement_observed_pct": 100 * g1, "improvement_noise_free_pct": 100 * g01, "noise_attributable_points": 100 * (g1 - g01)}
    print(f"{p:7s} var {var:.4f}  nu {100 * nu:5.1f}%  observed {100 * g:5.2f}%  noise-free {100 * g0:5.2f}%  noise-attributable {100 * (g - g0):5.2f} points", flush=True)
summ = {k: dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean({p: rows[p][k] for p in PILOT}, n_boot=10000))) for k in ("nu", "improvement_observed_pct", "improvement_noise_free_pct", "noise_attributable_points")}
print(f"\nsigma_n^2 = {sigma_n2:.5f} per element (pooled from three proteins; ASSUMED the same for all 19)")
for k, v in summ.items():
    print(f"  {k:28s} {v['mean']:7.3f} [{v['lo']:7.3f}, {v['hi']:7.3f}]")
summ1 = {k: dict(zip(("mean", "lo", "hi"), entry_bootstrap_mean({p: rows1[p][k] for p in PILOT}, n_boot=10000))) for k in ("nu", "improvement_observed_pct", "improvement_noise_free_pct", "noise_attributable_points")}
print("\nV1 (mean from the other replicas):")
for k, v in summ1.items():
    print(f"  {k:28s} {v['mean']:7.3f} [{v['lo']:7.3f}, {v['hi']:7.3f}]")
(RES / "noise_attribution.json").write_text(json.dumps({"sigma_n2": sigma_n2, "per_protein": rows, "summary": summ, "per_protein_V1": rows1, "summary_V1": summ1}, indent=1))
