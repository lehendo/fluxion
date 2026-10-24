#!/usr/bin/env python3
"""Per-protein recomposition of the 10 ps mean-reversion headline with its nested bootstrap (paper Section 4.1, Appendix A.3)."""
from __future__ import annotations
from fluxion.paths import DATA_ROOT
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from fluxion.analysis.noise_recompose import corrected_gain, flags, recompose
from fluxion.analysis.stats import entry_bootstrap_mean

PILOT = ("16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B").split(",")
CACHE = DATA_ROOT / "phase1_cache"; RES = DATA_ROOT / "phase1_results"; RM = RES / "replica_means"
NP = json.loads((RES / "noise_per_protein.json").read_text()); IND = json.loads((RES / "inductive_mean.json").read_text())["10ps"]; OLD = json.loads((RES / "noise_attribution.json").read_text())
PILOT3 = ("1ab1_A", "1bq8_A", "1bx7_A")
var0, var1, g0, g1 = {}, {}, {}, {}
for p in PILOT:
    c = torch.load(CACHE / f"{p}__lag1.pt", weights_only=False); m = torch.load(CACHE / f"{p}__meta.pt", weights_only=False)
    keep = ~c["is_pad"].bool() if "is_pad" in c else torch.ones(len(c["pos"]), dtype=torch.bool)
    var0[p] = float(((c["z0"] - m["mean_pos"].float()[c["pos"].long()])[keep] ** 2).mean())
    rm = torch.load(RM / f"{p}__replica_means.pt", weights_only=True); rep = c["replica"].long()
    mu = torch.stack([torch.stack([rm[r] for r in rm if r != rr]).mean(0) for rr in (1, 2, 3)])
    var1[p] = float(((c["z0"] - mu[rep - 1, c["pos"].long()])[keep] ** 2).mean())
    g0[p] = 1 - IND["V0"][p]["shrink"] / IND["V0"][p]["identity"]; g1[p] = 1 - IND["V1"][p]["shrink"] / IND["V1"][p]["identity"]
S2 = {p: NP[p]["sigma2_original"] for p in PILOT}; DR = {p: np.array(NP[p]["draws_original"]) for p in PILOT}
out = {"per_protein": {}}
print(f"{'protein':8s}{'n_res':>6s}{'sigma2_o':>11s}{'pooled':>10s}{'ratio':>7s}{'max/min':>8s}{'Var0':>8s}{'nu0 %':>7s}{'Var1':>8s}{'nu1 %':>7s}{'g1 obs':>8s}{'g1 free':>8s}")
for p in PILOT:
    nu0, nu1 = S2[p] / var0[p], S2[p] / var1[p]
    out["per_protein"][p] = {"n_res": NP[p]["n_res"], "sigma2_original": S2[p], "sigma2_pooled": NP[p]["sigma2_pooled"], "pairwise_max_over_min": NP[p]["pairwise_max_over_min"], "var_V0": var0[p], "var_V1": var1[p], "nu_V0": nu0, "nu_V1": nu1,
                             "gain_obs_V0_pct": 100 * g0[p], "gain_free_V0_pct": 100 * float(corrected_gain(g0[p], nu0)), "gain_obs_V1_pct": 100 * g1[p], "gain_free_V1_pct": 100 * float(corrected_gain(g1[p], nu1))}
    r = out["per_protein"][p]
    print(f"{p:8s}{r['n_res']:6d}{S2[p]:11.6f}{r['sigma2_pooled']:10.6f}{S2[p] / r['sigma2_pooled']:7.3f}{r['pairwise_max_over_min']:8.3f}{var0[p]:8.4f}{100 * nu0:7.2f}{var1[p]:8.4f}{100 * nu1:7.2f}{100 * g1[p]:8.2f}{r['gain_free_V1_pct']:8.2f}")
# R1, R2
vals = np.array([S2[p] for p in PILOT]); med = float(np.median(vals))
R1_bad = [p for p in PILOT if NP[p]["pairwise_max_over_min"] > 1.10]
R2_out = [p for p in PILOT if abs(S2[p] / med - 1) > 0.25]
n_res = [NP[p]["n_res"] for p in PILOT]
out["R1"] = {"proteins_with_max_over_min_gt_1.10": R1_bad}; out["R2"] = {"median": med, "min": float(vals.min()), "max": float(vals.max()), "max_over_min": float(vals.max() / vals.min()), "outside_pm25pct": R2_out, "homogeneous": not R2_out,
             "spearman_vs_n_res": [float(x) for x in spearmanr(vals, n_res)], "spearman_vs_Var0": [float(x) for x in spearmanr(vals, [var0[p] for p in PILOT])], "spearman_vs_Var1": [float(x) for x in spearmanr(vals, [var1[p] for p in PILOT])]}
pool_c = OLD["sigma_n2"]; out["pilot_constant"] = {"value": pool_c, "mean_abs_rel_dev_over_19": float(np.mean(np.abs(vals / pool_c - 1))), "ratio_range": [float(vals.min() / pool_c), float(vals.max() / pool_c)],
             "three_pilot_proteins_remeasured": {p: S2[p] for p in PILOT3}}
print(f"\nR1: proteins with max/min of the three pairwise MSEs > 1.10: {R1_bad}")
print(f"R2: sigma_o^2 median {med:.6f}, range [{vals.min():.6f}, {vals.max():.6f}] (max/min {vals.max() / vals.min():.2f}); outside +-25% of the median: {R2_out}; HOMOGENEOUS = {not R2_out}")
print(f"    Spearman with n_res {out['R2']['spearman_vs_n_res'][0]:+.2f} (p={out['R2']['spearman_vs_n_res'][1]:.3f}), with Var_V0 {out['R2']['spearman_vs_Var0'][0]:+.2f} (p={out['R2']['spearman_vs_Var0'][1]:.3f})")
print(f"    pilot constant {pool_c:.6f}: mean |relative deviation| over the 19 = {100 * out['pilot_constant']['mean_abs_rel_dev_over_19']:.1f}%, range of ratio {out['pilot_constant']['ratio_range'][0]:.2f}-{out['pilot_constant']['ratio_range'][1]:.2f}; the three pilot proteins re-measured: {PILOT3 and {p: round(S2[p], 6) for p in PILOT3}}")
# noise share and size relative to the 10 ps step (per protein; entry-bootstrap summaries)
ratio = {p: 100 * (NP[p]["mse_oA"] + NP[p]["mse_oB"] + NP[p]["mse_AB"]) / 3 / IND["V0"][p]["identity"] for p in PILOT}
out["noise_vs_step_pct"] = {"per_protein": ratio, "min": min(ratio.values()), "max": max(ratio.values()), "summary": entry_bootstrap_mean(ratio, n_boot=10000)}
out["nu_summary"] = {"V0": entry_bootstrap_mean({p: 100 * S2[p] / var0[p] for p in PILOT}, n_boot=10000), "V1": entry_bootstrap_mean({p: 100 * S2[p] / var1[p] for p in PILOT}, n_boot=10000),
                     "V0_range": [min(100 * S2[p] / var0[p] for p in PILOT), max(100 * S2[p] / var0[p] for p in PILOT)], "V1_range": [min(100 * S2[p] / var1[p] for p in PILOT), max(100 * S2[p] / var1[p] for p in PILOT)]}
sg = np.array([S2[p] for p in PILOT]); out["sigma2_summary"] = {"mean": float(sg.mean()), "median": float(np.median(sg)), "ci_mean": entry_bootstrap_mean({p: S2[p] for p in PILOT}, n_boot=10000)}
print(f"noise MSE between two extractions as % of the 10 ps step MSE (identity): min {out['noise_vs_step_pct']['min']:.1f}, max {out['noise_vs_step_pct']['max']:.1f}, mean {out['noise_vs_step_pct']['summary'][0]:.1f} [{out['noise_vs_step_pct']['summary'][1]:.1f}, {out['noise_vs_step_pct']['summary'][2]:.1f}]")
print("noise share nu (V0, own mean), percent: mean %.2f [%.2f, %.2f], range %.2f-%.2f;  (V1) mean %.2f [%.2f, %.2f], range %.2f-%.2f" % (*out["nu_summary"]["V0"], *out["nu_summary"]["V0_range"], *out["nu_summary"]["V1"], *out["nu_summary"]["V1_range"]))
print("sigma_o^2 mean %.6f [%.6f, %.6f] median %.6f" % (out["sigma2_summary"]["mean"], out["sigma2_summary"]["ci_mean"][1], out["sigma2_summary"]["ci_mean"][2], out["sigma2_summary"]["median"]))
rk = sorted(PILOT, key=lambda q: S2[q]); print("noise variance by protein, ascending:", ", ".join(f"{q}({NP[q]['n_res']}) {S2[q]:.6f}" for q in rk))
# R3/R4
fl0 = [p for p, f in zip(PILOT, flags([g0[p] for p in PILOT], [S2[p] / var0[p] for p in PILOT])) if f]; fl1 = [p for p, f in zip(PILOT, flags([g1[p] for p in PILOT], [S2[p] / var1[p] for p in PILOT])) if f]
out["R4_flagged"] = {"V0": fl0, "V1": fl1}; print(f"R4 flagged (nu >= 0.5 or rho > 1): V0 {fl0}  V1 {fl1}")
obs0 = entry_bootstrap_mean({p: 100 * g0[p] for p in PILOT}, n_boot=10000); obs1 = entry_bootstrap_mean({p: 100 * g1[p] for p in PILOT}, n_boot=10000)
free1, diff1 = recompose(PILOT, var1, g1, S2, DR, return_diff=True); free0, diff0 = recompose(PILOT, var0, g0, S2, DR, return_diff=True)
out["decomposition"] = {"V0_observed": obs0, "V1_observed": obs1, "V1_noise_free_HEADLINE": free1, "V0_noise_free": free0,
                        "noise_attributable_points": {"V0": diff0, "V1": diff1}, "old_pilot_based": {"V1_noise_free": OLD["summary_V1"]["improvement_noise_free_pct"], "V0_noise_free": OLD["summary"]["improvement_noise_free_pct"]}}
print("\nDECOMPOSITION (shrinkage improvement over identity at 10 ps, %, mean over 19 proteins [95% CI]):")
print(f"  own mean, noisy latents (as first measured)          {obs0[0]:6.2f} [{obs0[1]:6.2f}, {obs0[2]:6.2f}]")
print(f"  mean from the other replicas, noisy latents          {obs1[0]:6.2f} [{obs1[1]:6.2f}, {obs1[2]:6.2f}]")
print(f"  + noise removed, per-protein (HEADLINE, V1)          {free1[0]:6.2f} [{free1[1]:6.2f}, {free1[2]:6.2f}]   (pilot-based pooled estimate was {OLD['summary_V1']['improvement_noise_free_pct']['mean']:.2f} [{OLD['summary_V1']['improvement_noise_free_pct']['lo']:.2f}, {OLD['summary_V1']['improvement_noise_free_pct']['hi']:.2f}])")
print(f"  other order: noise removed first (own mean), per-protein {free0[0]:6.2f} [{free0[1]:6.2f}, {free0[2]:6.2f}]   (pilot-based {OLD['summary']['improvement_noise_free_pct']['mean']:.2f})")
print(f"  noise-attributable points: own mean {diff0[0]:.2f} [{diff0[1]:.2f}, {diff0[2]:.2f}]; other-replica mean {diff1[0]:.2f} [{diff1[1]:.2f}, {diff1[2]:.2f}]")
d = free1[0] - OLD["summary_V1"]["improvement_noise_free_pct"]["mean"]; print(f"  R3 reading: headline minus pilot-based 10.40 = {d:+.2f} points (old half-width 1.6) -> extrapolation {'CONFIRMED' if abs(d) <= 1.6 else 'OFF by ' + format(abs(d), '.2f') + ' points'}")
out["R3"] = {"difference_to_pilot_based": d, "confirmed": abs(d) <= 1.6}
# robustness: the headline under a common multiplier of the measured per-protein noise variance (appendix table)
rob = {}
for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    rob[str(mult)] = recompose(PILOT, var1, g1, {p: mult * S2[p] for p in PILOT}, {p: mult * DR[p] for p in PILOT}, n_boot=4000)
out["robustness_multiplier"] = rob
print("\nROBUSTNESS (common multiplier of every protein's measured noise variance):")
for k, v in rob.items(): print(f"  x{float(k):4.2f}: {v[0]:6.2f} [{v[1]:6.2f}, {v[2]:6.2f}]")
(RES / "noise_attribution_perprotein.json").write_text(json.dumps(out, indent=1))
