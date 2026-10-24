#!/usr/bin/env python3
"""Gradient signal-to-noise table and figure (paper Section 6.2)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("paper_results/phase1"); d = json.loads((R / "gradsnr.json").read_text())
MB = d["mb_size"]; mu2, tr = d["mu2"], d["tr_cov"]; k = d["kstar"]; jk = d["jackknife"]["kstar_leave_one_out"]
jk_se = float(np.sqrt((len(jk) - 1) / len(jk) * ((np.asarray(jk) - np.mean(jk)) ** 2).sum()))
snr = lambda n: float(np.sqrt(mu2 / (tr / (n / MB))))
lines = ["# E3 gradient SNR: full table", "",
         f"Checkpoint step {d['ckpt_step']} of the ORIGINAL run (raw weights, 39.86M parameters, bf16 autocast, train mode), {d['microbatches']} microbatches of {MB} samples, training loss, random t.",
         "Original run's effective batch CONFIRMED from its config (slurm/training/train_full_run.sh: torchrun --nproc_per_node=4, --batch-size 4, DDP-averaged gradients => 16; LR 1e-4, gate LR 1e-3).", "",
         "## Scalars", "",
         f"- ||E g|| = {d['norm_Eg']:.4e};  mu2 = ||E g||^2 = {mu2:.4e} (jackknife SE {d['jackknife']['mu2_se']:.2e})",
         f"- tr Cov(g) per microbatch = {tr:.4e} (jackknife SE {d['jackknife']['tr_se']:.2e});  per-microbatch std = {d['std_g_per_microbatch']:.4e}",
         f"- k* = trCov/||Eg||^2 = {k:.2f} microbatches = {k * MB:.0f} samples.  Leave-one-chunk-out k* (8 chunks): " + ", ".join(f"{x:.1f}" for x in jk) + f"  (range {min(jk):.1f}-{max(jk):.1f}); jackknife SE of k* = {jk_se:.2f}, i.e. approx. 95% interval [{k - 1.96 * jk_se:.1f}, {k + 1.96 * jk_se:.1f}] microbatches = [{(k - 1.96 * jk_se) * MB:.0f}, {(k + 1.96 * jk_se) * MB:.0f}] samples",
         f"- gate/adaLN-zero group ({d['gate_group']['n_params']:,} params): k* = {d['gate_group']['kstar']:.1f} microbatches (jackknife range {min(d['gate_group']['jackknife']['kstar_leave_one_out']):.1f}-{max(d['gate_group']['jackknife']['kstar_leave_one_out']):.1f})",
         f"- pre-stated verdict (log, E3): 4 < k* <= 64 microbatches => AMBIGUOUS", "",
         "## SNR versus batch size (SNR = 1 at k*)", "", "| samples per step | microbatches k | SNR | note |", "|---|---|---|---|"]
notes = {8: "NOT used: below noise floor", 16: "the ORIGINAL run (effective batch 16)", 64: "", 103: "SNR = 1 (k*)", 256: "controlled-experiment batch", 1024: "(measured group)"}
for n in (4, 8, 16, 64, 103, 256, 1024):
    lines.append(f"| {n} | {n / MB:g} | {snr(n):.3f} | {notes.get(n, '')} |")
lines += ["", "Measured directly (empirical mean squared norm of k-averaged gradients vs ||Eg||^2 + trCov/k):", "", "| k | n groups | empirical | predicted | ratio |", "|---|---|---|---|---|"]
for kk in ("1", "4", "16", "64", "256"):
    e, p = d["empirical_mean_sq_norm_avg_of_k"][kk], d["predicted_mean_sq_norm_avg_of_k"][kk]
    lines.append(f"| {kk} | {d['n_groups'][kk]} | {e:.4e} | {p:.4e} | {e / p:.3f} |")
lines += ["", "Fraction of coordinates with per-coordinate SNR > 1 at k = 1, 4, 16, 64, 256: " + ", ".join(f"{100 * v:.2f}%" for v in d["fraction_coordinates_with_snr_above_1_at_k"].values()), "",
          "Caveats: measured at the ORIGINAL run's final (collapsed) checkpoint; SNR at initialization or for the 9M flow models can differ. The original 192k run is not a data point for any controlled comparison."]
(R / "e3_table.md").write_text("\n".join(lines) + "\n")
n = np.logspace(0.5, 3.2, 100)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(n, [snr(x) for x in n], color="#1f6fd0", lw=2)
lo, hi = min(jk), max(jk)
ax.fill_between(n, [np.sqrt(mu2 / (tr / (x / MB)) ) * np.sqrt(k / hi) for x in n], [np.sqrt(mu2 / (tr / (x / MB))) * np.sqrt(k / lo) for x in n], color="#1f6fd0", alpha=0.18, lw=0, label="jackknife range of k*")
ax.axhline(1, color="black", lw=0.8, ls=":")
for nn, lab, c in ((8, "batch 8", "#c0392b"), (16, "original run (16)", "#8a8f98"), (256, "controlled arm (256)", "#2e9e5b")):
    ax.plot([nn], [snr(nn)], "o", color=c, ms=7); ax.annotate(f"{lab}: {snr(nn):.2f}", (nn, snr(nn)), textcoords="offset points", xytext=(6, -12 if nn != 256 else 6), fontsize=8, color=c)
ax.set_xlabel("samples per optimizer step"); ax.set_ylabel("gradient SNR  ||E g|| / sd(batch gradient)"); ax.set_title("Gradient SNR at the original run's final checkpoint", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); (R / "figs").mkdir(exist_ok=True); fig.savefig(R / "figs/fig5_gradient_snr.png", dpi=200); fig.savefig(R / "figs/fig5_gradient_snr.pdf")
print((R / "e3_table.md").read_text())
