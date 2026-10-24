#!/usr/bin/env python3
"""Regenerates manifest.json and MANIFEST.md: the configs and seeds of every training run and the script behind every result the scripts write to paper_results/."""
from __future__ import annotations

import ast
import fnmatch
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = {"architecture": "FluxionDiT, d_model 256, 6 layers, 8 heads, d_ff 1024 (9.15M parameters)", "latents": "13 training proteins (12 PDB entries); 5 held-out proteins (4 entries: 16pk, 1aol, 1bq8, 1bxy)",
          "optimizer": "AdamW, weight decay 1e-5, gate learning rate = 10x the base rate, cosine decay to 0.1x, EMA of the weights", "t_clip": "t clamped to [0.01, 0.99]", "micro_batch": 8,
          "script": "src/fluxion/training/train_t3.py", "milestone_checkpoints": "25, 50, 75 and 100% of the steps"}
B256 = {"steps": 6400, "accum": 32, "effective_batch": 256, "lr": 5.657e-4, "warmup_steps": 350, "ema_decay": 0.999}
B16 = {"steps": 102400, "accum": 2, "effective_batch": 16, "lr": 1.414e-4, "warmup_steps": 5600, "ema_decay": 0.99994}
RUNS = [
    {"run": "original run", "paper": "Section 7.1", "script": "src/fluxion/training/train.py", "seed": "not recorded (default RNG state)", "config": {"gpus": "4 x A100, DDP", "proteins": "all 19", "per_rank_batch": 4, "lr": 1e-4,
     "formulation": "temporal pairs, padding counted, uncentered latents, no held-out split", "stopped": "wall-time checkpoint at step 192,325 (epoch 4 of a 200-epoch schedule)", "parameters": "about 40M"}},
    {"run": "uniform (seed 0)", "paper": "Sections 6.1, 6.3", "seed": 0, "config": {**COMMON, **B256, "t_dist": "uniform"}},
    {"run": "beta (seed 0)", "paper": "Sections 6.1, 6.3, 6.4", "seed": 0, "config": {**COMMON, **B256, "t_dist": "Beta(0.25, 0.25)"}},
    {"run": "uniform (seed 1)", "paper": "Section 6.1, Appendix A.7", "seed": 1, "config": {**COMMON, **B256, "t_dist": "uniform"}},
    {"run": "uniform (seed 2)", "paper": "Section 6.1, Appendix A.7", "seed": 2, "config": {**COMMON, **B256, "t_dist": "uniform"}},
    {"run": "3c", "paper": "Sections 6.1 to 6.3, 7.1", "seed": 0, "config": {**COMMON, **B16, "t_dist": "uniform", "divergence_guard": "halt if the loss exceeds 2x the v=0 loss over a 12,800-sample window"}},
    {"run": "T3a", "paper": "Section 7.1", "seed": 0, "config": {**COMMON, **B16, "t_dist": "uniform", "padding": "counted in loss and attention", "centering": "uncentered"}},
    {"run": "T3b (seed 0)", "paper": "Section 7.1", "seed": 0, "config": {**COMMON, **B16, "t_dist": "uniform", "padding": "counted in loss and attention", "centering": "centered"}},
    {"run": "T3b (seed 1)", "paper": "Section 7.1", "seed": 1, "config": {**COMMON, **B16, "t_dist": "uniform", "padding": "counted in loss and attention", "centering": "centered"}},
    {"run": "T3c", "paper": "Section 7.1", "seed": 0, "config": {**COMMON, **B16, "t_dist": "uniform", "padding": "masked", "centering": "uncentered"}},
] + [
    {"run": f"stability diagnostic {c} (seeds 0 and 1)", "paper": "Section 6.4, Appendix A.7", "seed": "0 and 1", "config": {**COMMON, **B256, "steps": "6400 scheduled, stopped after 600", "t_dist": td, "lr": lr}}
    for c, td, lr in (("A", "uniform", 5.657e-4), ("B", "Beta(0.25, 0.25), clamped", 5.657e-4), ("C", "Beta(0.25, 0.25), truncated", 5.657e-4), ("D", "Beta(0.25, 0.25), clamped", 2.828e-4), ("E", "uniform", 2.828e-4))
]
# result files and what produces them (glob relative to paper_results/ -> producer); every file in paper_results/ must match one
PRODUCERS = {
    "tables.md": "scripts/make_tables.py",
    "phase1/extraction_metadata.json": "scripts/build_extraction_metadata.py",
    "phase1/b1/*": "scripts/phase1b_cartesian.py", "phase1/b1_ridge/*": "scripts/phase1b_cartesian.py (ridge probe cells)", "phase1/b1_10ps/*": "scripts/phase1b_cartesian_10ps.py",
    "phase1/b1_10ps_controls/*": "scripts/phase1b_controls_10ps.py", "phase1/b1_10ps_stability/*": "scripts/phase1b_stability_10ps.py", "phase1/b1_10ps_tf/*": "scripts/phase1b_templatefeat_10ps.py",
    "phase1/b1_*report*.txt": "scripts/phase1b_report_10ps.py, scripts/phase1b_controls_report.py, scripts/phase1b_stability_report.py, scripts/phase1b_stability_null.py, scripts/phase1b_tf_report.py",
    "phase1/b1_10ps_controls_topm.txt": "scripts/phase1b_controls_topm.py", "phase1/b1_10ps_stability_null.txt": "scripts/phase1b_stability_null.py",
    "phase1/atlas/*": "scripts/verify_atlas_refetch.py", "phase1/atlas10ps/*": "scripts/verify_atlas_10ps.py, scripts/verify_atlas_local_alignment.py", "phase1/acf.json": "scripts/phase1a_acf.py",
    "phase1/sweep/*": "scripts/phase1a_fit.py", "phase1/ladder.json": "scripts/phase1a_ladder.py", "phase1/headline.json": "scripts/phase1_headline.py", "phase1/gradsnr*": "scripts/phase1_gradsnr.py",
    "phase1/e3_table.md": "scripts/e3_table.py", "phase1/extraction_noise_structure.json": "scripts/extraction_noise_structure.py", "phase1/inductive_mean.json": "scripts/phase1_inductive_mean.py",
    "phase1/noise_attribution.json": "scripts/noise_attribution.py", "phase1/noise_attribution_perprotein.*": "scripts/noise_attribution_perprotein.py", "phase1/noise_per_protein.json": "scripts/phase1_noise_per_protein.py",
    "phase1/padding_all19.json": "scripts/phase1_padding_all19.py", "phase1/padding_hypothesis_check.json": "scripts/check_padding_hypothesis.py", "phase1/padding_shards.json": "scripts/characterize_padding_shards.py",
    "phase1/padding_snr.json": "scripts/padding_snr.py", "phase1/real_token_reproducibility.json": "scripts/real_token_reproducibility.py", "phase1/reextract_padding.json": "scripts/analyze_reextract_padding.py",
    "phase1/shard_boundaries_b.json": "scripts/check_shard_boundaries.py", "phase1/template_feature_verification.json": "scripts/verify_template_features.py", "phase1/vtheta_summary.json": "scripts/phase1a_vtheta_report.py",
    "phase3/3c_readings_100.json": "scripts/phase3_3c_readings.py", "phase3/config_*.json": "training run configuration (see the runs table)", "phase3/diag/*": "stability diagnostic runs A to E (see the runs table)",
    "phase3/eval_*.json": "scripts/phase3_eval.py", "phase3/steps_*.json": "scripts/phase3_eval.py (Euler-step sweep)", "phase3/loss_log_*.csv": "training logs of the runs in the runs table", "phase3/seeds/*": "training logs of the seed runs",
    "phase3/matched_linear_curve.json": "scripts/phase3_matched_linear_curve.py", "phase3/matched_report.txt": "scripts/phase3_matched_report.py", "phase3/phase3_report.json": "scripts/phase3_report.py",
    "phase3/predictions.json": "scripts/phase3_predict.py", "phase3/seed_report.txt": "scripts/phase3_seed_report.py",
}


def script_info(path: Path) -> dict:
    src = path.read_text(); tree = ast.parse(src)
    doc = (ast.get_docstring(tree) or "").strip().splitlines()[0] if ast.get_docstring(tree) else ""
    seeds = sorted(set(re.findall(r"default_rng\((\d+)\)|manual_seed\((\d+)\)|np\.random\.seed\((\d+)\)", src) and [a or b or c for a, b, c in re.findall(r"default_rng\((\d+)\)|manual_seed\((\d+)\)|np\.random\.seed\((\d+)\)", src)]))
    m = re.search(r'add_argument\("--seed"[^)]*default=(\d+)', src)
    if m: seeds = sorted(set(seeds) | {m.group(1)})
    outs = sorted(set(re.findall(r'["\']([A-Za-z0-9_\-./]+\.(?:json|txt|csv|md))["\']', src)))
    return {"script": f"scripts/{path.name}", "artifact": doc, "seeds": seeds, "files_named": outs}


def main() -> None:
    scripts = [script_info(p) for p in sorted((ROOT / "scripts").glob("*.py")) if p.name not in ("__init__.py", "make_manifest.py")]
    res = ROOT / "paper_results"
    files = sorted(str(f.relative_to(res)) for f in res.rglob("*") if f.is_file()) if res.exists() else []
    unmatched = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in PRODUCERS)]
    manifest = {"training_runs": RUNS, "scripts": scripts, "result_producers": PRODUCERS, "unmatched_result_files": unmatched,
                "notes": "Latent extraction is not bitwise reproducible (paper Sections 3.1 and 3.2), and kernels are non-deterministic, so a seed fixes data order and initialization but not the bits of a trained checkpoint."}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    L = ["# Manifest: configs, seeds and the producer of every result", "", "Generated by `scripts/make_manifest.py`; `tests/test_manifest.py` checks that it is complete.", "", manifest["notes"], "", "## Training runs", "",
         "| run | paper | seed | settings |", "|---|---|---|---|"]
    for r in RUNS:
        cfg = "; ".join(f"{k}: {v}" for k, v in r["config"].items() if k not in ("architecture", "latents", "optimizer", "script", "milestone_checkpoints", "micro_batch"))
        L.append(f"| {r['run']} | {r['paper']} | {r['seed']} | {cfg} |")
    L += ["", f"Shared by the 9M-parameter runs: {COMMON['architecture']}; {COMMON['latents']}; {COMMON['optimizer']}; {COMMON['t_clip']}; micro-batch {COMMON['micro_batch']}; milestone checkpoints at {COMMON['milestone_checkpoints']}; trainer `{COMMON['script']}`.", "",
          "## Scripts", "", "| script | paper artifact | random seeds in the code |", "|---|---|---|"]
    for s in scripts: L.append(f"| `{s['script']}` | {s['artifact']} | {', '.join(s['seeds']) or 'none'} |")
    L += ["", "## Results and their producers", "", "| output pattern under paper_results/ | produced by |", "|---|---|"]
    for pat, prod in PRODUCERS.items(): L.append(f"| `{pat}` | {prod} |")
    if unmatched: L += ["", "Unmatched files: " + ", ".join(unmatched)]
    (ROOT / "MANIFEST.md").write_text("\n".join(L) + "\n")
    print(f"{len(RUNS)} runs, {len(scripts)} scripts, {len(files)} result files, {len(unmatched)} unmatched")


if __name__ == "__main__":
    main()
