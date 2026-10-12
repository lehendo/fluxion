# SLURM job scripts

All batch scripts for the UIUC Campus Cluster, grouped by purpose. Submit from the repository root (scripts `cd` to the cluster checkout themselves and write logs to `logs/` relative to where you submit):

```bash
sbatch slurm/training/train_full_run.sh
```

| Folder | Contents |
|---|---|
| `extraction/` | Chai-1 latent and pair extraction (`swarm_extract*.slurm`, `extract_pairs.slurm`, `extract_latents.sh`), normalization statistics, ATLAS sequence fetch |
| `training/` | DiT training runs (temporal, static-ensemble, smoke tests and diagnostics, the small-model arms in `phase3_train.sbatch` / `phase3_diag.sbatch`) |
| `evaluation/` | Checkpoint and ensemble evaluation, resampling, verification of extractions and the ATLAS re-fetch, held-out evaluations of the small-model arms |
| `analysis/` | Baseline and probe analyses, lag sweeps, extraction caches and the frame-prediction measurement pipeline (`phase1*`) |

Environment setup is not a SLURM job and lives in `scripts/setup_env.sh`.
