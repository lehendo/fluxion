#!/bin/bash
# Sweeps the position-mean baseline (scripts/compute_position_mean_baseline.py)
# across multiple frame lags in parallel SLURM array tasks, to find where real
# conformational-dynamics signal starts to appear above thermal noise.
#
# At native ATLAS stride (lag=1, 10ps), the static per-position mean explained
# 0.00% of frame-to-frame displacement across all 14 pilot proteins (2026-07-03
# pilot finding) — consistent with instantaneous displacement being dominated
# by fast thermal/vibrational noise at that timescale, not genuine
# conformational drift. This sweep tests longer lags (up to 50ns, half the
# 100ns trajectory length) to find where that stops being true.
#
# No data re-extraction needed — lag only changes which already-extracted
# frames get paired as (z0, z1); see data_pipeline/dataset.py's `lag` param.
#
# After all array tasks finish, run:
#   python scripts/summarize_lag_sweep.py --logs-dir logs --job-id <jobid>
#
#SBATCH --job-name=flux_lagsweep
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/lagsweep_%A_%a.out
#SBATCH --error=logs/lagsweep_%A_%a.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

# frames: 1=10ps, 10=100ps, 50=500ps, 100=1ns, 500=5ns, 1000=10ns, 2500=25ns, 5000=50ns
LAGS=(1 10 50 100 500 1000 2500 5000)
LAG=${LAGS[$SLURM_ARRAY_TASK_ID]}

echo "[sweep] Array task $SLURM_ARRAY_TASK_ID: lag=$LAG frames ($((LAG * 10))ps)"

python scripts/compute_position_mean_baseline.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --proteins    "$PILOT_PROTEINS" \
    --batch-size  256 \
    --num-workers 6 \
    --lag         "$LAG"
