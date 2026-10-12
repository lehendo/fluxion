#!/bin/bash
# Recompute norm_stats.pt against the corrected latents/ (post coordinate-
# injection fix, post swap 2026-08-24). The prior norm_stats.pt only exists
# under the old, now-renamed latents_prefix_broken_20260813/ -- every
# diagnostic script silently proceeds WITHOUT normalization if this file is
# missing, which would make the re-run not comparable to the original pilot
# numbers (which were computed WITH normalization). Covers all 19 proteins
# (default: all proteins found under --latents-dir).
#SBATCH --job-name=flux_normstats
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/normstats_%j.out
#SBATCH --error=logs/normstats_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

python data_pipeline/compute_norm_stats.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents
