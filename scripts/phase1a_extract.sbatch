#!/bin/bash
# Phase 1A row-cache extraction, one array task per protein (19 total).
# Override for a smoke test:  sbatch --array=0 --export=ALL,SMOKE=1 scripts/phase1a_extract.sbatch
#SBATCH --job-name=p1a_extract
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-18
#SBATCH --output=logs/p1a_extract_%A_%a.out
#SBATCH --error=logs/p1a_extract_%A_%a.err

mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
ulimit -n 65536

PROTEINS=(16pk_A 1a62_A 1ab1_A 1af7_A 1ah7_A 1ail_A 1aol_A 1b0n_A 1b25_D 1b2s_E 1b2s_F 1b8d_K 1bgf_A 1bkp_A 1bq8_A 1btk_A 1bx7_A 1bxy_A 1bxy_B)
P=${PROTEINS[$SLURM_ARRAY_TASK_ID]}
OUT=/scratch/arjunc4/fluxion_data/phase1_cache
EXTRA=""
if [ -n "$SMOKE" ]; then OUT=/scratch/arjunc4/fluxion_data/phase1_cache_smoke; EXTRA="--lags 1 100 --frames-per-lag 48"; P=1btk_A; fi

python scripts/phase1a_extract.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --protein "$P" --out-dir "$OUT" \
    --dit-checkpoint /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt \
    --num-workers 8 $EXTRA
