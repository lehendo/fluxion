#!/bin/bash
#SBATCH --job-name=flux_restoring
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/restoring_%j.out
#SBATCH --error=logs/restoring_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

python scripts/compute_restoring_force_baseline.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --proteins    "$PILOT_PROTEINS" \
    --batch-size  256 \
    --num-workers 6 \
    --lag         1
