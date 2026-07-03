#!/bin/bash
#SBATCH --job-name=flux_posmean
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/posmean_%j.out
#SBATCH --error=logs/posmean_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

python scripts/compute_position_mean_baseline.py \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --proteins    "$PILOT_PROTEINS" \
    --batch-size  256 \
    --num-workers 4
