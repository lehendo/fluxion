#!/bin/bash
#SBATCH --job-name=flux_eval
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

python scripts/evaluate_checkpoint.py \
    --checkpoint  /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --proteins    "$PILOT_PROTEINS" \
    --n-batches   200 \
    --batch-size  16 \
    --num-workers 4 \
    --device      cpu
