#!/bin/bash
#SBATCH --job-name=flux_eval
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

# 2026-08-25: held out (never trained on, in any form) -- same split used
# by today's nonlinear probe (compute_nonlinear_probe.sh) so this checkpoint's
# real generalization is directly comparable to that probe's 91.25%
# reduction / MSE=0.001701 result and the linear baseline (MSE=0.001638),
# not conflated with train-set fit.
VAL_PROTEINS="1bxy_A,1bxy_B,1btk_A"

python scripts/evaluate_checkpoint.py \
    --checkpoint  /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt \
    --latents-dir /scratch/arjunc4/fluxion_data/latents \
    --proteins    "$VAL_PROTEINS" \
    --n-batches   100 \
    --batch-size  4 \
    --num-workers 2 \
    --device      cpu
