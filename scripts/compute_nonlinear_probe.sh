#!/bin/bash
# Held-out-PROTEIN split: 1bxy_A, 1bxy_B, 1btk_A are never seen in any form
# during training, not even different time windows of them — only genuine
# generalization to unseen proteins counts as evidence here. See
# compute_nonlinear_probe.py's docstring for why this matters.
#
#SBATCH --job-name=flux_nlprobe
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nlprobe_%j.out
#SBATCH --error=logs/nlprobe_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

TRAIN_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1bx7_A"
VAL_PROTEINS="1bxy_A,1bxy_B,1btk_A"

python scripts/compute_nonlinear_probe.py \
    --latents-dir     /scratch/arjunc4/fluxion_data/latents \
    --train-proteins  "$TRAIN_PROTEINS" \
    --val-proteins    "$VAL_PROTEINS" \
    --lag             1 \
    --velocity-lag    1 \
    --max-samples-per-protein 150000 \
    --hidden          256 \
    --lr              1e-3 \
    --batch-size      1024 \
    --max-epochs      30 \
    --patience        4 \
    --device          cpu \
    --num-workers     6
