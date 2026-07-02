#!/bin/bash
# Short diagnostic run: validates the AdaLN-zero weight-decay fix and
# grad-norm instrumentation on a few thousand steps (~15-30 min) before
# committing to another multi-day full run. Checks:
#   - total_grad_norm vs gate_grad_norm in loss_log.csv: is the global clip
#     (max_norm=1.0) squashing gate gradients relative to the "always-on"
#     parameters?
#   - does loss show ANY measurable dip below the trivial baseline (~0.00083)
#     within a few thousand steps now that AdaLN-zero weights aren't being
#     pulled back toward zero by weight decay every step?
#
# Requires a fresh start — delete the stalled checkpoint/log first:
#   rm /scratch/arjunc4/fluxion_data/checkpoints/checkpoint.pt
#   rm /scratch/arjunc4/fluxion_data/loss_log.csv
#
#SBATCH --job-name=flux_diag
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

PILOT_PROTEINS="1a62_A,1ab1_A,1ah7_A,1ail_A,1b0n_A,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/train.py \
    --scratch-dir          /scratch/arjunc4/fluxion_data \
    --proteins             "$PILOT_PROTEINS" \
    --batch-size            16 \
    --lr                    1e-4 \
    --epochs                50 \
    --log-interval           25 \
    --max-steps            5000
