#!/bin/bash
# One-off diagnostic: does a larger batch size change the loss trajectory
# over the same number of steps, compared to 10151511 (the ongoing
# --batch-size 4 run)? 10151511's raw training loss stayed flat around
# ~0.0036-0.0041 for tens of thousands of steps, and its step-5000
# checkpoint tied the trivial baseline on held-out data -- but gradient
# norms throughout were consistently TINY (~0.0005-0.003) and never grew
# or shrank, which is at least as consistent with "LR mismatched to this
# loss landscape" or "just needs far more steps" as with "batch-4 is too
# noisy." Rather than commit another full 4h run to the batch-size
# hypothesis blind, this is a short (--max-steps 3000, ~15-25min at
# observed pace), --batch-size 8 run to see whether a less noisy batch
# visibly changes the trajectory over an equivalent number of updates.
#
# --scratch-dir points at an ISOLATED directory (fluxion_data_diag/,
# latents/ symlinked to the real corrected data, no duplication) rather
# than the live /scratch/arjunc4/fluxion_data/ -- 10151511 is still
# actively running and periodically checkpointing to
# fluxion_data/checkpoints/checkpoint.pt; sharing that path would make
# this run silently RESUME from 10151511's in-progress state instead of
# starting fresh at step 0 (defeating the comparison), and risk both
# jobs writing the same checkpoint file concurrently.
#SBATCH --job-name=flux_train_bs8diag
#SBATCH --partition=ic-express
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_bs8diag_%j.out
#SBATCH --error=logs/train_bs8diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

PILOT_PROTEINS="16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 \
    --max-steps 3000
