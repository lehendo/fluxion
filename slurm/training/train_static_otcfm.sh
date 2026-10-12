#!/bin/bash
# Retrain of the static-ensemble pivot WITH minibatch OT coupling enabled
# (training/ot_coupling.py, --ot-coupling default on) — the training-side
# fix for the diversity-compression / mode-mismatch signature diagnosed
# 2026-07 across two proteins (1bxy_A: uniform variance underestimation
# despite strong RMSF shape correlation; 1b25_D: generated samples landing
# outside the real ensemble's occupied conformational modes). Two cheap
# sampling-time interventions (RK4 vs Euler integration, noise-scale
# rescaling) were both tested and ruled out first, confirming the issue is
# training-time, not sampling-time.
#
# Uses --checkpoint-dir-name and --loss-log-name to write to FRESH paths,
# separate from the original (pre-OT-coupling) run's checkpoints_static/ —
# resuming from that old checkpoint would start from weights trained under
# the arbitrary index-pairing scheme, which is not a clean comparison.
#
# Same 20-epoch budget as the original run for direct comparability.
# --time set to the partition's max (3 days) per standing project policy —
# costs nothing if the job finishes sooner, and this WILL need multiple
# resubmissions like the original run did (each ~8.7h/epoch x 20 epochs
# does not fit in one 3-day allocation).
#SBATCH --job-name=flux_train_otcfm
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120
#SBATCH --output=logs/train_otcfm_%j.out
#SBATCH --error=logs/train_otcfm_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    training/train_static.py \
    --scratch-dir          /scratch/arjunc4/fluxion_data \
    --batch-size            16 \
    --lr                    1e-4 \
    --epochs                20 \
    --log-interval          50 \
    --checkpoint-interval   500 \
    --ot-coupling \
    --checkpoint-dir-name   checkpoints_static_otcfm \
    --loss-log-name         loss_log_static_otcfm.csv
