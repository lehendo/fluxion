#!/bin/bash
# Full-scale run of the static-ensemble pivot (noise -> equilibrium frame
# OT-CFM, see training/train_static.py). Validated first via the 5-minute
# smoke test (slurm/training/train_static_diagnostic.sh, 2026-07-14): 170+ real
# steps on real GPU/data, finite loss, no crash.
#
# Uses ALL 19 extracted proteins (not just the 14-protein fast-extraction
# pilot subset — that restriction was about extraction wall-clock time,
# which no longer matters since extraction is already complete for all 19).
#
# --checkpoint-interval is the PRIMARY safety net here, not the SIGUSR1
# preemption signal: srun + torchrun have a known race (see 2026-07-14
# diagnostic run) where torchrun's own process can die from the broadcast
# SIGUSR1 before our worker's graceful-shutdown handler finishes, causing
# srun to tear down the step early. Periodic checkpointing bounds any lost
# progress to --checkpoint-interval steps regardless of whether the signal
# path works.
#SBATCH --job-name=flux_train_static
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
#SBATCH --output=logs/train_static_%j.out
#SBATCH --error=logs/train_static_%j.err

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
    --checkpoint-interval   500
