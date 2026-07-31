#!/bin/bash
# Cheap smoke test for the OT-coupling training code path (training/
# ot_coupling.py + its wiring into train_static.py) BEFORE committing to
# the multi-day retrain (scripts/train_static_otcfm.sh). Unit-tested
# thoroughly against synthetic data already — this just confirms it runs
# correctly on real GPU + real multi-protein batches (real mixed padding,
# real batch size/token counts) with no shape/dtype/perf surprises, same
# discipline used for the original static-ensemble pivot's diagnostic run.
#
# Throwaway checkpoint/log paths — not meant to be resumed into the real
# retrain.
#SBATCH --job-name=flux_otcfm_smoke
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/otcfm_smoke_%j.out
#SBATCH --error=logs/otcfm_smoke_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29501

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
    --epochs                1000 \
    --log-interval          10 \
    --max-steps              300 \
    --ot-coupling \
    --checkpoint-dir-name   checkpoints_static_otcfm_smoketest \
    --loss-log-name         loss_log_static_otcfm_smoketest.csv
