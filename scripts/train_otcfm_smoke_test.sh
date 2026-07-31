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
#
# GPU sizing note (2026-07): an earlier attempt on eng-research-gpu's A10
# (22GB) hit a CUDA OOM at `pair[order]` — the reordering copies made by OT
# coupling (z1[order], pair[order], padding_mask[order]) are transient
# duplicates, and `pair` alone can be up to [16, 768, 768, 256] float32
# (~39GB) if a batch happens to include the largest-crop protein (1b25_D) —
# doubling transiently during the reorder. Matching the smoke test to the
# SAME GPU class the real retrain uses (H200, 80-141GB) rather than a
# smaller/faster-to-schedule GPU, so this is an actually representative
# test of the real training codepath at real scale.
#SBATCH --job-name=flux_otcfm_smoke
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
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
