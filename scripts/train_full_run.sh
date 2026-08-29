#!/bin/bash
# Full training run on IllinoisComputes-GPU (72h cap, full A100/H200 GPUs
# -- not MIG-sliced, so the MIG+multi-process-DDP "Duplicate GPU detected"
# bug from ic-express should not apply here; multi-GPU DDP restored,
# --nproc_per_node=$NUM_GPUS instead of the single-GPU workaround used on
# ic-express).
#
# SIGNAL HANDLING FIX (2026-08-27): the earlier #SBATCH --signal=SIGUSR1@120
# delivered the signal to the srun/torchrun launcher, not the
# training/train.py worker process(es) torchrun spawns underneath it --
# Python's default (terminate) SIGUSR1 behavior killed the launcher before
# train.py's own registered handler ever ran, losing all progress on every
# interruption (confirmed repeatedly: 10142030 ran a full 4h with zero
# checkpoints saved despite the signal firing on schedule). Real fix:
# --signal=B:SIGUSR1@120 delivers the signal to THIS BATCH SCRIPT instead
# (the "B" prefix), which is guaranteed to receive it (SLURM tracks the
# batch script directly), and the trap below explicitly forwards it via
# pkill to every training/train.py process by command-line match -- this
# correctly reaches ALL DDP ranks (not just rank 0), which matters because
# train.py's preemption handler has every rank participate in a
# dist.barrier() before exiting; if only some ranks received the signal,
# that barrier would hang forever. pkill only matches this user's own
# processes, safe on a shared node.
#SBATCH --job-name=flux_train_full
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=71:30:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# 2026-08-28 fix, take 2: excluding the torchrun launcher by cmdline match
# (job 10206925 -> 10213358) still wasn't surgical enough. sacct showed the
# actual torchrun step failed (exit 1), and the log showed 3x "[Rank 1]"
# preemption prints for what should be a single delivery -- pkill -f
# "training/train.py" also matches each rank's num_workers=4 DataLoader
# worker subprocesses (forked from the rank process AFTER the handler was
# installed, so they inherit it and print too), and signaling a worker
# while it's blocked in a low-level IPC read to its parent can corrupt that
# pipe, crashing the rank before it finishes checkpointing. Real fix: only
# signal torchrun's DIRECT children (the exact rank processes) by PID via
# `pkill -P`, never touching the DataLoader worker grandchildren or the
# launcher itself.
forward_signal() {
    echo "[wrapper] Caught SIGUSR1 at $(date), locating torchrun launcher to signal its direct rank children only..."
    TORCHRUN_PID=$(pgrep -f "torchrun.*--standalone" | head -1)
    if [ -n "$TORCHRUN_PID" ]; then
        echo "[wrapper] torchrun launcher PID=$TORCHRUN_PID, signaling its direct children:"
        pgrep -P "$TORCHRUN_PID"
        pkill -SIGUSR1 -P "$TORCHRUN_PID"
    else
        echo "[wrapper] WARNING: could not find torchrun launcher PID, falling back to broad pkill"
        pkill -SIGUSR1 -f "training/train.py"
    fi
}
trap forward_signal SIGUSR1

PILOT_PROTEINS="16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --standalone \
    --nproc_per_node=4 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 200 \
    --log-interval 50 \
    --checkpoint-interval 2000 &

wait $!
