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

# 2026-08-28 fix: `pkill -SIGUSR1 -f "training/train.py"` was too broad --
# it also matches the torchrun LAUNCHER process itself (its own cmdline
# includes "training/train.py" as the script it's launching), which has no
# signal handler and dies instantly on SIGUSR1, triggering torchrun's own
# child-teardown before the rank processes can finish their graceful
# checkpoint+exit (confirmed via job 10206925: both ranks logged "preparing
# to checkpoint", but no checkpoint was saved and the job ended FAILED/138
# before "Graceful exit" ever printed). Fix: only signal PIDs whose cmdline
# does NOT also contain "torchrun" -- this reaches the two rank processes
# and their forked DataLoader workers (which inherit the real handler via
# fork) while leaving the launcher alone.
forward_signal() {
    echo "[wrapper] Caught SIGUSR1 at $(date), forwarding to training/train.py rank process(es), excluding the torchrun launcher..." | tee -a logs/train_full_${SLURM_JOB_ID}.out
    for pid in $(pgrep -f "training/train.py"); do
        if ! grep -q "torchrun" "/proc/$pid/cmdline" 2>/dev/null; then
            kill -SIGUSR1 "$pid" 2>/dev/null
        fi
    done
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
