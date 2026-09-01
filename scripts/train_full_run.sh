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
# --exclude=ccc0390 (2026-08-31): job 10252106 landed there and failed in
# 24s -- all 4 ranks hit "device >= 0 && device < num_gpus INTERNAL ASSERT
# FAILED ... num_gpus=" (blank), meaning CUDA saw zero visible GPUs despite
# SLURM allocating 4. This node has shown State=drain* in every sinfo scan
# this session; it looks like it got un-drained without the underlying
# GPU/driver fault being fixed. Cluster infra issue, not our code.
#SBATCH --exclude=ccc0390
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=71:30:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120
# --mem bumped 256G -> 384G (2026-08-31): the verification tests showed
# memory scales faster than linear with rank count (96G sufficed for
# 1 rank, 2 ranks needed 160G, not the naively-scaled 128G -- each
# rank runs its own num_workers=4 DataLoader workers + pinned buffers).
# Extrapolating to 4 ranks suggests ~288G; 384G leaves real margin for
# a 72h unattended run, and IllinoisComputes-GPU nodes have 515G-1.5T
# total so this costs nothing in schedulability.
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# 2026-08-28 fix, take 3: two prior attempts at guessing torchrun's process
# tree from outside both failed -- broad `pkill -f "training/train.py"`
# (job 10206925/10213358) also hit each rank's num_workers=4 DataLoader
# worker subprocesses (which inherit the handler via fork and can crash
# when signaled mid-IPC-read); narrowing to torchrun's direct children via
# `pkill -P <torchrun_pid>` (job 10213525) hit a single intermediate
# supervisor process that was neither actual rank, killing the job before
# either rank's handler ran (confirmed: zero "[Rank X] preparing" prints
# that run). torchrun's internal tree shape isn't stable enough to guess
# externally. Real fix: training/train.py now self-reports each rank's own
# PID to $SCRATCH_DIR/.rank_pids/$SLURM_JOB_ID/rank_<N>.pid at startup --
# the wrapper just reads those files, no process-tree guessing at all.
SCRATCH_DIR=/scratch/arjunc4/fluxion_data
forward_signal() {
    echo "[wrapper] Caught SIGUSR1 at $(date), signaling rank processes via self-reported PID files..."
    PID_DIR="$SCRATCH_DIR/.rank_pids/$SLURM_JOB_ID"
    if [ -d "$PID_DIR" ]; then
        for f in "$PID_DIR"/rank_*.pid; do
            [ -f "$f" ] || continue
            pid=$(cat "$f")
            echo "[wrapper] signaling rank pid=$pid (from $f)"
            kill -SIGUSR1 "$pid" 2>/dev/null
        done
    else
        echo "[wrapper] WARNING: $PID_DIR not found yet (too early?), falling back to broad pkill"
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
CHILD_PID=$!

# 2026-08-30 fix: found the REAL root cause of every prior verification
# attempt's failure. `wait $!` returns immediately (exit status 128+signum)
# the moment a TRAPPED signal interrupts it -- it does not wait for the
# background job to actually finish. Since this batch script is the job's
# root process, the instant it falls through to its own end and exits,
# SLURM tears down the entire job step (including the still-running
# torchrun/rank processes), killing them mid-checkpoint regardless of how
# correctly train.py's own signal handling works. Fix: after a trapped
# signal interrupts wait, re-wait on the same PID -- wait on an already-
# finished process returns immediately with its real exit code, so this
# loop costs nothing once the child actually exits, but keeps the script
# (and therefore the job) alive until it does.
wait "$CHILD_PID"
RC=$?
while [ "$RC" -ge 128 ] && kill -0 "$CHILD_PID" 2>/dev/null; do
    wait "$CHILD_PID"
    RC=$?
done
exit "$RC"
