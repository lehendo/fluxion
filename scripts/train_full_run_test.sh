#!/bin/bash
# Short combined test before trusting scripts/train_full_run.sh with a real
# 72h commitment. Verifies two things on IllinoisComputes-GPU:
#   1. Multi-GPU DDP starts cleanly on full (non-MIG) GPUs -- no "Duplicate
#      GPU detected" NCCL error like the ic-express MIG node hit.
#   2. The SIGUSR1 signal-forwarding fix (--signal=B:... + pkill trap)
#      actually reaches training/train.py and triggers a clean checkpoint
#      save + exit -- NOT sent automatically by SLURM here (this script has
#      no --time-triggered @N so it won't self-fire); instead, manually
#      send it once the job is confirmed running:
#        scancel --signal=SIGUSR1 <jobid>
#      then check for a saved checkpoint and a clean (not killed) exit.
#
# Uses the isolated --scratch-dir (fluxion_data_diag/, latents/ symlinked)
# so this doesn't touch any other run's checkpoint state. No
# --checkpoint-interval set deliberately, so ANY checkpoint that appears
# can only have come from the signal handler, not periodic saving --
# a clean test of the signal path specifically.
#SBATCH --job-name=flux_train_full_test
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --signal=B:SIGUSR1@1200
#SBATCH --output=logs/train_full_test_%j.out
#SBATCH --error=logs/train_full_test_%j.err

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
SCRATCH_DIR=/scratch/arjunc4/fluxion_data_diag
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
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 &

wait $!
