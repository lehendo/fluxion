#!/bin/bash
# Same combined verification test as train_full_run_test.sh (multi-GPU DDP
# clean-start + SIGUSR1 signal-forwarding fix), rerouted to `secondary`
# instead of IllinoisComputes-GPU.
#
# 2026-08-27: IllinoisComputes-GPU had zero nodes with >=4 free full GPUs
# (job 10194965 sat PENDING/Priority for 12+h). secondary has a hard 4h
# cap, but this test only needs minutes, so that's not a real constraint.
#
# 2026-08-29: had been manually picking a specific free node via `scontrol
# show node` + --nodelist, but secondary's free-GPU nodes are churning too
# fast to hand-pick this way (checked ccc0420 at 3 free H100s, resubmit
# ~30s later and it was down to 2 -- another job grabbed one mid-check).
# Stopped racing other users' jobs via SSH polling: dropped --nodelist and
# just request --gres=gpu:H100:3 on the whole partition, letting Slurm's
# own scheduler atomically place it on whichever of secondary's several
# H100 nodes currently qualifies. --nproc_per_node=3 (not the real run's
# 4): closest to 4 that's realistically gettable without risking another
# multi-hour IllinoisComputes-GPU-style wait -- prior 2-way tests already
# confirmed non-MIG GPUs avoid the "Duplicate GPU detected" bug (which was
# about MIG slice aliasing, not GPU count).
#
# 2026-08-29, second thought: tried 3-way (bumping --mem to 192G after an
# OOM at 128G -- job 10220794, 5 oom_kill events), but that request then
# sat PENDING because nothing on secondary currently had 3 free H100s +
# 192G free mem at once -- a much harder bin to fill than 2-way. Dropped
# back to 2 GPUs / 96G: DDP-cleanliness on non-MIG hardware is already
# answered by three separate 2-way runs, so holding out for 3-way bought
# no new information, just a longer queue wait for a question (SIGUSR1
# graceful-exit, where rank 0 currently goes silent) that's orthogonal to
# GPU count anyway.
#
# 2026-08-29, third thought: landed on ccc0423 again (job 10222972) --
# ran cleanly for ~5 min and 250+ steps, then a real oom_kill hit rank 1.
# ccc0423/ccc0424 have now shown memory contention twice from apparently
# OTHER tenants' jobs sharing the node (fluctuating over time, not our own
# config -- job 10214022 ran fine on this same node with the same 96G
# request earlier). Excluding both explicitly so Slurm places us on a
# less-contended secondary H100 node instead of rolling the dice again.
#SBATCH --exclude=ccc0423,ccc0424
#SBATCH --job-name=flux_train_full_test_sec
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --signal=B:SIGUSR1@1200
#SBATCH --output=logs/train_full_test_sec_%j.out
#SBATCH --error=logs/train_full_test_sec_%j.err

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
    --nproc_per_node=2 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 &

wait $!
