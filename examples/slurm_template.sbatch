#!/bin/bash
# Template for a multi-GPU training job that checkpoints when it is preempted or reaches its wall limit.
# Replace every <placeholder>. Explanations of each line are in guides/slurm_preemption.md and guides/ddp_four_gpu.md.
#SBATCH --job-name=fluxion_train
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=<wall limit, minus a few minutes>
#SBATCH --requeue
# "B:" delivers the signal to THIS batch script (not to srun/torchrun); 120 s is the notice before the wall limit.
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
export FLUXION_DATA=<path to the data root>
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1
ulimit -n 65536          # many DataLoader workers need many descriptors

# Forward SIGUSR1 to the ranks. Each rank writes its own PID at start-up, so no process-tree guessing is needed.
forward_signal() {
    echo "[wrapper] SIGUSR1 at $(date); signalling ranks"
    PID_DIR="$FLUXION_DATA/.rank_pids/$SLURM_JOB_ID"
    for f in "$PID_DIR"/rank_*.pid; do
        [ -f "$f" ] || continue
        kill -SIGUSR1 "$(cat "$f")" 2>/dev/null
    done
}
trap forward_signal SIGUSR1

srun torchrun --standalone --nproc_per_node=4 -m fluxion.training.train \
    --scratch-dir "$FLUXION_DATA" \
    --proteins "<comma-separated protein ids>" \
    --batch-size 4 --lr 1e-4 --epochs 200 --log-interval 50 --checkpoint-interval 2000 &
CHILD_PID=$!

# `wait` returns at once when a trapped signal interrupts it; keep the batch script (and therefore the job step) alive
# until the training process has really exited, or SLURM kills it in the middle of the checkpoint.
wait "$CHILD_PID"
RC=$?
while [ "$RC" -ge 128 ] && kill -0 "$CHILD_PID" 2>/dev/null; do
    wait "$CHILD_PID"
    RC=$?
done
exit "$RC"
