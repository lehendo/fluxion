#!/bin/bash
# Same combined verification test as train_full_run_test.sh (multi-GPU DDP
# clean-start + SIGUSR1 signal-forwarding fix), rerouted to `secondary`
# instead of IllinoisComputes-GPU.
#
# 2026-08-27: IllinoisComputes-GPU had zero nodes with >=4 free full GPUs
# (job 10194965 sat PENDING/Priority for 12+h -- checked via `scontrol show
# node`: every node was either fully allocated or short by exactly the GPUs
# we needed). secondary has a hard 4h cap, but this test only needs minutes,
# so that's not a real constraint. Checked node-by-node for a node with free
# GPUs AND free mem (the obvious H100:8 nodes, ccc0423/ccc0424, looked idle
# in sinfo's gres column but turned out to be memory-starved by a co-located
# CPU-only job -- ~1TB mem allocated, only ~8GB free). Node target has moved
# around as free nodes get claimed by other jobs between checks (first
# ccc0333/A40:2 free -- ran as job 10206925, then got claimed before the
# rerun; now ccc0478/A30:2, fully idle, 96 free CPUs, 257G free mem) --
# always re-check `scontrol show node <n>` for actual free GPU/mem
# immediately before submitting. --nproc_per_node=2 (not 4): 2 real,
# distinct, non-MIG GPUs is sufficient to answer the actual question (does
# DDP avoid the "Duplicate GPU detected" collision outside of MIG slices)
# -- the bug was about MIG slice aliasing, not GPU count.
#SBATCH --job-name=flux_train_full_test_sec
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=ccc0478
#SBATCH --gres=gpu:A30:2
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
    echo "[wrapper] Caught SIGUSR1 at $(date), forwarding to training/train.py rank process(es), excluding the torchrun launcher..." | tee -a logs/train_full_test_sec_${SLURM_JOB_ID}.out
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
    --nproc_per_node=2 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 4 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 &

wait $!
