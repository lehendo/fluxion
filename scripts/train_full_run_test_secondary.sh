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
# CPU-only job -- ~1TB mem allocated, only ~8GB free). ccc0333 (A40:3, 1
# already in use) has 2 genuinely free A40s plus ~225G free mem and 56 free
# CPUs -- used --nproc_per_node=2 instead of 4 here. 2 real, distinct,
# non-MIG GPUs is sufficient to answer the actual question (does DDP avoid
# the "Duplicate GPU detected" collision outside of MIG slices) -- the bug
# was about MIG slice aliasing, not GPU count.
#SBATCH --job-name=flux_train_full_test_sec
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=ccc0333
#SBATCH --gres=gpu:A40:2
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

forward_signal() {
    echo "[wrapper] Caught SIGUSR1 at $(date), forwarding to training/train.py worker(s)..." | tee -a logs/train_full_test_sec_${SLURM_JOB_ID}.out
    pkill -SIGUSR1 -f "training/train.py"
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
