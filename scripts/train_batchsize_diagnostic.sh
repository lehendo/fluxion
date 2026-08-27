#!/bin/bash
# One-off diagnostic, now serving two purposes:
#
# 1. (2026-08-25) Does a larger batch size change the loss trajectory over
#    the same number of steps, compared to 10151511 (the batch-4 full run)?
#    First attempt (10153220) got to step ~2160 (well past warmup) before
#    an OOM cut it short. Post-warmup mean loss (~0.00387) was statistically
#    indistinguishable from the batch-4 run's post-warmup mean (~0.00399-
#    0.00407) -- no evidence larger batch alone fixes the flat-loss plateau,
#    though the sample was small.
#
# 2. (2026-08-26) Test a fix for the OOM itself. 10153220's crash was a
#    Linux cgroup oom_kill (host RAM past --mem, not CUDA OOM), arriving
#    gradually around step ~2000+ rather than immediately -- the signature
#    of glibc allocator fragmentation in the 4 DataLoader worker processes,
#    not a real Python-level leak. The pair tensor is padded to L_max PER
#    BATCH (256/384/512/768 depending on which proteins the random sampler
#    draws -- only 1b25_D uses 768), so workers repeatedly allocate/free
#    differently-sized large tensors across thousands of iterations, a
#    well-documented trigger for this exact glibc fragmentation pattern.
#    Standard fix: cap glibc's allocation arenas via MALLOC_ARENA_MAX,
#    applied here before deciding whether a more invasive code-level fix
#    (e.g. always padding to a fixed L_max) is also needed. --max-steps
#    raised to 5000 (comfortably past the ~2160 step failure point) to
#    give real confidence the fix holds, not just delays the crash.
#
# --scratch-dir points at an ISOLATED directory (fluxion_data_diag/,
# latents/ symlinked to the real corrected data, no duplication) rather
# than the live /scratch/arjunc4/fluxion_data/ -- keeps this fully
# independent of any other training run's checkpoint state.
#SBATCH --job-name=flux_train_bs8diag
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_bs8diag_%j.out
#SBATCH --error=logs/train_bs8diag_%j.err

mkdir -p logs

source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion

export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# Standard fix for glibc allocator fragmentation in long-running
# multi-process workers handling variable-sized allocations (a widely
# documented PyTorch DataLoader issue, not specific to this codebase).
# Caps each process to a single memory arena instead of glibc's default
# per-thread arena pool, which is what lets fragmented-but-freed memory
# get reused instead of accumulating as new OS-level allocations.
export MALLOC_ARENA_MAX=1

PILOT_PROTEINS="16pk_A,1a62_A,1ab1_A,1af7_A,1ah7_A,1ail_A,1aol_A,1b0n_A,1b25_D,1b2s_E,1b2s_F,1b8d_K,1bgf_A,1bkp_A,1bq8_A,1btk_A,1bx7_A,1bxy_A,1bxy_B"

srun torchrun \
    --standalone \
    --nproc_per_node=1 \
    training/train.py \
    --scratch-dir /scratch/arjunc4/fluxion_data_diag \
    --proteins     "$PILOT_PROTEINS" \
    --batch-size 8 \
    --lr 1e-4 \
    --epochs 1 \
    --log-interval 10 \
    --max-steps 5000
