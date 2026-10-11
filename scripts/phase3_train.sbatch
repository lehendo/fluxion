#!/bin/bash
# TMLR Phase 3a: one arm per job. Usage:
#   ARM=uniform T_DIST=uniform STEPS=N sbatch --export=ALL scripts/phase3_train.sbatch
#   ARM=beta    T_DIST=beta BETA_A=0.25 BETA_B=0.25 STEPS=N sbatch --export=ALL scripts/phase3_train.sbatch
# On IllinoisComputes-GPU (3-day cap) override: sbatch --partition=IllinoisComputes-GPU --gres=gpu:1 --time=3-00:00:00 with TIME_LIMIT_MIN=4200.
# EXTRA_ARGS: extra trainer flags (arm 3c: --lr 1.414e-4 --warmup-steps 5600 --ema-decay 0.99994 --ckpt-interval 2000 --monitor-interval 1600 --log-interval 160).
# RESUBMIT_OPTS: the same command-line overrides used for the first submission (partition/gres/constraint/time/job-name), re-applied when the job requeues itself.
# MICRO x ACCUM = effective batch 256 (default 8 x 32). Everything except --t-dist/--beta-* is identical between arms. Requeues itself (exit 3 = wall-clock checkpoint) until STEPS is reached.
#SBATCH --job-name=p3_train
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/p3_train_%x_%j.out
#SBATCH --error=logs/p3_train_%x_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
export MALLOC_ARENA_MAX=2
OUT=${OUT:-/scratch/arjunc4/fluxion_data/phase3/${ARM}}
python training/train_t3.py --out-dir "$OUT" --t-dist "$T_DIST" --beta-a "${BETA_A:-0.25}" --beta-b "${BETA_B:-0.25}" \
    --max-steps "$STEPS" --batch-size "${MICRO:-8}" --accum "${ACCUM:-32}" --time-limit-min "${TIME_LIMIT_MIN:-225}" --num-workers 6 ${EXTRA_ARGS}
rc=$?
if [ $rc -eq 4 ]; then
    echo "[sbatch] DIVERGENCE GUARD HALT (sentinel HALTED_DIVERGENCE written); not requeueing"; exit 4
fi
if [ $rc -eq 3 ]; then
    echo "[sbatch] wall-clock checkpoint, resubmitting with: ${RESUBMIT_OPTS}"; eval sbatch --export=ALL ${RESUBMIT_OPTS} "$0"
fi
exit $rc
