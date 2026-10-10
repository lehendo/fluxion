#!/bin/bash
# TMLR Phase 3a: one arm per job. Usage:
#   ARM=uniform T_DIST=uniform STEPS=N sbatch --export=ALL scripts/phase3_train.sbatch
#   ARM=beta    T_DIST=beta BETA_A=0.5 BETA_B=0.5 STEPS=N sbatch --export=ALL scripts/phase3_train.sbatch
# Everything except --t-dist/--beta-* is identical between arms. Requeues itself (exit 3 = wall-clock checkpoint) until STEPS is reached.
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
python training/train_t3.py --out-dir "$OUT" --t-dist "$T_DIST" --beta-a "${BETA_A:-0.5}" --beta-b "${BETA_B:-0.5}" \
    --max-steps "$STEPS" --time-limit-min 225 --num-workers 6
rc=$?
if [ $rc -eq 3 ]; then
    echo "[sbatch] wall-clock checkpoint, resubmitting"; sbatch --export=ALL "$0"
fi
exit $rc
