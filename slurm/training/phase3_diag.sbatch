#!/bin/bash
# TMLR Phase 3 stability diagnostic (pre-registered in experiments/EXPERIMENT_LOG.md): 4 conditions x 2 seeds, 600 optimizer steps each, with the SAME schedule as the 3a arms
# (warmup 350, cosine over 6400) so peak LR falls inside the window; guard in RECORD mode (observe the excursion, do not halt).
#   idx = 2*cond + seed;  cond 0=A uniform (peak LR)  1=B beta-clamp (peak LR)  2=C beta-truncate (peak LR)  3=D beta-clamp (half peak LR)
#SBATCH --job-name=p3_diag
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/p3_diag_%A_%a.out
#SBATCH --error=logs/p3_diag_%A_%a.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
export MALLOC_ARENA_MAX=2
C=$((SLURM_ARRAY_TASK_ID / 2)); S=$((SLURM_ARRAY_TASK_ID % 2))
NAMES=(A B C D)
PEAK=5.657e-4; HALF=2.828e-4
case $C in
  0) ARGS="--t-dist uniform --lr $PEAK" ;;
  1) ARGS="--t-dist beta --t-mode clamp --lr $PEAK" ;;
  2) ARGS="--t-dist beta --t-mode truncate --lr $PEAK" ;;
  3) ARGS="--t-dist beta --t-mode clamp --lr $HALF" ;;
esac
OUT=/scratch/arjunc4/fluxion_data/phase3/diag/${NAMES[$C]}_s${S}
python training/train_t3.py --out-dir "$OUT" --max-steps 6400 --stop-after 600 --seed $S --batch-size 8 --accum 32 \
    --beta-a 0.25 --beta-b 0.25 --guard-mode record --guard-ratio 2.0 --monitor-interval 100 --ckpt-interval 100000 --time-limit-min 4200 --num-workers 6 $ARGS
