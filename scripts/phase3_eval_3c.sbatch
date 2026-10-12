#!/bin/bash
# Evaluate arm 3c (bs16) alone at one milestone with the same pre-registered protocol (frames, seed, alpha fit are deterministic and match the uniform/beta evaluation). Usage: M=100 sbatch --export=ALL scripts/phase3_eval_3c.sbatch
#SBATCH --job-name=p3_3c_eval
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/p3_3c_eval_%j.out
#SBATCH --error=logs/p3_3c_eval_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
P3=/scratch/arjunc4/fluxion_data/phase3
mkdir -p $P3/eval_sec
python scripts/phase3_eval.py --arm bs16 $P3/bs16/milestone_$M.pt --out $P3/eval_sec/eval_3c_$M.json ${EVAL_ARGS}
