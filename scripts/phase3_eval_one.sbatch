#!/bin/bash
# One milestone, both arms, on the secondary partition (4 h cap). Usage: M=100 sbatch --export=ALL scripts/phase3_eval_one.sbatch
#SBATCH --job-name=p3_eval1
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/p3_eval1_%j.out
#SBATCH --error=logs/p3_eval1_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
P3=/scratch/arjunc4/fluxion_data/phase3
mkdir -p $P3/eval_sec
python scripts/phase3_eval.py --arm uniform $P3/uniform/milestone_$M.pt --arm beta $P3/beta/milestone_$M.pt --out $P3/eval_sec/${OUT_NAME:-eval_$M}.json ${EVAL_ARGS}
