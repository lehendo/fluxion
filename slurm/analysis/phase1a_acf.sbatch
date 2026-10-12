#!/bin/bash
#SBATCH --job-name=p1a_acf
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1a_acf_%j.out
#SBATCH --error=logs/p1a_acf_%j.err
mkdir -p logs
source ~/.bashrc
conda activate fluxion
cd /u/arjunc4/fluxion
python scripts/phase1a_acf.py --out /scratch/arjunc4/fluxion_data/phase1_results/acf.json
