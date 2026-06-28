#!/bin/bash
#SBATCH --job-name=flux_swarm
#SBATCH --output=logs/swarm_%A_%a.out
#SBATCH --error=logs/swarm_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=48G
#SBATCH --array=0-56   # 19 proteins × 3 replicas (1bcp_F unavailable)

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=/scratch/arjunc4/fluxion_data/latents

# Build flat list of all (protein, replica) pairs
# Each entry: "protein_id replica_num"
PAIRS=()
for protein_dir in "$ATLAS_RAW"/*/; do
    protein_id=$(basename "$protein_dir")
    for r in 1 2 3; do
        xtc="$protein_dir/${protein_id}_prod_R${r}_fit.xtc"
        if [ -f "$xtc" ]; then
            PAIRS+=("${protein_id} ${r}")
        fi
    done
done

# Select this worker's assignment
PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
PROTEIN_ID=$(echo "$PAIR" | awk '{print $1}')
REPLICA=$(echo "$PAIR"   | awk '{print $2}')

PDB="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}.pdb"
XTC="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}_prod_R${REPLICA}_fit.xtc"

echo "Worker $SLURM_ARRAY_TASK_ID: $PROTEIN_ID replica $REPLICA"
echo "  PDB: $PDB"
echo "  XTC: $XTC"

python data_pipeline/extract_latents.py \
    --pdb          "$PDB"                     \
    --traj         "$XTC"                     \
    --out-dir      "$OUT_DIR/$PROTEIN_ID"     \
    --shard-prefix "${PROTEIN_ID}_R${REPLICA}" \
    --shard-size   500                        \
    --skip-pair
