#!/bin/bash
# One-off variant of swarm_extract.slurm for large-protein OOM retries.
# 2026-08-14: 1b25_D (619 residues -> Chai-1's 768-token crop bucket) OOM'd
# twice on an untyped --gres=gpu:1 draw that landed on a 14.56 GiB card
# (trunk attention alone needed 6.75 GiB more than was free). Adds
# --low-memory to the extraction call, which only changes device placement
# of intermediate tensors between pipeline stages (return_on_cpu) -- the
# trunk.forward() recycle loop itself takes no low_memory argument and runs
# identically either way, so output latents are numerically identical to a
# normal run, just slower due to extra CPU<->GPU transfers. Not merged into
# the main swarm_extract.slurm since only this protein's tasks need it.
#SBATCH --job-name=flux_swarm_fixed_lowmem
#SBATCH --output=logs/swarm_fixed_%A_%a.out
#SBATCH --error=logs/swarm_fixed_%A_%a.err
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=/scratch/arjunc4/fluxion_data/latents_fixed

# Build flat list of all (protein, replica) pairs -- identical to
# swarm_extract.slurm so array indices line up exactly.
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

PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
PROTEIN_ID=$(echo "$PAIR" | awk '{print $1}')
REPLICA=$(echo "$PAIR"   | awk '{print $2}')

PDB="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}.pdb"
XTC="$ATLAS_RAW/$PROTEIN_ID/${PROTEIN_ID}_prod_R${REPLICA}_fit.xtc"

echo "Worker $SLURM_ARRAY_TASK_ID: $PROTEIN_ID replica $REPLICA (low-memory mode)"
echo "  PDB: $PDB"
echo "  XTC: $XTC"

python data_pipeline/extract_latents.py \
    --pdb          "$PDB"                     \
    --traj         "$XTC"                     \
    --out-dir      "$OUT_DIR/$PROTEIN_ID"     \
    --shard-prefix "${PROTEIN_ID}_R${REPLICA}" \
    --shard-size   500                        \
    --skip-pair                                \
    --low-memory
