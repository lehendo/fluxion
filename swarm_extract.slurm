#!/bin/bash
# 2026-08 fix: re-extraction with data_pipeline/chai1_conditioning.py's
# coordinate-injection fix (see docs/code_review_2026-08.md). Two changes
# from the original version of this script:
#   1. --gres=gpu:1 (untyped) instead of a hardcoded gpu:H100:1 -- H100/H200
#      on this cluster are scarce and heavily contested (confirmed directly:
#      a single H200 request sat queued for 4+ minutes while an untyped/
#      RTXA6000 request started in 6s). Requesting any GPU lets each of the
#      57 array tasks land on whatever's actually free (RTXA6000/A40/A100/
#      A30/H100/...), maximizing real throughput over waiting on one model.
#   2. OUT_DIR points at a FRESH latents_fixed/ directory, not the existing
#      latents/. Reusing the old directory is dangerous: the resumable
#      shard/manifest logic would see the old (pre-fix, frame-invariant)
#      shard counts and conclude those frames are already done, silently
#      skipping real re-extraction entirely. Once validated, swap
#      latents_fixed/ in for latents/ rather than merging them.
# No --array=%K throttle -- deliberately submitting all 57 at once and
# letting the scheduler maximize real concurrency, since secondary's actual
# availability fluctuates fast (observed 11 idle GPUs drop to 2 within
# minutes) and an artificial cap would only slow this down.
#SBATCH --job-name=flux_swarm_fixed
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
#SBATCH --array=0-56   # 19 proteins × 3 replicas (1bcp_F unavailable, verified 2026-08-13)

mkdir -p logs

source activate fluxion
cd /u/arjunc4/fluxion

ATLAS_RAW=/scratch/arjunc4/atlas_raw
OUT_DIR=/scratch/arjunc4/fluxion_data/latents_fixed

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
