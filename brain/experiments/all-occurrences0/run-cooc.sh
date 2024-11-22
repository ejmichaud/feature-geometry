#!/bin/bash
#SBATCH --job-name=coocs
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --output=/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/logs/slurm-%A_%a.out
#SBATCH --array=0-6

# Calculate the starting layer for this job
start_layer=$((SLURM_ARRAY_TASK_ID * 4))

# Calculate the ending layer for this job
end_layer=$((start_layer + 3))

# Create the layer argument string
layers_arg=$(seq -s ' ' $start_layer $end_layer)

python /om2/user/ericjm/sae-clouds/scripts/feature_cooccurrences_pile_all.py \
    --save_dir /om2/user/ericjm/sae-clouds/experiments/all-occurrences0/cooc-histograms/ \
    --n_docs 50_000 \
    --layers $layers_arg \
    --target_l0 50 \
    --k 256

