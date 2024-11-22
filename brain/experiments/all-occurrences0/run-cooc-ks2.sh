#!/bin/bash
#SBATCH --job-name=cooc2
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-24:00:00
#SBATCH --output=/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/logs/slurm-%A_%a.out
#SBATCH --array=0-2

# Define k values
k_values=(32 64 128)

# Get k value for this job
k=${k_values[$SLURM_ARRAY_TASK_ID]}

# Define layers
layers_arg="10 11 12 13 14"

# Create a unique save directory for each k value
save_dir="/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/cooc-histograms/"

# Ensure the save directory exists
mkdir -p $save_dir

python /om2/user/ericjm/sae-clouds/scripts/feature_cooccurrences_pile_all.py \
    --save_dir $save_dir \
    --n_docs 50_000 \
    --layers $layers_arg \
    --target_l0 50 \
    --k $k
