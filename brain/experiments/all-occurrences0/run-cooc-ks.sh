#!/bin/bash
#SBATCH --job-name=coocsks
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-6:00:00
#SBATCH --output=/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/logs/slurm-%A_%a.out
#SBATCH --array=0-7

# Define k values
k_values=(1 4 16 64)

# Calculate indices for layer group and k value
layer_group_index=$((SLURM_ARRAY_TASK_ID / 4))
k_index=$((SLURM_ARRAY_TASK_ID % 4))

# Get k value for this job
k=${k_values[$k_index]}

# Calculate the starting layer for this job
start_layer=$((layer_group_index * 12))

# Create the layer argument string
layers_arg=""
for i in {0..3}
do
    layer=$((start_layer + i * 4))
    layers_arg+="$layer "
done

# Trim the trailing space
layers_arg=$(echo $layers_arg | sed 's/ $//')

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
