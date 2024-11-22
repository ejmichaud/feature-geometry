#!/bin/bash
#SBATCH --job-name=eangs
#SBATCH --ntasks=2
#SBATCH --mem=32GB
#SBATCH --time=0-02:00:00
#SBATCH --output=/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/logs/slurm-%A_%a.out
#SBATCH --array=0-23

# Define the directory to save cluster results
SAVE_DIR="/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/clusters_enc_ang/"

# Array of input filenames
sae_names=(
    "layer_0/width_16k/average_l0_46"
    "layer_10/width_16k/average_l0_39"
    "layer_11/width_16k/average_l0_41"
    "layer_12/width_16k/average_l0_41"
    "layer_13/width_16k/average_l0_43"
    "layer_14/width_16k/average_l0_43"
    "layer_15/width_16k/average_l0_41"
    "layer_16/width_16k/average_l0_42"
    "layer_17/width_16k/average_l0_42"
    "layer_18/width_16k/average_l0_40"
    "layer_19/width_16k/average_l0_40"
    "layer_1/width_16k/average_l0_40"
    "layer_20/width_16k/average_l0_38"
    "layer_21/width_16k/average_l0_38"
    "layer_22/width_16k/average_l0_38"
    "layer_23/width_16k/average_l0_38"
    "layer_2/width_16k/average_l0_53"
    "layer_3/width_16k/average_l0_59"
    "layer_4/width_16k/average_l0_60"
    "layer_5/width_16k/average_l0_34"
    "layer_6/width_16k/average_l0_36"
    "layer_7/width_16k/average_l0_36"
    "layer_8/width_16k/average_l0_37"
    "layer_9/width_16k/average_l0_37"
)

# Ensure the SLURM_ARRAY_TASK_ID is within the bounds of the input_files array
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#sae_names[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    exit 1
fi

# Select the input file based on the current array task ID
SAE_NAME="${sae_names[$SLURM_ARRAY_TASK_ID]}"

# Execute the Python clustering script with the selected input
python /om2/user/ericjm/sae-clouds/experiments/all-occurrences0/cluster_enc_ang.py \
    --repo_id "google/gemma-scope-2b-pt-res" \
    --sae_name "${SAE_NAME}" \
    --n_clusters 2 4 8 16 32 64 128 256 512 1024 \
    --save_dir "${SAVE_DIR}"


