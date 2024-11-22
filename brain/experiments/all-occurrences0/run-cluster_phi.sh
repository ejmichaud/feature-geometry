#!/bin/bash
#SBATCH --job-name=phis
#SBATCH --ntasks=2
#SBATCH --mem=32GB
#SBATCH --time=0-02:00:00
#SBATCH --output=/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/logs/slurm-%A_%a.out
#SBATCH --array=0-23

# Define the directory containing input files
INPUT_DIR="/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/cooc-histograms"

# Define the directory to save cluster results
SAVE_DIR="/om2/user/ericjm/sae-clouds/experiments/all-occurrences0/clusters_phi/"

# Array of input filenames
input_files=(
    "pile_google_gemma-2-2b_res_layer_0_width_16k_average_l0_46_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_10_width_16k_average_l0_39_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_11_width_16k_average_l0_41_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_12_width_16k_average_l0_41_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_13_width_16k_average_l0_43_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_14_width_16k_average_l0_43_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_15_width_16k_average_l0_41_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_16_width_16k_average_l0_42_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_17_width_16k_average_l0_42_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_18_width_16k_average_l0_40_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_19_width_16k_average_l0_40_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_1_width_16k_average_l0_40_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_20_width_16k_average_l0_38_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_21_width_16k_average_l0_38_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_22_width_16k_average_l0_38_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_23_width_16k_average_l0_38_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_2_width_16k_average_l0_53_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_3_width_16k_average_l0_59_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_4_width_16k_average_l0_60_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_5_width_16k_average_l0_34_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_6_width_16k_average_l0_36_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_7_width_16k_average_l0_36_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_8_width_16k_average_l0_37_docs50k_keq256_cooccurrences.npz"
    "pile_google_gemma-2-2b_res_layer_9_width_16k_average_l0_37_docs50k_keq256_cooccurrences.npz"
)

# Ensure the SLURM_ARRAY_TASK_ID is within the bounds of the input_files array
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#input_files[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds."
    exit 1
fi

# Select the input file based on the current array task ID
INPUT_FILE="${input_files[$SLURM_ARRAY_TASK_ID]}"

# Full path to the selected input file
FULL_INPUT_PATH="${INPUT_DIR}/${INPUT_FILE}"

# Execute the Python clustering script with the selected input
python /om2/user/ericjm/sae-clouds/experiments/all-occurrences0/cluster_phi.py \
    --input "${FULL_INPUT_PATH}" \
    --n_clusters 2 4 8 16 32 64 128 256 512 1024 \
    --save_dir "${SAVE_DIR}"
