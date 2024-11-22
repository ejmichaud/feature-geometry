#!/usr/bin/env python3

import os
os.environ['HF_HOME'] = '/om2/user/ericjm/.cache/huggingface'
# os.environ['HF_HOME'] = '/Users/eric/.cache/huggingface'

import sys
import argparse

import numpy as np
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm
import torch

from huggingface_hub import hf_hub_download


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform spectral clustering on an SAE decoder feature angular similarity matrix.')
    parser.add_argument('--repo_id', help='hf repo name', default="google/gemma-scope-2b-pt-res")
    parser.add_argument('--sae_name', help='hf sae name', default="layer_12/width_16k/canonical")
    parser.add_argument('--n_clusters', required=True, nargs='+', type=int, metavar='CLUSTERS',
                        help='List of cluster sizes.')
    parser.add_argument('--save_dir', default=None, help='Directory to save the output .npz file.')
    return parser.parse_args()

def load_W_enc(repo_id, sae_name):
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=sae_name + "/params.npz",
        revision="0127b340ca980c3ee81df0275cea35f350f83488"
    )
    params = np.load(path_to_params)
    return torch.tensor(params['W_enc'])

def perform_spectral_clustering(matrix, n_clusters):
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels = clustering.fit_predict(matrix)
        print(f"Performed spectral clustering with {n_clusters} clusters.")
        return labels
    except Exception as e:
        print(f"Error during spectral clustering with {n_clusters} clusters: {e}")
        sys.exit(1)

def generate_output_filename(repo_id, sae_name, save_dir):
    name = f"{repo_id.replace('/', '_')}_{sae_name.replace('/', '_')}_ang_sims_clusters.npz"
    output_path = os.path.join(save_dir, name)
    return output_path

def main():
    args = parse_arguments()

    W_enc = load_W_enc(args.repo_id, args.sae_name) # shape (2304, 16384)
    W_enc = W_enc / torch.norm(W_enc, dim=0, keepdim=True)

    print("Computing angular similarities...")
    cos_sims = W_enc @ W_enc.T 
    cos_sims = torch.clamp(cos_sims, -1, 1)
    ang_sims = 1 - torch.acos(cos_sims) / np.pi

    # Initialize the dictionary to store cluster labels
    clusters_dict = {}

    for nc in tqdm(args.n_clusters):
        labels = perform_spectral_clustering(ang_sims, nc)
        clusters_dict[nc] = labels
    
    os.makedirs(args.save_dir, exist_ok=True)

    output_path = generate_output_filename(args.repo_id, args.sae_name, args.save_dir)

    # Save the clusters dictionary as a .npz file
    try:
        # Convert lists to numpy arrays for saving
        save_dict = {str(k): v for k, v in clusters_dict.items()}
        np.savez(output_path, **save_dict)
        print(f"Saved cluster labels to {output_path}.")
    except Exception as e:
        print(f"Error saving the output .npz file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
