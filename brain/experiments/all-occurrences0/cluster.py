#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform spectral clustering using Jaccard similarities computed from a matrix from a .npz file.')
    parser.add_argument('--input', required=True, help='Path to the input .npz file containing the matrix.')
    parser.add_argument('--n_clusters', required=True, nargs='+', type=int, metavar='CLUSTERS',
                        help='List of cluster sizes.')
    parser.add_argument('--save_dir', default=None, help='Directory to save the output .npz file.')
    return parser.parse_args()

def load_matrix(npz_path):
    cooccurrence_npz = np.load(npz_path)
    cooccurrence_hist, cooccurrence_n_chunks = cooccurrence_npz['histogram'], cooccurrence_npz['n_chunks']
    diag = np.diag(cooccurrence_hist)
    unions = diag[:, np.newaxis] + diag - cooccurrence_hist
    jaccard = cooccurrence_hist / unions
    # remove NaNs caused by division by zero
    jaccard[np.isnan(jaccard)] = 0
    return jaccard

def perform_spectral_clustering(matrix, n_clusters):
    try:
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels = clustering.fit_predict(matrix)
        print(f"Performed spectral clustering with {n_clusters} clusters.")
        return labels
    except Exception as e:
        print(f"Error during spectral clustering with {n_clusters} clusters: {e}")
        sys.exit(1)

def generate_output_filename(input_path, save_dir):
    filename = os.path.basename(input_path)
    filename = os.path.splitext(filename)[0]
    filename += "_clusters.npz"
    output_path = os.path.join(save_dir, filename)
    return output_path

def main():
    args = parse_arguments()

    jaccard = load_matrix(args.input)

    # Initialize the dictionary to store cluster labels
    clusters_dict = {}

    for nc in tqdm(args.n_clusters):
        labels = perform_spectral_clustering(jaccard, nc)
        clusters_dict[nc] = labels
    
    os.makedirs(args.save_dir, exist_ok=True)

    output_path = generate_output_filename(args.input, args.save_dir)

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
