#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform spectral clustering on a matrix from a .npz file.')
    parser.add_argument('--input', required=True, help='Path to the input .npz file containing the matrix.')
    parser.add_argument('--n_clusters', required=True, nargs='+', type=int, metavar='CLUSTERS',
                        help='List of cluster sizes.')
    parser.add_argument('--save_dir', default=None, help='Directory to save the output .npz file.')
    return parser.parse_args()

def binary_entropy(p):
    deterministic = (p == 0) | (p == 1)
    entropies = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    entropies[deterministic] = 0
    return entropies

def load_matrix(npz_path, split=4):
    """Computes mutual information between feature activations instead."""
    cooccurrence_npz = np.load(npz_path)
    cooccurrence_hist, cooccurrence_n_chunks = cooccurrence_npz['histogram'], cooccurrence_npz['n_chunks']
    marginals_on = cooccurrence_hist.diagonal() / cooccurrence_n_chunks
    joints_on = cooccurrence_hist / cooccurrence_n_chunks

    joints = np.zeros((*joints_on.shape, 2, 2))
    joints[:, :, 0, 0] = joints_on
    joints[:, :, 0, 1] = marginals_on[:, np.newaxis] - joints_on
    joints[:, :, 1, 0] = marginals_on[np.newaxis, :] - joints_on
    joints[:, :, 1, 1] = 1 - marginals_on[np.newaxis, :] - marginals_on[:, np.newaxis] + joints_on

    assert joints.shape[0] % split == 0
    assert joints.shape[1] % split == 0

    split_sizei = joints.shape[0] // split
    split_sizej = joints.shape[1] // split

    mis = np.zeros_like(joints_on)

    for idxi in tqdm(range(0, joints_on.shape[0], split_sizei)):
        for idxj in tqdm(range(0, joints_on.shape[1], split_sizej), leave=False):
            j = joints[idxi:idxi + split_sizei, idxj:idxj + split_sizej]
            P_A = np.sum(j, axis=3)
            P_B = np.sum(j, axis=2)
            outer = np.einsum('...i,...j->...ij', P_A, P_B)
            mi = np.sum(mask * j * np.log2(j / outer), axis=(2, 3))
            mis[idxi:idxi + split_sizei, idxj:idxj + split_sizej] = mi
    np.fill_diagonal(mis, binary_entropy(marginals_on))

    import code; code.interact(local=locals())
    exit()

    return mis

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
    filename += "_mis_clusters.npz"
    output_path = os.path.join(save_dir, filename)
    return output_path

def main():
    args = parse_arguments()

    mis = load_matrix(args.input)

    # Initialize the dictionary to store cluster labels
    clusters_dict = {}

    for nc in tqdm(args.n_clusters):
        labels = perform_spectral_clustering(mis, nc)
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
