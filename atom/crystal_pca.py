# %%
import os
import sys
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import numpy as np

# Load Model and Tokenizer
cache_dir = "/media/dbaek/sdd/MODELS"
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32, cache_dir=cache_dir)

# Directory containing JSON files
json_directory = "./function_vectors/dataset_files/abstractive/"

file_name_list = [
    "country-capital.json",
    "country-currency.json",
    "english-french.json",
    "english-german.json",
    "english-spanish.json",
    "landmark-country.json",
    "person-instrument.json",
    "person-occupation.json",
    "person-sport.json",
    "present-past.json",
    "product-company.json",
    "singular-plural.json"
]

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--layer', type=int, default=0,required=True)
parser.add_argument('--normalize', type=int, default=0,required=True)
args = parser.parse_args()

layer_num = args.layer
if args.normalize == 0:
    use_normalization = False
else:
    use_normalization = True


def compute_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_num][0][-1].cpu().numpy()

# Store vector differences and labels for coloring
all_vector_diffs = []
json_file_labels = []

json_idx_to_file_map = dict()

output_folder = f"./gemma-fv-layer{layer_num}-norm{args.normalize}-diff/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over JSON files
cnt = 0
for idx, json_file in enumerate(os.listdir(json_directory)):
    if json_file.endswith(".json") and json_file in file_name_list:
        with open(os.path.join(json_directory, json_file)) as f:
            data = json.load(f)
            print(json_file,len(data))
            for pair in data:
                print(pair, cnt)
                sys.stdout.flush()
                cnt += 1
                subject_text = pair.get('input')
                object_text = pair.get('output')

                # Compute the embeddings for subject and object
                subject_embedding = compute_embedding(subject_text)
                object_embedding = compute_embedding(object_text)

                # Compute the vector difference
                vector_diff = subject_embedding - object_embedding
                if use_normalization:
                    vector_diff = vector_diff / np.linalg.norm(vector_diff)
                    if np.isnan(vector_diff).any():
                        print("NaN detected")
                        continue
                all_vector_diffs.append(vector_diff)
                json_file_labels.append(idx)  # Label by file index

                with open(f"{output_folder}/{json_file[:-5]}.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(vector_diff)

                json_idx_to_file_map[idx] = json_file[:-5]
                

# %%
# Convert the list of vectors into a matrix for PCA
all_vector_diffs_matrix = torch.tensor(all_vector_diffs)

colors = []
num_colors = 100
random.seed(31)
for _ in range(num_colors):
    # Generate a random color in hexadecimal format
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    colors.append(color)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_vector_diffs_matrix)

# Plot results, coloring by JSON file
plt.figure(figsize=(10, 8))
for idx, label in enumerate(set(json_file_labels)):
    plt.scatter(pca_result[json_file_labels == label, 0], pca_result[json_file_labels == label, 1], label=f'{json_idx_to_file_map[idx]}',color=colors[label])


plt.title('PCA of Vector Differences from JSON Pairs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.savefig(f'gemma-layer{layer_num}-norm{args.normalize}.pdf')
plt.show()
