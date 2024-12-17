# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

from transformers import AutoTokenizer

seed = 83
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# For reproducibility in certain CUDA operations (may slow down performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed)
random.seed(seed)

cache_dir = "/media/dbaek/sdd/MODELS"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir=cache_dir)

# %%

layer = 0
# Directory containing CSV files
csv_directory = f"./gemma-fv-layer{layer}-norm0-diff/"
json_directory = "./function_vectors/dataset_files/abstractive/"
normalized = 0
if normalized == 0:
    use_normalize = False
else:
    use_normalize = True

# List to store numpy arrays from each CSV file
data_list = []
label_list = []
label_pair_list = []

idx = 0

json_idx_to_file_map = dict()
# Iterate over CSV files
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith(".csv"):
        # Read the CSV file into a pandas DataFrame
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path, header = None)
        
        array = df.to_numpy()
        print(csv_file, array.shape)
        
        # Append the NumPy array to the list
        data_list.append(array)
        label_list.extend([idx] * array.shape[0])
        json_idx_to_file_map[idx] = csv_file[:-4]

        json_filename = csv_file[:-4] + ".json"
        with open(os.path.join(json_directory, json_filename)) as f:
            label_pair_data = json.load(f)
            for pair in label_pair_data:
                label_pair_list.append([tokenizer.tokenize(pair.get("input"), return_tensors='pt')[-1], tokenizer.tokenize(pair.get("output"), return_tensors='pt')[-1]])
                if "product" in csv_file and len(tokenizer.tokenize(pair.get("input"), return_tensors='pt')) == 1 and len(tokenizer.tokenize(pair.get("output"), return_tensors='pt')) == 1:
                    print(pair.get("input"), pair.get("output"))

        idx += 1

# Concatenate all the arrays into a single NumPy array
concatenated_array = np.concatenate(data_list, axis=0) 
label_array = np.array(label_list)

concatenated_array = (concatenated_array - np.mean(concatenated_array, axis=0)) / np.std(concatenated_array, axis=0)

if use_normalize:
    concatenated_array = (concatenated_array - np.min(concatenated_array, axis=0)) / np.linalg.norm(concatenated_array, axis=1)[:, None]


concatenated_tensor = torch.tensor(concatenated_array)
label_tensor = torch.tensor(np.array(label_list))


# %%

colors = []
num_colors = 100
random.seed(99)
for _ in range(num_colors):
    # Generate a random color in hexadecimal format
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    colors.append(color)

input_data = concatenated_array
dim = 2


pca = PCA(n_components=dim)
input_data = (input_data - np.mean(input_data, axis=0)) / np.std(input_data, axis=0)
pca_result = pca.fit_transform(input_data)
sorted_indices = np.argsort(-pca_result[:, 0])

new_x_list = []
new_y_list = []

plt.figure()
for i in range(len(sorted_indices)):
    plt.scatter(len(label_pair_list[sorted_indices[i]][1]) - len(label_pair_list[sorted_indices[i]][0]),pca_result[sorted_indices[i], 0])
    new_x_list.append(len(label_pair_list[sorted_indices[i]][1]) - len(label_pair_list[sorted_indices[i]][0]))
    new_y_list.append(pca_result[sorted_indices[i], 0])
    if i % 1000 == 0:
        print(i)


plt.rcParams.update({'font.size':16})
plt.figure(figsize=(4, 3))
for label in range(np.max(label_array) + 1):
    print(label)
    plt.scatter(pca_result[label_array == label, 0], pca_result[label_array == label, 1], label=f'{json_idx_to_file_map[label]}',color=colors[label])

plt.xlabel('PC0')
plt.ylabel('PC1')

# %%

# Fit a line to the data
coefficients = np.polyfit(new_x_list, new_y_list, 1)
poly_eqn = np.poly1d(coefficients)
y_fit = poly_eqn(new_x_list)

# Plot the linear fit
plt.plot(new_x_list, y_fit, color='red', label='Best fit line')
plt.scatter(new_x_list, new_y_list, color='blue')


# %%
from collections import defaultdict
grouped_points = defaultdict(list)
for xi, yi in zip(new_x_list, new_y_list):
    grouped_points[xi].append(yi)

# Compute the mean and standard deviation for each group
x_unique = []
y_mean = []
y_std = []

for xi, yi_list in sorted(grouped_points.items()):
    x_unique.append(xi)
    y_mean.append(np.mean(yi_list))
    y_std.append(np.std(yi_list))

# Fit a line to the mean y-values
coefficients = np.polyfit(x_unique, y_mean, 1)  # Linear fit (degree 1)
poly_eqn = np.poly1d(coefficients)
y_fit = poly_eqn(x_unique)

# Plot the data
plt.errorbar(x_unique, y_mean, yerr=y_std, fmt='o', label='Mean and stdev', capsize=5)
plt.plot(x_unique, y_fit, color='red', label='Best fit line')

plt.xlabel('Last Token Length Difference')
plt.ylabel('Principal Component 1')
plt.tight_layout()
plt.savefig('pc0_vs_last_token_length_diff.pdf', bbox_inches='tight')
plt.show()
# %%
