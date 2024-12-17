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

seed = 83
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For reproducibility in certain CUDA operations (may slow down performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed)
random.seed(seed)

# %%

layer = 0
# Directory containing CSV files
csv_directory = f"./gemma-fv-layer{layer}-norm0-diff/"
normalized = 0
if normalized == 0:
    use_normalize = False
else:
    use_normalize = True

# List to store numpy arrays from each CSV file
data_list = []
label_list = []

idx = 0

json_idx_to_file_map = dict()
# Iterate over CSV files
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith(".csv"):
        # Read the CSV file into a pandas DataFrame
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path, header = None)
        

        # Convert the DataFrame to a NumPy array
        array = df.to_numpy()
        print(csv_file, array.shape)
        
        # Append the NumPy array to the list
        data_list.append(array)
        label_list.extend([idx] * array.shape[0])
        json_idx_to_file_map[idx] = csv_file[:-4]

        idx += 1

# Concatenate all the arrays into a single NumPy array
concatenated_array = np.concatenate(data_list, axis=0)
label_array = np.array(label_list)

if use_normalize:
    concatenated_array = (concatenated_array - np.min(concatenated_array, axis=0)) / np.linalg.norm(concatenated_array, axis=1)[:, None]

# Optionally, convert the NumPy array into a PyTorch tensor
concatenated_tensor = torch.tensor(concatenated_array)
label_tensor = torch.tensor(np.array(label_list))
n_clusters = torch.max(label_tensor).item() + 1

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score

## Perform LDA
n_comp_list = list(range(2,12))
for n_comp in n_comp_list:
    lda = LDA(n_components=n_comp)
    lda_result = lda.fit_transform(concatenated_array, label_array)
    with open(f"./Data/layer{layer}_{normalized}_lda-data.csv","a") as file:
        score = silhouette_score(lda_result, label_array)
        print(n_comp,score)
        file.write(f"{n_comp},{score}\n")
# %%

colors = []
num_colors = 100
random.seed(99)
for _ in range(num_colors):
    # Generate a random color in hexadecimal format
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    colors.append(color)

def plot_pca(input_data, label_array, dim = 2, plot_original = False):

    pca = PCA(n_components=dim)
    input_data = (input_data - np.mean(input_data, axis=0)) / np.std(input_data, axis=0)
    pca_result = pca.fit_transform(input_data)
    if plot_original:
        pca_result = input_data

    plt.rcParams.update({'font.size':16})
    plt.figure(figsize=(4, 3))
    for label in range(np.max(label_array) + 1):
        print(label)
        plt.scatter(pca_result[label_array == label, 0], pca_result[label_array == label, 1], label=f'{json_idx_to_file_map[label]}',color=colors[label])

#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
#    plt.xlim(-30,30)
#    plt.ylim(-30,30)

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
lda = LDA(n_components=8)
lda_result = lda.fit_transform(concatenated_array, label_array)
np.save(f'lda_layer{layer}_norm{normalized}_scalings.npy', lda.scalings_)

# %%
## Plot PCA results
plot_pca(concatenated_array, label_array)
# %%
## Plot LDA results
plot_pca(lda_result, label_array, plot_original=True)

# %%
## Plot LDA results
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':18})
for use_norm in [0]:
    for layer in [0,12]:
        filename = f"./Data/layer{layer}_{use_norm}_lda-data.csv"
        df = pd.read_csv(filename, header=None)
        plot_string = '-o' if use_norm == 0 else '--x'
        plot_color = 'b' if layer == 0 else 'r'
        plt.plot(df.iloc[:,0],df.iloc[:,1],plot_color+plot_string,label=f"Layer {layer}")

plt.legend()
plt.xlabel('Dimension')
plt.ylabel('Silhouette Score')
plt.legend()
plt.axhline(y=-0.07897946263811903,linestyle="--",c='k')
plt.tight_layout()
plt.savefig(f'lda-silhouette.pdf',bbox_inches='tight')