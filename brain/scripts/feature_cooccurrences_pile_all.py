""" 
Counts the co-occurrences of SAE features across some corpus.
"""
import os
os.environ['HF_HOME'] = '/om2/user/ericjm/.cache/huggingface'  
import argparse
import re
import time

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import datasets
from huggingface_hub import hf_hub_download, list_repo_files, notebook_login
from transformer_lens import HookedTransformer

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

def closest_l0_name(sae_names, layer, width, l0):
    """
    Given a list of SAE names, like `layer_0/width_16k/average_l0_50/params.npz`, 
    returns the SAE name at that layer, with that width, with the closest l0 to the given l0.
    """
    layer = str(layer)
    width = str(width)
    pattern = rf"^layer_{re.escape(layer)}/width_{re.escape(width)}/average_l0_(\d+)/params\.npz$"
    l0s = []
    for sae_name in sae_names:
        match = re.match(pattern, sae_name)
        if match:
            l0s.append(int(match.group(1)))
    if not l0s:
        raise ValueError(f"No SAEs found for layer {layer} and width {width}.")
    closest_l0 = l0s[np.argmin(np.abs(np.array(l0s) - l0))]
    return f"layer_{layer}/width_{width}/average_l0_{closest_l0}"

@torch.no_grad()
def main(args):

    DATASET = "monology/pile-uncopyrighted"

    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        args.model,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print("device:", device)
    print("dtype:", dtype)

    print("Layer indices:", args.layers)

    print("Loading SAEs...")
    res_saes_available = list_repo_files(
        "google/gemma-scope-2b-pt-res" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-res"
    )
    att_saes_available = list_repo_files(
        "google/gemma-scope-2b-pt-att" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-att"
    )
    mlp_saes_available = list_repo_files(
        "google/gemma-scope-2b-pt-mlp" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-mlp"
    )

    res_saes = []
    res_saes_names = []

    mlp_saes = []
    mlp_saes_names = []

    att_saes = []
    att_saes_names = []

    # Load residual SAEs
    repo_id = "google/gemma-scope-2b-pt-res" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-res"
    for layeri in args.layers:
        sae_name = closest_l0_name(res_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        res_saes.append(sae)
        res_saes_names.append(sae_name)

    # Load MLP SAEs
    repo_id = "google/gemma-scope-2b-pt-mlp" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-mlp"
    for layeri in args.layers:
        sae_name = closest_l0_name(mlp_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        mlp_saes.append(sae)
        mlp_saes_names.append(sae_name)

    # Load Attention SAEs
    repo_id = "google/gemma-scope-2b-pt-att" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-att"
    for layeri in args.layers:
        sae_name = closest_l0_name(att_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        att_saes.append(sae)
        att_saes_names.append(sae_name)

    # Initialize co-occurrence histograms
    res_cooc_histograms = {
        sae_name: torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
        for sae_name, sae in zip(res_saes_names, res_saes)
    }
    mlp_cooc_histograms = {
        sae_name: torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
        for sae_name, sae in zip(mlp_saes_names, mlp_saes)
    }
    att_cooc_histograms = {
        sae_name: torch.zeros((sae.W_enc.shape[1], sae.W_enc.shape[1]), device=device, dtype=torch.int32)
        for sae_name, sae in zip(att_saes_names, att_saes)
    }

    res_n_chunks = {sae_name: 0 for sae_name in res_saes_names}
    mlp_n_chunks = {sae_name: 0 for sae_name in mlp_saes_names}
    att_n_chunks = {sae_name: 0 for sae_name in att_saes_names}

    print("Loading dataset...")
    dataset = datasets.load_dataset(DATASET, streaming=True, split="train")

    loop_t0 = time.time()
    for i, doc in tqdm(enumerate(dataset), total=args.n_docs):
        if i >= args.n_docs:
            break
        inputs = model.tokenizer.encode(
            doc['text'], 
            return_tensors="pt", 
            add_special_tokens=True,
            max_length=1024,  
            truncation=True
        ).to(device)
        _, cache = model.run_with_cache(inputs)

        # Process res SAEs
        for layeri, sae_name, sae in zip(args.layers, res_saes_names, res_saes):
            target_act = cache[f'blocks.{layeri}.hook_resid_post']
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]  # remove BOS token

            for j in range(0, sae_acts.shape[0], args.k):
                if j + args.k <= sae_acts.shape[0]:
                    chunk = sae_acts[j:j+args.k]
                    chunk_features = torch.any(chunk, dim=0)
                    co_occurrences = torch.outer(chunk_features, chunk_features)
                    res_cooc_histograms[sae_name] += co_occurrences.int()
                    res_n_chunks[sae_name] += 1

        # Process mlp SAEs
        for layeri, sae_name, sae in zip(args.layers, mlp_saes_names, mlp_saes):
            target_act = cache[f'blocks.{layeri}.hook_mlp_out']
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]  # remove BOS token

            for j in range(0, sae_acts.shape[0], args.k):
                if j + args.k <= sae_acts.shape[0]:
                    chunk = sae_acts[j:j+args.k]
                    chunk_features = torch.any(chunk, dim=0)
                    co_occurrences = torch.outer(chunk_features, chunk_features)
                    mlp_cooc_histograms[sae_name] += co_occurrences.int()
                    mlp_n_chunks[sae_name] += 1

        # Process att SAEs
        for layeri, sae_name, sae in zip(args.layers, att_saes_names, att_saes):
            target_act = cache[f'blocks.{layeri}.attn.hook_z']  # shape (1, seq_len, n_heads, d_head)
            _, seq_len, n_heads, d_head = target_act.shape
            target_act = target_act.reshape(1, seq_len, n_heads * d_head)
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]  # remove BOS token

            for j in range(0, sae_acts.shape[0], args.k):
                if j + args.k <= sae_acts.shape[0]:
                    chunk = sae_acts[j:j+args.k]
                    chunk_features = torch.any(chunk, dim=0)
                    co_occurrences = torch.outer(chunk_features, chunk_features)
                    att_cooc_histograms[sae_name] += co_occurrences.int()
                    att_n_chunks[sae_name] += 1

    loop_tot = time.time() - loop_t0
    print(f"Total time: {loop_tot} s")

    # Save histograms
    os.makedirs(args.save_dir, exist_ok=True)
    for sae_name, histogram in res_cooc_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_res_{sae_name.replace('/', '_')}_docs{args.n_docs // 1_000}k_keq{args.k}_cooccurrences.npz"),
            histogram=histogram.cpu().numpy(),
            n_chunks=res_n_chunks[sae_name],
        )
    for sae_name, histogram in mlp_cooc_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_mlp_{sae_name.replace('/', '_')}_docs{args.n_docs // 1_000}k_keq{args.k}_cooccurrences.npz"),
            histogram=histogram.cpu().numpy(),
            n_chunks=mlp_n_chunks[sae_name],
        )
    for sae_name, histogram in att_cooc_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_att_{sae_name.replace('/', '_')}_docs{args.n_docs // 1_000}k_keq{args.k}_cooccurrences.npz"),
            histogram=histogram.cpu().numpy(),
            n_chunks=att_n_chunks[sae_name],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use for feature extraction.", default="google/gemma-2-2b", choices=["google/gemma-2-2b", "google/gemma-2-9b"])
    parser.add_argument("--sae_features", type=str, help="Number of features in SAE, formatted like 16k, 32k, 65k, 131k, etc.", default="16k")
    parser.add_argument('--layers', nargs='+', type=int, help='List of integers specifying layer indices.')
    parser.add_argument('--target_l0', type=int, help='The target l0 to use when selecting SAEs.', default=50)
    parser.add_argument("--n_docs", type=int, help="The number of documents to analyze.", default=10_000)
    parser.add_argument("--save_dir", type=str, help="The directory to save the histogram to.", default="histograms")
    parser.add_argument("--k", type=int, help="Co-occurrence memory (in tokens)", default=256)
    args = parser.parse_args()
    main(args)
