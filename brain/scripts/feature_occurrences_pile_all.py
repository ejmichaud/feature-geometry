""" 
Counts the number of times each SAE feature fires across some corpus.
"""
import os
# os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
os.environ['HF_HOME'] = '/om2/user/ericjm/.cache/huggingface' # file lock issue in /om 
import argparse
import re

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import datasets
from huggingface_hub import hf_hub_download, list_repo_files, notebook_login
# from nnsight import NNsight
from transformer_lens import HookedTransformer


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

def gather_residual_activations(model, target_layer, inputs):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs
    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act

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
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch.float32 
    # device_map = {'': 0} if device.type == 'cuda' else None

    print("Loading model...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     device_map=device_map,
    #     torch_dtype=dtype,
    #     low_cpu_mem_usage=True,
    # )
    # tokenizer =  AutoTokenizer.from_pretrained(args.model)
    # model = NNsight(model, device=device)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
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

    # n_layers = 26 if args.model == "google/gemma-2-2b" else 42
    
    repo_id = "google/gemma-scope-2b-pt-res" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-res"
    for layeri in args.layers:
        sae_name = closest_l0_name(res_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        res_saes.append(sae)
        res_saes_names.append(sae_name)
    
    repo_id = "google/gemma-scope-2b-pt-mlp" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-mlp"
    for layeri in args.layers:
        sae_name = closest_l0_name(mlp_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        mlp_saes.append(sae)
        mlp_saes_names.append(sae_name)
    
    repo_id = "google/gemma-scope-2b-pt-att" if args.model == "google/gemma-2-2b" else "google/gemma-scope-9b-pt-att"
    for layeri in args.layers:
        sae_name = closest_l0_name(att_saes_available, layeri, args.sae_features, args.target_l0)
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_name}/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        sae.eval()
        att_saes.append(sae)
        att_saes_names.append(sae_name)

    print("Loading dataset...")
    dataset = datasets.load_dataset(DATASET, streaming=True, split="train")

    res_histograms = {
        sae_name: np.zeros(sae.W_enc.shape[1]) for sae_name, sae in zip(res_saes_names, res_saes)
    }
    mlp_histograms = {
        sae_name: np.zeros(sae.W_enc.shape[1]) for sae_name, sae in zip(mlp_saes_names, mlp_saes)
    }
    att_histograms = {
        sae_name: np.zeros(sae.W_enc.shape[1]) for sae_name, sae in zip(att_saes_names, att_saes)
    }
    n_tokens = 0
    for i, doc in tqdm(enumerate(dataset), total=args.n_docs):
        if i >= args.n_docs:
            break
        inputs = model.tokenizer.encode(
            doc['text'], 
            return_tensors="pt", 
            add_special_tokens=True,
            max_length=1024,  
            truncation=True  # Add this line
        ).to(device)
        _, cache = model.run_with_cache(inputs)
        for layeri, sae_name, sae in zip(args.layers, res_saes_names, res_saes):
            target_act = cache[f'blocks.{layeri}.hook_resid_post']
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]
            res_histograms[sae_name] += sae_acts.sum(dim=0).cpu().numpy()

        for layeri, sae_name, sae in zip(args.layers, mlp_saes_names, mlp_saes):
            target_act = cache[f'blocks.{layeri}.hook_mlp_out']
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]
            mlp_histograms[sae_name] += sae_acts.sum(dim=0).cpu().numpy()
        
        for layeri, sae_name, sae in zip(args.layers, att_saes_names, att_saes):
            target_act = cache[f'blocks.{layeri}.attn.hook_z'] # shape (1, seq_len, n_heads, d_head)
            _, seq_len, n_heads, d_head = target_act.shape
            target_act = target_act.reshape(1, seq_len, n_heads * d_head)
            sae_acts = sae.encode(target_act)
            sae_acts = (sae_acts > 1).float()
            sae_acts = sae_acts[0, 1:]
            att_histograms[sae_name] += sae_acts.sum(dim=0).cpu().numpy()
        n_tokens += sae_acts.shape[0]
    
    # save histograms
    os.makedirs(args.save_dir, exist_ok=True)
    for sae_name, histogram in res_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_res_{sae_name.replace('/', '_')}_occurrences.npz"),
            histogram=histogram,
            n_tokens=n_tokens,
        )
    for sae_name, histogram in mlp_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_mlp_{sae_name.replace('/', '_')}_occurrences.npz"),
            histogram=histogram,
            n_tokens=n_tokens,
        )
    for sae_name, histogram in att_histograms.items():
        np.savez(
            os.path.join(args.save_dir, f"pile_{args.model.replace('/', '_')}_att_{sae_name.replace('/', '_')}_occurrences.npz"),
            histogram=histogram,
            n_tokens=n_tokens,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use for feature extraction.", default="google/gemma-2-2b", choices=["google/gemma-2-2b", "google/gemma-2-9b"])
    parser.add_argument("--sae_features", type=str, help="Number of features in SAE, formatted like 16k, 32k, 65k, 131k, etc.", default="16k")
    parser.add_argument('--layers', nargs='+', type=int, help='List of integers specifying layer indices.')
    parser.add_argument('--target_l0', type=int, help='The target l0 to use when selecting SAEs.', default=50)
    parser.add_argument("--n_docs", type=int, help="The number of documents to analyze.", default=10_000)
    parser.add_argument("--save_dir", type=str, help="The directory to save the histogram to.", default="histograms")
    args = parser.parse_args()
    main(args)
