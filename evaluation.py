from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math
from torch.cuda.amp import autocast
import argparse
from sampler_jay import mdm_sampling, mdm_style_sampling
from data.text import setup_tokeniser, get_text_dataset, decode_sequence_with_mask, wt_detokeniser
import mauve
from typing import List
from train import TransdimensionalFlowModule
from train_MDM import MDM
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Evaluation library for discrete diffusion models
# ------------------------------------------------------------

# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def model_load(model_name: str):
    if model_name == "vlmdm":
        checkpoint_path = f"/n/netscratch/sham_lab/Everyone/jay_brian/our_model/epoch-epoch=505-val_loss-val_loss=1.8743.ckpt"
        model = TransdimensionalFlowModule.load_from_checkpoint(checkpoint_path)
    elif model_name == "mdm":
        checkpoint_path = f"/n/netscratch/sham_lab/Everyone/jay_brian/mdm/epoch-epoch=73-val_loss-val_loss=1.7349.ckpt"
        model = MDM.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Model {model_name} not found")
    print("Model loaded")
    return model

# ------------------------------------------------------------
# Generative perplexity and entropy
# This just works on H100 GPU, because of the memory issue of loading Llama-7B
# ------------------------------------------------------------

def eval_gen_ppl(text_samples, max_length: int = 512, retokenize: bool = True) -> None:
    # load a pretrained llama 7B
    custom_cache_dir = "/n/netscratch/sham_lab/Lab/workdir/llama7b"
    eval_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=custom_cache_dir, torch_dtype=torch.float16).eval()
    eval_model = eval_model.to('cuda')
    eval_model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=custom_cache_dir)
    eval_model_tokenizer.pad_token = eval_model_tokenizer.eos_token

    # tokenize the batch
    tokenized= eval_model_tokenizer(text_samples, return_tensors="pt", 
        padding = 'max_length', truncation = True, max_length = max_length).to(eval_model.device)
    input_ids = tokenized['input_ids']

    eos_token_id = eval_model_tokenizer.eos_token_id
    eos_mask = (input_ids == eos_token_id)
    first_eos = eos_mask.cumsum(dim=-1) == 1 

    # generative perplexity
    with autocast(), torch.no_grad():
        outputs = eval_model(input_ids)
        logits = outputs.logits

    logits = logits.transpose(-1,-2) # size b X D X N, D = the number of possible tokens
    nlls = F.cross_entropy(logits[...,:-1], input_ids[..., 1:], reduction='none')
    effective_mask = (first_eos[..., 1:] + (input_ids[..., 1:] != eos_token_id)).bool()
    nlls = nlls * effective_mask

    # compute per-sample perplexity
    likelihood_list = []
    for b in range(input_ids.size(0)):
        nll = nlls[b]
        mask = effective_mask[b]
        likelihood = nll.sum() / mask.sum()
        likelihood_list.append(likelihood.item())

    entropy_avg = calculate_entropy(input_ids, first_eos)

    return sum(likelihood_list) / len(likelihood_list), entropy_avg


def calculate_entropy(tokenized_tensor_batch, first_eos_batch):
    entropies = []

    for tokenized_tensor, first_eos in zip(tokenized_tensor_batch, first_eos_batch):
        # Find the index of the first EOS token
        if first_eos.any():
            first_eos_idx = torch.where(first_eos)[0][0].item()  # Get the first index where EOS is True
            valid_tokens = tokenized_tensor[:first_eos_idx]  # Slice tokens up to the first EOS
        else:
            valid_tokens = tokenized_tensor  # If no EOS, use all tokens

        # Ensure valid_tokens is a list for entropy calculation
        if isinstance(valid_tokens, torch.Tensor):
            valid_tokens = valid_tokens.tolist()

        token_counts = Counter(valid_tokens)
        N = len(valid_tokens)
        probabilities = {token: count / N for token, count in token_counts.items()}

        entropy = -sum(p * math.log(p) for p in probabilities.values())
        entropies.append(entropy)

    return sum(entropies) / len(entropies)


# ------------------------------------------------------------
# MAUVE score
# NOTE: MAUVE score has not been debugged yet
# ------------------------------------------------------------

def calculate_mauve_score(text_samples: List[str], max_length: int, reference_dataset: str = "wikitext2", num_reference: int = 2) -> float:
    # load true (reference) dataset
    dataset = get_text_dataset(
        name = reference_dataset,
        split = "train",
        max_length = max_length,
        num_proc = 8
    )

    # convert token IDs back to strings
    tokeniser = setup_tokeniser()
    pad_token, mask_token = tokeniser.pad_token_id, tokeniser.mask_token_id

    # decode reference samples
    refer_tokens = dataset[:num_reference]["input_ids"]
    refer_texts = decode_sequence_with_mask(refer_tokens, tokeniser, pad_token, mask_token)
    refer_texts = [wt_detokeniser(text) for text in refer_texts]

    refer_texts = refer_texts[:num_reference]
    text_samples = text_samples[:num_reference]

    # compute MAUVE score
    mauve_result = mauve.compute_mauve(
        p_text = refer_texts,
        q_text = text_samples,
        device_id = 0,
        max_text_length = max_length,
        verbose = True
    )

    return mauve_result.mauve

# ------------------------------------------------------------
# Length statistics
# ------------------------------------------------------------

def length_stat_plot(metrics, model_name = "mdm", data = "wikitext2"):
    plt.figure(figsize=(10, 5))
    plt.hist(metrics, bins = 50, edgecolor="steelblue", alpha=0.85)
    plt.xlabel("Sequence Length")
    plt.title(f"Length distribution of generated samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pngs/length_distribution_generated_samples_{model_name}_{data}.png")
    plt.close()


# ------------------------------------------------------------
# main function
# ------------------------------------------------------------

def main(args):
    model = model_load(args.model)
    print("Evaluation started for model: ", args.model)
    tokeniser = setup_tokeniser()
    processing_size = 16
    num_batches = math.ceil(args.num_samples / processing_size)
    metrics = []

    for b_idx in range(num_batches):
        # --- draw one batch of samples ---
        if args.model == "vlmdm":
            sampling_trace, len_trace = mdm_style_sampling(model, 
            num_steps = args.num_steps,
            tokeniser = tokeniser,
            mask = tokeniser.mask_token_id,
            pad = tokeniser.pad_token_id,
                batch_size = processing_size,
                max_length = args.max_length,
                temperature = args.temperature)
        else:
            sampling_trace, len_trace = mdm_sampling(model, 
                num_steps = args.num_steps,
                tokeniser = tokeniser,
                mask = tokeniser.mask_token_id,
                pad = tokeniser.pad_token_id,
                batch_size = processing_size,
                max_length = args.max_length,
                temperature = args.temperature)

        # --- compute metrics ---
        if args.mauve:
            # TODO: fill in the code for MAUVE score
            pass

        if args.genppl:
            sample = sampling_trace[-1]
            likelihood, entropy = eval_gen_ppl(sample, args.max_length)
            print(f"{b_idx}th batch: Likelihood: {likelihood}, Entropy: {entropy}")
            metrics.append({"likelihood": likelihood, "entropy": entropy})

        if args.len_stats:
            lengths = len_trace[-1][1].tolist()
            print(f"{b_idx}th batch: Lengths: {lengths}")
            metrics.extend(lengths)
    
    if args.genppl:
        print(f"Average likelihood: {sum([m['likelihood'] for m in metrics]) / len(metrics)}")
        print(f"Average entropy: {sum([m['entropy'] for m in metrics]) / len(metrics)}")
    
    if args.mauve:
        # TODO: implement the code for MAUVE score
        pass

    if args.len_stats:
        length_stat_plot(metrics, model_name = args.model, data = "wikitext2")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mdm")
    # metrics
    parser.add_argument("--mauve", action="store_true")
    parser.add_argument("--genppl", action="store_true")
    parser.add_argument("--len_stats", action="store_true")
    # sampler parameters
    parser.add_argument("--num_steps", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=32, help="need to be the multiple of 16")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for categorical sampling")
    args = parser.parse_args()
    main(args)