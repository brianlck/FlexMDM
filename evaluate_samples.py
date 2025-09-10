from transformers import LlamaForCausalLM, GPT2LMHeadModel, AutoTokenizer
import torch
import math
from collections import Counter
import argparse
from torch.cuda.amp import autocast
import torch.nn.functional as F

import json
import matplotlib.pyplot as plt
import os
from data import get_text_dataset
from data.text import setup_tokeniser_from_dataset

llama_model_path = "meta-llama/Llama-2-7b-hf"
gpt2_model_path = "gpt2-large"


def get_reference_text_dataset():
    dataset = get_text_dataset(
        "openwebtext",
        split="train",
        max_length=1024,
        filter_max_length=False
    )[:5000]["input_ids"]
    tokeniser = setup_tokeniser_from_dataset("openwebtext")
    return tokeniser.batch_decode(dataset, skip_special_tokens=True)


def batch_reduce(batch, func, reduce_fn, init, step=16):
    """
    Function signature: Tensor[B, L] -> func:(Tensor[B', L] -> A) -> reduce_fn:(B -> A -> B) -> init:B' -> steps:int -> B
    """
    result = init
    for i in range(0, len(batch), step):
        sub_batch = batch[i : min(i + step, len(batch))]
        sub_result = func(sub_batch)
        result = reduce_fn(result, sub_result)
    return result


@torch.no_grad()
def compute_generative_perplexity(
    text_samples, max_length: int = 1024, retokenize: bool = True, 
    input_is_tokenized: bool = False, tokenizer=None, model_type="llama"
) -> None:
    # load the specified model based on model_type
    if model_type == "llama":
        eval_model = LlamaForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16,
        ).eval()
        model_path = llama_model_path
    elif model_type == "gpt2-xl":
        eval_model = GPT2LMHeadModel.from_pretrained(
            gpt2_model_path,
            torch_dtype=torch.float16,
        ).eval()
        model_path = gpt2_model_path
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    eval_model = eval_model.to("cuda")
    
    if tokenizer is None:
        eval_model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        eval_model_tokenizer.pad_token = eval_model_tokenizer.eos_token
    else:
        eval_model_tokenizer = tokenizer

    # tokenize the batch or use pre-tokenized input
    if input_is_tokenized:
        # If input is already token IDs, create the tensor and pad if necessary
        max_len = max(len(seq) for seq in text_samples)
        padded_max_len = min(max_len, max_length)
        input_ids = torch.ones((len(text_samples), padded_max_len), 
                              dtype=torch.long) * eval_model_tokenizer.pad_token_id
        
        for i, seq in enumerate(text_samples):
            seq_len = min(len(seq), padded_max_len)
            input_ids[i, :seq_len] = torch.tensor(seq[:seq_len])
        
        input_ids = input_ids.to(eval_model.device)

        print(input_ids)
    else:
        # tokenize the text samples
        tokenized = eval_model_tokenizer(
            text_samples,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(eval_model.device)
        input_ids = tokenized["input_ids"]

    eos_token_id = eval_model_tokenizer.eos_token_id
    eos_mask = input_ids == eos_token_id
    first_eos = eos_mask.cumsum(dim=-1) == 1

    # generative perplexity
    with autocast(), torch.no_grad():
        outputs = eval_model(input_ids)
        logits = outputs.logits

    logits = logits.transpose(
        -1, -2
    )  # size b X D X N, D = the number of possible tokens
    nlls = F.cross_entropy(logits[..., :-1], input_ids[..., 1:], reduction="none")
    effective_mask = (first_eos[..., 1:] + (input_ids[..., 1:] != eos_token_id)).bool()
    nlls = nlls * effective_mask

    # compute per-sample perplexity
    likelihood_list = []
    for b in range(input_ids.size(0)):
        nll = nlls[b]
        mask = effective_mask[b]
        likelihood = nll.sum() / mask.sum()
        likelihood_list.append(likelihood.exp().item())

    return likelihood_list


def compute_entropy(samples: list, model_name: str = llama_model_path, 
                    input_is_tokenized: bool = False, tokenizer=None):
    """
    Compute the entropy of each text sample using subword tokens.
    Can accept either text samples or pre-tokenized token IDs.
    """
    # initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # use provided token IDs or encode text samples
    if input_is_tokenized:
        token_id_seqs = samples
    else:
        # encode each sample into subword IDs (no special tokens)
        token_id_seqs = [
            tokenizer.encode(sample, add_special_tokens=False) for sample in samples
        ]
    
    # compute per-sample entropy
    entropies = []
    for seq in token_id_seqs:
        counts = Counter(seq)
        total = sum(counts.values())
        entropy = (
            -sum((cnt / total) * math.log(cnt / total, 2) for cnt in counts.values())
            if total > 0
            else 0.0
        )
        entropies.append(entropy)
    return entropies

def compute_mauve_score(candidate_samples, reference_samples):
    import mauve 
    score = mauve.compute_mauve(p_text=candidate_samples, q_text=reference_samples, device_id=0, max_text_length=1024, verbose=False)
    return score.mauve

def main():
    parser = argparse.ArgumentParser(
        description="Compute average entropy, generative perplexity, and mauve score for a list of text samples."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        help="Path to a JSON file containing a list of strings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for computing generative perplexity",
    )
    parser.add_argument(
        "--length-plot-output",
        type=str,
        default="length_distribution.png",
        help="Output path for the sentence length distribution plot",
    )
    parser.add_argument(
        "--perplexity-plot-output",
        type=str,
        default=None,  # Will be derived from length-plot-output
        help="Output path for the perplexity vs length scatter plot",
    )
    parser.add_argument(
        "--results-output",
        type=str,
        default=None,
        help="Path to JSON file to save computed metrics",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["sentence", "chunk"],
        default="sentence",
        help="sentence: eval each input as one; chunk: tokenize & split into 1024‐length segments",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "gpt2-large"],
        default="llama",
        help="Model to use for generative perplexity evaluation",
    )
    # New flags to control metric evaluation (default false)
    parser.add_argument("--entropy", action="store_true", default=False, help="Evaluate entropy")
    parser.add_argument("--perplexity", action="store_true", default=False, help="Evaluate generative perplexity")
    parser.add_argument("--mauve", action="store_true", default=False, help="Evaluate mauve score")
    parser.add_argument("--reference-perplexity", action="store_true", default=False, help="Evaluate reference text perplexity")
    args = parser.parse_args()

    # Derive perplexity plot path from length plot path if not specified
    if args.perplexity_plot_output is None:
        base, ext = os.path.splitext(args.length_plot_output)
        args.perplexity_plot_output = f"{base}_perplexity{ext}"

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # choose sentence‐level or chunk‐level inputs
    if args.eval_mode == "chunk":
        # pre‐load tokenizer based on model type
        model_path = llama_model_path if args.model_type == "llama" else gpt2_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        chunk_size = 1024
        
        # Tokenize all samples
        token_id_seqs = [tokenizer.encode(s, add_special_tokens=False) for s in samples]
        
        # Concatenate all sentences with EOS tokens between them
        concatenated_tokens = []
        for seq in token_id_seqs:
            concatenated_tokens.extend(seq)
            concatenated_tokens.append(tokenizer.eos_token_id)  # Add EOS between sentences
        
        # Truncate concatenated_tokens to be a multiple of chunk_size
        truncated_length = (len(concatenated_tokens) // chunk_size) * chunk_size
        concatenated_tokens = concatenated_tokens[:truncated_length]
        
        # Split the concatenated tokens into chunks of size chunk_size
        chunks = []
        for i in range(0, len(concatenated_tokens), chunk_size):
            chunks.append(concatenated_tokens[i:i + chunk_size])
        
        # Keep chunks as token IDs for direct use
        target_samples = chunks
        use_tokenized_input = True
    else:
        target_samples = samples
        use_tokenized_input = False

    # Conditionally compute entropy
    if args.entropy:
        entropy_list = compute_entropy(
            target_samples, 
            input_is_tokenized=use_tokenized_input,
            tokenizer=tokenizer if use_tokenized_input else None
        )
        avg_entropy = sum(entropy_list) / len(entropy_list)
        print(f"Average entropy: {avg_entropy:.4f}")
    else:
        avg_entropy = None
        print("Entropy evaluation skipped")
    
    # Conditionally compute generative perplexity
    if args.perplexity:
        all_perps = batch_reduce(
            target_samples,
            lambda batch: compute_generative_perplexity(
                batch, 
                input_is_tokenized=use_tokenized_input,
                tokenizer=tokenizer if use_tokenized_input else None,
                model_type=args.model_type
            ),
            lambda acc, res: acc + res,
            init=[],
            step=args.batch_size,
        )
        avg_perp = sum(all_perps) / len(all_perps)
        print(f"Average generative perplexity: {avg_perp:.4f}")
    else:
        avg_perp = None
        all_perps = None
        print("Generative perplexity evaluation skipped")
    
    # Conditionally compute reference text perplexity
    if args.reference_perplexity:
        print("Computing reference text perplexity...")
        reference_samples = get_reference_text_dataset()
        reference_perps = batch_reduce(
            reference_samples,
            lambda batch: compute_generative_perplexity(
                batch, 
                input_is_tokenized=False,
                tokenizer=None,
                model_type=args.model_type
            ),
            lambda acc, res: acc + res,
            init=[],
            step=args.batch_size,
        )
        avg_reference_perp = sum(reference_perps) / len(reference_perps)
        print(f"Average reference perplexity: {avg_reference_perp:.4f}")
    else:
        avg_reference_perp = None
        reference_perps = None
        reference_samples = None
        print("Reference perplexity evaluation skipped")
    
    # Conditionally compute mauve score
    if args.mauve:
        if reference_samples is None:
            reference_samples = get_reference_text_dataset()
        mauve_score = compute_mauve_score(samples, reference_samples)
        print(f"Mauve score: {mauve_score:.4f}")
    else:
        mauve_score = None
        print("Mauve evaluation skipped")
    
    # Calculate lengths early for use in filtered perplexities
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
    lengths = [len(gpt2_tokenizer.encode(s, add_special_tokens=False)) for s in samples]
    
    # Conditionally create perplexity vs. tokenized length plot when perplexity is evaluated
    filtered_perplexities = None
    reference_filtered_perplexities = None
    
    if args.perplexity and args.eval_mode == "sentence":
        idx = []
        val = []
        for i in range(0, 1024):
            _val = []
            for l, perp in zip(lengths, all_perps):
                if l >= i:
                    _val.append(perp)
            idx.append(i)
            val.append(sum(_val) / len(_val) if _val else 0)
        
        # Store filtered perplexities for JSON output
        filtered_perplexities = {
            "token_thresholds": idx,
            "avg_perplexities": val
        }
        
        plt.figure(figsize=(12, 6))
        
        # Plot candidate samples
        plt.scatter(idx, val, alpha=0.6, color="blue", label="Candidate samples")
        
        # Plot reference samples if available
        if args.reference_perplexity and reference_samples is not None:
            reference_lengths = [len(gpt2_tokenizer.encode(s, add_special_tokens=False)) for s in reference_samples]
            ref_idx = []
            ref_val = []
            for i in range(0, 1024):
                _ref_val = []
                for l, perp in zip(reference_lengths, reference_perps):
                    if l >= i:
                        _ref_val.append(perp)
                ref_idx.append(i)
                ref_val.append(sum(_ref_val) / len(_ref_val) if _ref_val else 0)
            
            # Store reference filtered perplexities for JSON output
            reference_filtered_perplexities = {
                "token_thresholds": ref_idx,
                "avg_perplexities": ref_val
            }
            
            plt.scatter(ref_idx, ref_val, alpha=0.6, color="red", label="Reference samples")
        
        # Add horizontal lines for specific token lengths
        for tlen in [10, 20, 30, 40, 50, 75, 100]:
            if tlen < len(val):
                plt.axhline(y=val[tlen], linestyle='--', color='blue', alpha=0.3)
        
        plt.title("Perplexity vs. Tokenized Length")
        plt.xlabel("Number of tokens")
        plt.ylabel("Log Perplexity")
        plt.legend()
        ax = plt.gca()
        ticks = list(ax.get_yticks())
        for tlen in [10, 20, 30, 40, 50, 75, 100]:
            if tlen < len(val):
                tick_value = val[tlen]
                if tick_value not in ticks:
                    ticks.append(tick_value)
        ax.set_yticks(sorted(ticks))
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        plt.yscale("log")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.perplexity_plot_output)
        print(f"Saved perplexity vs. length scatter plot to {args.perplexity_plot_output}")
    elif args.eval_mode == "sentence":
        print("Perplexity plot skipped because the --perplexity flag was not provided")

    if args.results_output:
        results = {
            "avg_entropy": avg_entropy, 
            "avg_perplexity": avg_perp, 
            "avg_reference_perplexity": avg_reference_perp,
            "mauve_score": mauve_score,
            "filtered_perplexities": filtered_perplexities,
            "reference_filtered_perplexities": reference_filtered_perplexities
        }
        with open(args.results_output, "w", encoding="utf-8") as outf:
            json.dump(results, outf, indent=2)
        print(f"Saved metrics to {args.results_output}")
    
    # plot cumulative distribution of GPT2‐tokenized sentence lengths
    # Create cumulative distribution
    sorted_lengths = sorted(lengths)
    cumulative_percentages = [i / len(sorted_lengths) * 100 for i in range(1, len(sorted_lengths) + 1)]
    
    # Save length data to JSON file
    length_data = {
        "lengths": lengths,
        "sorted_lengths": sorted_lengths,
        "cumulative_percentages": cumulative_percentages,
        "num_samples": len(samples)
    }
    base, ext = os.path.splitext(args.length_plot_output)
    length_data_output = f"{base}.json"
    with open(length_data_output, "w", encoding="utf-8") as f:
        json.dump(length_data, f, indent=2)
    print(f"Saved length distribution data to {length_data_output}")
    
    plt.figure()
    plt.plot(sorted_lengths, cumulative_percentages, color="skyblue", linewidth=2)
    plt.title("Tokenized Sentence Length Cumulative Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Cumulative percentage (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.length_plot_output)

    if args.eval_mode == "chunk":
        print(f"Evaluated in chunk mode over {len(target_samples)} segments (using pre-tokenized input)")
    else:
        print(f"Evaluated in sentence mode over {len(target_samples)} samples")
        if args.perplexity:
            print(f"Saved perplexity vs. length scatter plot to {args.perplexity_plot_output}")

if __name__ == "__main__":
    main()