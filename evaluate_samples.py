from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import math
from collections import Counter
import argparse
from torch.cuda.amp import autocast
import torch.nn.functional as F

import json
import matplotlib.pyplot as plt

llama_model_path = "meta-llama/Llama-2-7b-hf"


def batch_reduce(batch, func, reduce_fn, init, step=16):
    """
    Function signature: Tensor[B, L] -> func:(Tensor[B', L] -> A) -> reduce_fn:(B -> A -> B) -> init:B' -> steps:int -> B
    """
    assert len(batch) % step == 0, "Batch size must be divisible by step size."
    result = init
    for i in range(0, len(batch), step):
        sub_batch = batch[i : i + step]
        sub_result = func(sub_batch)
        result = reduce_fn(result, sub_result)
    return result


@torch.no_grad()
def compute_generative_perplexity(
    text_samples, max_length: int = 1024, retokenize: bool = True
) -> None:
    # load a pretrained llama 7B
    eval_model = LlamaForCausalLM.from_pretrained(
        llama_model_path,
        torch_dtype=torch.float16,
    ).eval()
    eval_model = eval_model.to("cuda")
    eval_model_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    eval_model_tokenizer.pad_token = eval_model_tokenizer.eos_token

    # tokenize the batch
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


def compute_entropy(samples: list[str], model_name: str = llama_model_path):
    """
    Compute the entropy of each text sample using subword tokens.
    """
    # initialize the same tokenizer used for perplexity
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute average entropy and generative perplexity for a list of text samples."
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
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # choose sentence‐level or chunk‐level inputs
    if args.eval_mode == "chunk":
        # pre‐load tokenizer to split and decode
        tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        chunk_size = 1024
        token_id_seqs = [tokenizer.encode(s, add_special_tokens=False) for s in samples]
        chunks: list[list[int]] = []
        for seq in token_id_seqs:
            for i in range(0, len(seq), chunk_size):
                chunks.append(seq[i : i + chunk_size])
        # back to text for existing metrics API
        target_samples = [
            tokenizer.decode(ids, clean_up_tokenization_spaces=True) for ids in chunks
        ]
    else:
        target_samples = samples

    # compute metrics on target_samples
    entropy_list = compute_entropy(target_samples)
    avg_entropy = sum(entropy_list) / len(entropy_list)

    all_perps = batch_reduce(
        target_samples,
        compute_generative_perplexity,
        lambda acc, res: acc + res,
        init=[],
        step=args.batch_size,
    )
    avg_perp = sum(all_perps) / len(all_perps)

    print(f"Average entropy: {avg_entropy:.4f}")
    print(f"Average generative perplexity: {avg_perp:.4f}")

    if args.results_output:
        results = {"avg_entropy": avg_entropy, "avg_perplexity": avg_perp}
        with open(args.results_output, "w", encoding="utf-8") as outf:
            json.dump(results, outf, indent=2)
        print(f"Saved metrics to {args.results_output}")

    # plot distribution of GPT2‐tokenized sentence lengths
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lengths = [len(gpt2_tokenizer.encode(s, add_special_tokens=False)) for s in samples]
    plt.figure()
    plt.hist(lengths, bins=50, color="skyblue", edgecolor="black")
    plt.title("GPT2 Tokenized Sentence Length Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(args.length_plot_output)
    print(f"Saved sentence length distribution plot to {args.length_plot_output}")

    if args.eval_mode == "chunk":
        print(f"Evaluated in chunk mode over {len(target_samples)} segments")
    else:
        print(f"Evaluated in sentence mode over {len(target_samples)} samples")


if __name__ == "__main__":
    main()
