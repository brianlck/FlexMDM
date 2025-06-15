import argparse
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.text import setup_tokeniser, get_text_dataset
from tqdm import tqdm

def collect_length_data(dataset, pad_id: int, batch_size:int) -> List[int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    lengths = []
    for batch in tqdm(loader, desc="Scanning dataset", leave=False):
        ids = batch["input_ids"]
        lens = (ids != pad_id).sum(dim=1).cpu().tolist()
        lengths.extend(lens)
    return lengths

def plot_histogram(lengths: List[int], bins: int):
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=bins, edgecolor="steelblue", alpha=0.85)
    plt.xlabel("Sequence Length")
    plt.title(f"Length distribution of wikitext2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"length_distribution_wikitext2.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    # --- dataset setup ---
    tokeniser = setup_tokeniser()
    pad_id = tokeniser.pad_token_id
    dataset = get_text_dataset(
        "wikitext2",
        split = args.split,
        max_length = args.max_length,
    )

    lengths = collect_length_data(dataset, pad_id, args.batch_size)
    plot_histogram(lengths, bins=50)

if __name__ == "__main__":
    main()