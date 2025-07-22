from datasets import load_dataset
from transformers import GPT2TokenizerFast
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
LOG_INTERVAL = 1_000_000
BATCH_SIZE = 1000
SAVE_DIR = "length_dist_plots-2"

# Create output directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Load full dataset (cached)
dataset = load_dataset("openwebtext", split="train", num_proc=64)
MAX_EXAMPLES = len(dataset)

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Initialize counter and length list
counter = Counter()
length_list = []

# Process in batches
for start in range(0, MAX_EXAMPLES, BATCH_SIZE):
    end = min(start + BATCH_SIZE, MAX_EXAMPLES)
    batch = dataset[start:end]["text"]
    encodings = tokenizer(batch, truncation=False, add_special_tokens=False)
    lengths = [len(ids) for ids in encodings["input_ids"]]
    counter.update(lengths)
    length_list.extend(lengths)

    # Save plot every 1M
    if (end % LOG_INTERVAL == 0) or (end == MAX_EXAMPLES):
        count_millions = end // 1_000_000
        x, y = zip(*sorted(counter.items()))

        # Compute percentiles for current subset
        lengths_np = np.array(length_list)
        results = {
            p: int(np.percentile(lengths_np, p))
            for p in [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        }

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(x, y, color="skyblue")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency (log scale)")
        plt.title(f"Token Length Distribution (Up to {end:,} Examples)")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Annotate percentiles
        for p in [50, 60, 70, 80, 90, 95, 99]:
            val = results[p]
            plt.axvline(val, color="red", linestyle="--", linewidth=1.5)
            plt.text(
                val + 10,
                max(y) / 10,
                f"{p}%",
                rotation=90,
                color="red",
                fontsize=10,
                verticalalignment="center",
            )

        plt.tight_layout()
        filename = os.path.join(SAVE_DIR, f"length_dist_{count_millions}M.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved plot: {filename}")

# --- Final Percentiles ---
print("\n📊 Computing Final Percentiles...")

lengths_np = np.array(length_list)
all_percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
final_results = {p: int(np.percentile(lengths_np, p)) for p in all_percentiles}

print("\n📊 Token Length Percentiles:")
for p in all_percentiles:
    print(f"  {p:>3}%: {final_results[p]:,} tokens")

# --- Save final plot with annotations ---
x, y = zip(*sorted(counter.items()))
plt.figure(figsize=(12, 6))
plt.bar(x, y, width=5, color="skyblue", edgecolor="black")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Token Length")
plt.ylabel("Frequency (log scale)")
plt.title("Final Token Length Distribution (4.096M Samples)")
plt.grid(True, linestyle="--", alpha=0.5)

for p in [50, 95, 99]:
    val = final_results[p]
    plt.axvline(val, color="red", linestyle="--", linewidth=1.5)
    plt.text(
        val + 10,
        max(y) / 10,
        f"{p}%",
        rotation=90,
        color="red",
        fontsize=10,
        verticalalignment="center",
    )

plt.tight_layout()
final_path = os.path.join(SAVE_DIR, "length_dist_final_annotated.png")
plt.savefig(final_path)
plt.close()
print(f"\n✅ Final annotated plot saved: {final_path}")
