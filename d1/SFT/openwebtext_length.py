import json, pathlib, numpy as np, matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer

cache_dir = pathlib.Path("/n/netscratch/sham_lab/Everyone/jay_brian/openwebtext_cache_train")
sample_frac = 0.1
bins = 50

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", padding_side="right", trust_remote_code=True, use_fast=True)

ds = load_from_disk(cache_dir)
data = ds.train_test_split(test_size=0.1)

# see if the training samples are sane
print(tokenizer.decode(data["train"][9]["input_ids"]))



# get the length of the text
# lengths = np.fromiter(ds["length"], dtype=np.int32) 

# plt.hist(lengths, bins=bins)
# plt.savefig("openwebtext_length_distribution.png")







