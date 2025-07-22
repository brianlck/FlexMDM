import json, sys, torch
import sys, json
from pathlib import Path
from datasets import load_dataset
import torch
from generate import generate
from transformers import AutoModel, AutoTokenizer
import random, os
import numpy as np
import torch.distributed as dist
from tqdm import tqdm

# ------------------------------------------------------------
# most of the functions adapted from eval.py
# ------------------------------------------------------------

STOP_TOKENS = ["\nclass", "\n@", "\n#", "\nprint",
               "\nif __name__ == \"__main__\":", "\n\n"]

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def humaneval_ckpt(model, tokenizer, num_iterations: int):
    """
    Evaluate the humaneval peformance for a given model
    num_iterations: training iterations for the model
    """
    model.eval()
    sample_kwargs = {
        "steps": 512,
        "gen_length": 512,
        "block_length": 512,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336
    }
    ds = load_dataset("openai/openai_humaneval" , split="test")
    out_file = Path(f"results/test_baseline_llada.jsonl").open("w")

    for elem in tqdm(ds):
        task_id, prompt_text = elem["task_id"], elem["prompt"]
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device).input_ids

        # generate the code
        with torch.no_grad():
            out = generate(model, prompt_ids, tokenizer, **sample_kwargs)

        # post-process the generated code
        for seq in out:
            code = tokenizer.decode(seq[prompt_ids.shape[1]:], skip_special_tokens=False)

            for tok in STOP_TOKENS:
                if tok in code:
                    code = code.split(tok)[0]
                    break
            
            clean_code = tokenizer.decode(tokenizer(code, add_special_tokens=False)["input_ids"], 
            skip_special_tokens=True
                )

            print("raw prompt: ", prompt_text)
            print("generated code: ", code)
            print("cleaned code: ", clean_code)
            print("--------------------------------")
            out_file.write(json.dumps({"task_id": task_id, "completion": clean_code}) + "\n")

    out_file.close()


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        return_dict=True,
    ).to(local_rank)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", padding_side="right", trust_remote_code=True, use_fast=True)
    print("Tokenizer and backbone model loaded")
    humaneval_ckpt(model, tokenizer, num_iterations=0)
    cleanup_ddp()






