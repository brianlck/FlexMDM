import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from safetensors.torch import load_file
from peft import LoraConfig, TaskType, get_peft_model
from pathlib import Path

from generate import generate
from generate_ours import generate_ours
from gsm8k import GSM8KDataset

from sft_train import get_reasoning_tokenizer
from llada_dit import LLaDA_DIT


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
}

def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


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


def uncond_generate(model, tokenizer, args):
    model.eval()
    wall_times = []
    all_generations = []
    device = model.device

    num_samples = 100
    for _ in range(num_samples):
        start_time = time.time()
        # for unconditional generation, we set prompt to None

        if args.variable_length:
            out = generate_ours(
                model,
                None,
                tokenizer,
                steps = args.diffusion_steps,
                beta = 0.5,
                model_type = "variable",
                temperature = args.temperature,
                remasking = args.remasking,
            )
        else:
            out = generate_ours(
                model,
                None,
                tokenizer,
                steps = args.diffusion_steps,
                beta = 0.0,
                model_type = "mdm",
                temperature = args.temperature,
                remasking = args.remasking,
            )
        generated_texts = tokenizer.batch_decode(out[:, :], skip_special_tokens=True)
        print(generated_texts)
        print("-" * 50)
        all_generations.append(generated_texts)
        wall_times.append(time.time() - start_time)
    
    avg_wall_time = sum(wall_times) / len(wall_times)
    return avg_wall_time, all_generations


def evaluate(
    model,
    tokenizer,
    dataloader,
    args,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=False, desc=f"Rank {dist.get_rank()}", position=dist.get_rank()):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        print(prompts)

        tokenised_prompts = tokenizer(prompts, padding="max_length", max_length=args.gen_length, return_tensors="pt").input_ids.to(device)

        if args.variable_length:
            out = generate_ours(
                model, 
                tokenised_prompts, 
                tokenizer, 
                steps = args.diffusion_steps, 
                beta = 0.5,
                model_type = "variable",
                temperature = args.temperature,
                remasking = args.remasking,
            )
            generated_texts = tokenizer.batch_decode(out[:, :], skip_special_tokens=True)
        else:
            out = generate_ours(
                model, 
                tokenised_prompts,
                tokenizer,
                steps = 256,
                beta = 0.0,
                model_type = "mdm",
                temperature = args.temperature,
                remasking = args.remasking,
            )
            generated_texts = tokenizer.batch_decode(out[:, :], skip_special_tokens=False)
            # out = generat(
            #     model,
            #     input_ids,
            #     tokenizer,
            #     steps=1024,
            #     gen_length=args.gen_length,
            #     block_length=args.block_length,
            #     temperature=args.temperature,
            #     cfg_scale=0.0,
            #     remasking=args.remasking,
            # )
            # generated_texts = tokenizer.batch_decode(out[:, -args.gen_length:], skip_special_tokens=False)
        
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--variable_length", action="store_true")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    # model path: for the test run
    parser.add_argument("--checkpoint_path", type=str, default="/n/netscratch/albergo_lab/Lab/sft-checkpoints/llada-sft-openwebtext/checkpoint-40000")

    # sampler setup
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=1024)
    parser.add_argument("--diffusion_steps", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--remasking", type=str, default="random")


    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/mdm_test")
    parser.add_argument("--dont_use_box", action="store_true")
    args = parser.parse_args()
    num_evals = {"gsm8k": -1}

    # backbone model load
    backbone = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).to(local_rank)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", padding_side="right", trust_remote_code=True, use_fast=True)
    print("Tokenizer and backbone model loaded")


    if args.variable_length:
        lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "transformer.ff_out"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM)

        backbone = get_peft_model(backbone, lora_config)
        model = LLaDA_DIT(backbone, pad_token_id = tokenizer.pad_token_id, d_model = 4096)

        ckpt_dir = Path(args.checkpoint_path)
        state = torch.load(ckpt_dir/ "pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        model = model.to(device = local_rank, dtype = torch.bfloat16).to(local_rank)
        print("Variable length model loaded")
    else:
        # fine-tuned model load
        if args.checkpoint_path:
            model = PeftModel.from_pretrained(backbone, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
                local_rank
            )
            if dist.get_world_size() > 1:
                dist.barrier()  # Make sure all processes are ready
                for param in model.parameters():
                    dist.broadcast(param.data, src=0)
                print(f"Rank {local_rank}: Parameters synchronized")
        else:
            model = backbone.to(device = local_rank, dtype = torch.bfloat16).to(local_rank)
        print("Baseline model loaded")



    if args.unconditional:
        avg_wall_time, all_generations = uncond_generate(model, tokenizer, args)
        print(f"Average wall time: {avg_wall_time}")
        print(f"All generations: {all_generations}")
    
    else:
        dataset = DATASET_MAP[args.dataset](
            tokenizer,
            subsample=num_evals[args.dataset],
            num_examples=args.few_shot,
            add_reasoning=True,  # prefill for all models
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=CustomDistributedSampler(dataset, shuffle=False),
            collate_fn=dataset.collate_fn,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        if args.variable_length:
            filename = f"{args.output_dir}/{args.diffusion_steps}_ours_{args.remasking}_{args.beta}_{dist.get_rank()}_generations.json"
        else:
            filename = f"{args.output_dir}/mdm_{args.remasking}_{args.gen_length}_{args.block_length}_{dist.get_rank()}_generations.json"
        print(f"Saving generations to {filename}")

        metrics = evaluate(model, tokenizer, dataloader, args)

        if not args.dont_save:
            with open(filename, "w") as f:
                json.dump(
                    {
                        "generations": metrics["generations"],
                        "metrics": {
                            "wall_time": metrics["wall_time"],
                            "total_processed": metrics["total_processed"],
                        },
                        "checkpoint_path": args.checkpoint_path,
                        "gen_length": args.gen_length,
                        "diffusion_steps": args.diffusion_steps,
                        "block_length": args.block_length,
                    },
                    f,
                    indent=2,
                )

    cleanup_ddp()
