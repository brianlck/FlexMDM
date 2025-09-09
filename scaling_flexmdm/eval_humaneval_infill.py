import argparse
import json
import math
import os
import random
import time
import pickle
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from pathlib import Path

from flexmdm_inference import generate_flexmdm_infilling
from llada_inference import generate_mdm
from instruction_finetuning import get_reasoning_tokenizer
from llada_dit import LLaDA_DIT
from human_eval_infilling.data import write_jsonl, read_problems


class HumanEvalInfillingDataset(Dataset):
    """Dataset for HumanEval infilling tasks with prefix and suffix."""
    
    def __init__(self, tokenizer, variable_length: bool = True, data_path: str = None, subsample: int = -1):
        self.tokenizer = tokenizer
        self.data = []

        problems = read_problems(benchmark_name="single-line")

        if variable_length:
            for task_id in problems:
                prompt = problems[task_id]['prompt'] + "<infill-boundary>"
                suffix = "<infill-boundary>" + problems[task_id]['suffix']
                self.data.append({
                    'task_id': task_id,
                    'prompt': prompt,
                    'suffix': suffix,
                    'ground_truth': problems[task_id]['canonical_solution']
                })
        else:
            for task_id in problems:
                instruction = "You are instructed to perform an infill task. Please fill in the missing parts of the code. The prefix and suffix are provided and delimited by <prefix> </prefix> and <suffix> </suffix> tokens."
                prompt = instruction + "<prefix>" + problems[task_id]['prompt'] + "</prefix><suffix>" + problems[task_id]['suffix'] + "</suffix>"
                self.data.append({
                    'task_id': task_id,
                    'prompt': prompt,
                    'suffix': "",
                    'ground_truth': problems[task_id]['canonical_solution']
                })    
        if subsample > 0:
            self.data = self.data[:subsample]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        task_ids = [item['task_id'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        suffixes = [item['suffix'] for item in batch]
        ground_truths = [item['ground_truth'] for item in batch]
        
        return {
            'task_ids': task_ids,
            'prompts': prompts,
            'suffixes': suffixes,
            'ground_truths': ground_truths
        }


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


def save_progress(filename, all_generations, processed_count):
    """Save progress to allow resumption after preemption"""
    progress_file = filename.replace('.json', '_progress.pkl')
    progress_data = {
        'all_generations': all_generations,
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)
    if dist.get_rank() == 0:
        print(f"Progress saved: {processed_count} samples processed")


def load_progress(filename):
    """Load previous progress if available"""
    progress_file = filename.replace('.json', '_progress.pkl')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
            if dist.get_rank() == 0:
                print(f"Resuming from progress: {progress_data['processed_count']} samples already processed")
                print(f"Previous run timestamp: {progress_data['timestamp']}")
            return progress_data['all_generations'], progress_data['processed_count']
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Error loading progress file: {e}")
            return [], 0
    return [], 0


def evaluate_infilling_mdm(
    model,
    tokenizer,
    dataloader,
    args,
    filename,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    device = model.device
    
    # Load previous progress
    all_generations, processed_count = load_progress(filename)
    
    # Skip already processed samples
    samples_to_skip = processed_count
    current_sample = 0
    
    save_interval = 10  # Save progress every 10 batches
    batch_count = 0

    for batch in tqdm(dataloader, disable=False, desc=f"Rank {dist.get_rank()}", position=dist.get_rank()):
        # Skip already processed batches
        batch_size = len(batch["task_ids"])
        if current_sample + batch_size <= samples_to_skip:
            current_sample += batch_size
            continue
        
        start_time = time.time()
        task_ids = batch["task_ids"]
        prompts = batch["prompts"]
        ground_truths = batch["ground_truths"]

        if dist.get_rank() == 0:
            print(f"Processing batch {batch_count + 1}, samples {current_sample}-{current_sample + batch_size}")

        # For MDM: prompts are already formatted with instruction and delimiters
        tokenized_prompts = tokenizer(
            prompts,
            max_length=args.gen_length,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)

        print(tokenized_prompts.shape)

        out = generate_mdm(
            model,
            tokenized_prompts,
            tokenizer,
            args.diffusion_steps,
            args.gen_length,
            args.gen_length,
            args.temperature,
            cfg_scale=0.0,  # No classifier-free guidance for MDM
            remasking=args.remasking
        )
        
        generated_texts = tokenizer.batch_decode(out[:, :], skip_special_tokens=True)
        
        example_result = [
            {
                "task_id": task_ids[j],
                "prompt": prompts[j],
                "generated_infill": generated_texts[j],
                "ground_truth": ground_truths[j],
            }
            for j in range(len(task_ids))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        current_sample += batch_size
        wall_times.append(time.time() - start_time)
        batch_count += 1

        # Periodic progress saving
        if batch_count % save_interval == 0:
            save_progress(filename, all_generations, current_sample)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(task_ids) - 1)
            print(f"Task ID: {task_ids[idx]}")
            print(f"Full prompt: {prompts[idx]}")
            print("-" * 30)
            print("Generated infill:")
            print(generated_texts[idx])
            print("-" * 30)
            print(f"Ground truth: {ground_truths[idx]}")
            print("=" * 50)

    # Final progress save
    save_progress(filename, all_generations, current_sample)
    
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def evaluate_infilling_flexmdm(
    model,
    tokenizer,
    dataloader,
    args,
    filename,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    device = model.device
    
    # Load previous progress
    all_generations, processed_count = load_progress(filename)
    
    # Skip already processed samples
    samples_to_skip = processed_count
    current_sample = 0
    
    save_interval = 10  # Save progress every 10 batches
    batch_count = 0

    for batch in tqdm(dataloader, disable=False, desc=f"Rank {dist.get_rank()}", position=dist.get_rank()):
        # Skip already processed batches
        batch_size = len(batch["task_ids"])
        if current_sample + batch_size <= samples_to_skip:
            current_sample += batch_size
            continue
        
        start_time = time.time()
        task_ids = batch["task_ids"]
        prompts = batch["prompts"]
        suffixes = batch["suffixes"]
        ground_truths = batch["ground_truths"]

        if dist.get_rank() == 0:
            print(f"Processing batch {batch_count + 1}, samples {current_sample}-{current_sample + batch_size}")

        # Tokenize prefixes and suffixes
        tokenized_prefixes = tokenizer(
            prompts, 
            padding="max_length", 
            max_length=args.gen_length,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)
        
        tokenized_suffixes = tokenizer(
            suffixes, 
            padding="max_length", 
            max_length=args.gen_length,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)

        # Generate infilled content
        out, trace = generate_flexmdm_infilling(
            model,
            tokenized_prefixes,
            tokenized_suffixes,
            tokenizer,
            steps=args.diffusion_steps,
            model_type="variable",
            temperature=args.temperature,
            remasking=args.remasking,
            alpha=args.alpha,
            max_window=args.max_window,
            confidence_method=args.confidence_method,
            use_sliding_window=args.use_sliding_window,
        )
        
        generated_texts = tokenizer.batch_decode(out[:, :], skip_special_tokens=True)
        
        example_result = [
            {
                "task_id": task_ids[j],
                "prompt": prompts[j],
                "suffix": suffixes[j],
                "generated_infill": generated_texts[j],
                "ground_truth": ground_truths[j],
            }
            for j in range(len(task_ids))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        current_sample += batch_size
        wall_times.append(time.time() - start_time)
        batch_count += 1

        # Periodic progress saving
        if batch_count % save_interval == 0:
            save_progress(filename, all_generations, current_sample)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(task_ids) - 1)
            print(f"Task ID: {task_ids[idx]}")
            print("-" * 30)
            print("Generated infill:")
            print(generated_texts[idx])
            print("-" * 30)
            print(f"Suffix: {suffixes[idx]}")
            print("-" * 30)
            print(f"Ground truth: {ground_truths[idx]}")
            print("=" * 50)

    # Final progress save
    save_progress(filename, all_generations, current_sample)
    
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """Custom distributed sampler that doesn't add padding indices."""

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
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=None, help="Path to HumanEval infilling dataset")
    parser.add_argument("--subsample", type=int, default=-1, help="Number of samples to evaluate (-1 for all)")
    
    # Model path
    parser.add_argument("--checkpoint_path", type=str, default="/n/netscratch/albergo_lab/Lab/sft-checkpoints/llada-sft-openwebtext/checkpoint-40000")

    # Model type
    parser.add_argument("--variable_length", action="store_true", help="whether to use FlexMDM or MDM")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the inference")

    # FlexMDM sampling configs-only valid for FlexMDM
    parser.add_argument("--confidence_method", type=str, default="prob_diff", choices=["position", "prob_diff", "top_prob", "entropy"])
    parser.add_argument("--use_sliding_window", action="store_true", help="Use sliding window for confidence calculation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for the sliding window")
    parser.add_argument("--max_window", type=int, default=10, help="Maximum window size for the sliding window")

    # MDM sampling configs-only valid for MDM
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--diffusion_steps", type=int, default=1024)
    parser.add_argument("--remasking", type=str, default="random")

    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/human_eval_infilling")
    args = parser.parse_args()

    
    # Log all arguments
    if dist.get_rank() == 0:
        print("=" * 50)
        print("HUMAN EVAL INFILLING ARGUMENTS:")
        print("=" * 50)
        for arg_name, arg_value in vars(args).items():
            print(f"{arg_name}: {arg_value}")
        print("=" * 50)

    # Load backbone model
    backbone = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).to(local_rank)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", padding_side="right", trust_remote_code=True, use_fast=True)
    print("Tokenizer and backbone model loaded")
    
    if args.variable_length:
        # instruction finetuned FlexMDM load--same configurations as in the training script
        lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "transformer.ff_out"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(backbone, lora_config)
        model = LLaDA_DIT(model, pad_token_id=tokenizer.pad_token_id, d_model=4096)

        ckpt_dir = Path(args.checkpoint_path)
        state = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        model = model.to(device=local_rank, dtype=torch.bfloat16).to(local_rank)
        print("FlexMDM loaded")
    else:
        # fine-tuned LLaDA model load--same configurations as in the training script
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
        print("LLaDA model loaded")


    # Create dataset and dataloader
    dataset = HumanEvalInfillingDataset(
        tokenizer,
        variable_length=args.variable_length,
        data_path=args.data_path,
        subsample=args.subsample
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/infill_{args.diffusion_steps}_alpha_{args.alpha}_max_window_{args.max_window}_{args.confidence_method}_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    if args.variable_length:
        metrics = evaluate_infilling_flexmdm(model, tokenizer, dataloader, args, filename)
    else:
        metrics = evaluate_infilling_mdm(model, tokenizer, dataloader, args, filename)

        
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
                    "confidence_method": args.confidence_method,
                    "use_sliding_window": args.use_sliding_window,
                    "alpha": args.alpha,
                    "max_window": args.max_window,
                },
                f,
                indent=2,
            )
        
        # Clean up progress file after successful completion
        progress_file = filename.replace('.json', '_progress.pkl')
        if os.path.exists(progress_file):
            os.remove(progress_file)
            if dist.get_rank() == 0:
                print(f"Cleaned up progress file: {progress_file}")

    cleanup_ddp()
