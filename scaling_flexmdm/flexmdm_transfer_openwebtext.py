import torch
import torch.nn as nn
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_from_disk, Features, Sequence, Value, concatenate_datasets
from datasets.distributed import split_dataset_by_node
import os, multiprocessing, random, pathlib
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from flexmdm_trainer import *
from collections import Counter
from llada_dit import LLaDA_DIT
from pathlib import Path
import torch.distributed as dist
import random
import tqdm
import numpy as np
import wandb
import glob



def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # for the training speed, we comment this out


# ------------------------------------------------------------
# Util function for logging
# ------------------------------------------------------------
def count_parameters(named_params, key: str | None = None):
    return sum(p.numel()
        for n, p in named_params
        if p.requires_grad and (key is None or key in n)
    )

class LogLrCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if not is_main_process(args):
            return
        opt = kwargs["optimizer"]
        wandb.log(
            {
                "lr/lora": opt.param_groups[0]["lr"],
                "lr/token_head": opt.param_groups[1]["lr"],
                "lr/from_scratch": opt.param_groups[2]["lr"],
                "step": state.global_step,
            }
        )


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Name of the pretrained model"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="batch size per device")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Learning rate for the LoRA")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for other parameters")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=500000, help="Maximum number of training steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")
    parser.add_argument("--low_discrepancy", type=bool, default=False, help="whether to use low discrepancy sampling")

    # Output directory and job name
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/albergo_lab/Lab/transdim-flow/sft-datamix-checkpoints",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-sft-openwebtext", help="Job Name")
    parser.add_argument("--train_data", type=str, default="openwebtext", help="Path to training data")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--variable_length", action="store_true", help="whether to use variable length training")
    parser.add_argument("--sanity_run", action="store_true", help="whether to run the sanity run (overfitting the model)")

    # CLI flags for openwebtext dataset preprocessing
    parser.add_argument("--sft_max_length", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--cache_path", type=str, default="/n/netscratch/albergo_lab/Everyone/jay_brian/datamix", help="Path of the tokenized openwebtext dataset")

    return parser.parse_args()



# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    # Load the backbone LLaDA model
    backbone = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        return_dict=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right", trust_remote_code=True, use_fast=True)

    print("Tokenizer and backbone loaded!")

    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True

    # lora adapter for the backbone LLaDA
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "transformer.ff_out"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone = backbone.to(torch.bfloat16)

    if args.variable_length:
        model = LLaDA_DIT(backbone, pad_token_id = tokenizer.pad_token_id, d_model = 4096)
    else:
        model = backbone

    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.resume_from_checkpoint)
        state = torch.load(ckpt_dir/ "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Resumed from checkpoint {args.resume_from_checkpoint}")

    print("Final trainer model loaded!")
    
    return tokenizer, model


# Dataset loading
def load_data(args, tokenizer):
    # load the pre-processed tokenzied dataset (already int64)
    cache_dir = pathlib.Path(args.cache_path)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    ds = load_from_disk(cache_dir)
    ds = ds.shuffle(seed=42)
    data = ds.train_test_split(test_size=0.001, seed=42)
    print("Training and evaluation datasets successfully loaded!")

    if args.sanity_run:
        data = data["train"].select(range(128))
        print("Sanity run dataset loaded!")
        data.save_to_disk("sanity_run_dataset")
        return data, data

    return data["train"], data["test"]


# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        max_steps = args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy= 'steps',
        eval_steps = 1000,
        prediction_loss_only = True,
        logging_steps = 10,
        save_steps = 10000,
        save_total_limit=20,
        save_safetensors=False,
        max_grad_norm=1.0,
        bf16=True,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else None,
    )

    # setup the trainable parameters
    lora_params  = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    head_params  = [p for n, p in model.named_parameters() if "lora" not in n and "ff_out" in n and p.requires_grad]
    from_scratch_params = [p for n, p in model.named_parameters() if "lora" not in n and "ff_out" not in n and p.requires_grad]

    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    assert set(trainable) == set(lora_params) | set(head_params) | set(from_scratch_params), "Trainable parameters are not correctly set"

    # parameter count check
    print(f"Total trainable parameters: {count_parameters(model.named_parameters(), key = None)}")
    print(f"  └─ LoRA adapter params          : {count_parameters(model.named_parameters() , key = 'lora')}")
    print(f"  └─ Token Head params                  : {count_parameters(model.named_parameters(), key = 'ff_out')}")
    print(f"  └─ Scalar Length Head params        : {count_parameters(model.named_parameters(), key = 'scalar_length_head')}")
    print(f"  └─ Time embedding params            : {count_parameters(model.named_parameters(), key = '.temb_mod')}")


    # Initialize Trainer with custom dLLMTrainer
    if args.variable_length:
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0},
                {"params": head_params, "lr": args.lora_lr / 4, "weight_decay": 0.01},
                {"params": from_scratch_params, "lr": args.lr, "weight_decay": 0.01}
            ],
        )
        trainer = dLLMVariableLengthTrainer(
            model=model,
            args=training_args,
            data_collator=dLLMVariableDataCollator(tokenizer=tokenizer, mask_token_id=126336, 
            max_length=args.sft_max_length, compute_metrics = None, 
                low_discrepancy = args.low_discrepancy),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, None),
        )
    else:
        raise NotImplementedError("Currently we don't support fixed length training")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.wandb and local_rank == 0:
        wandb.init(project="SFT-llada", name=args.job_name, entity="jaeyeon_kim-harvard-university")

    # double-check the optimizer
    for i, g in enumerate(trainer.optimizer.param_groups):
        print(f"group {i}  init-lr={g['lr']}  wd={g['weight_decay']}")

    # add the callback
    trainer.add_callback(LogLrCallback())

    # Start training
    trainer.train()


if __name__ == "__main__":
    init_seed(42)
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, model)