import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
import torch.distributed as dist
import random
import numpy as np
from sft_trainer import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B")
    parser.add_argument("--sft_loss", type=str, default="which loss to train on")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity", default = "jaeyeon_kim-harvard-university")

    return parser.parse_args()

# Moel Loading with LoRA integration
def load_model_and_tokenizer(args):
    # load tokenizer
    # TODO: check caching / todo: check padding
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side = 'right', trust_remote_code = True, use_fast = True)
    
    # load model
    model = AutoModel.from_pretrained(
        args.model_name, 
        trust_remote_code = True,
        torch_dtype = torch.bfloat16)
    
    # LoRA config: as a default, we use LLaDA-D1's LoRA config
    lora_config = LoraConfig(
        r = 128,
        lora_alpha = 256,
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = TaskType.CAUSAL_LM)
    
    # applyig LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    return tokenizer, model


# Dataset loading
def load_dataset(args, tokenizer):
    # TODO: implement
    return None


# training loop
def train_model(args, tokenizer, model, dataset):
    # load dataset
    train_dataset, eval_dataset = load_dataset(args, tokenizer)

    # training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=2,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    # Initialize Trainer with custom dLLMTrainer

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    torch.manual_seed(42)
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)