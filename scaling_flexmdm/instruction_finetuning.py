import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from llada_trainer import *
from flexmdm_trainer import *
import random
import numpy as np
import wandb
from preprocess_math import preprocess_gsm8k
from llada_dit import LLaDA_DIT
import preprocess_code_infilling
from pathlib import Path


INFILL_DATASETS = ["code-infill"]

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------
# Util function for counting the number of parameters
# ------------------------------------------------------------
def count_parameters(named_params, key: str | None = None):
    return sum(p.numel()
        for n, p in named_params
        if p.requires_grad and (key is None or key in n)
    )


# ------------------------------------------------------------
# Util function for adding special tokens to the tokenizer
# ------------------------------------------------------------
SPECIAL_TOKENS = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]

def get_reasoning_tokenizer(model_name: str, model: AutoModel | None = None, add_to_model: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", trust_remote_code=True, use_fast=True)
    device = model.device
    
    missing = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    
    if add_to_model:
        model.resize_token_embeddings(len(tokenizer)).to(device)
    
    return tokenizer, model


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Learning rate for the LoRA")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for other parameters")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/albergo_lab/Lab/sft-datamix-gsm8k-checkpoints",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-sft-gsm8k-test-run", help="Job Name")
    parser.add_argument("--train_data", type=str, default="gsm8k", help="Path to training data")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--variable_length", action="store_true", help="whether to use variable length training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")

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
    print("Backbone LLaDA loaded!")

    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True

    # Add LoRA to backbone first (like eval.py)
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,  
        target_modules=["q_proj", "k_proj", "v_proj", "transformer.ff_out"],  
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    backbone = get_peft_model(backbone, lora_config)

    # Load tokenizer and add reasoning tokens if needed for GSM8K
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right", trust_remote_code=True, use_fast=True)
    
    print("Tokenizer loaded!")

    if args.variable_length:
        model = LLaDA_DIT(backbone, pad_token_id=tokenizer.pad_token_id, d_model=4096)
    else:
        model = backbone

    # Load checkpoint if provided (following eval.py pattern)
    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.resume_from_checkpoint)
        checkpoint_path = ckpt_dir / "pytorch_model.bin"
        
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
            print(f"Resumed from checkpoint {args.resume_from_checkpoint}")
        else:
            print(f"Checkpoint file not found at {checkpoint_path}")

    model = model.to(torch.bfloat16)
    print("Final trainer model loaded!")
    
    return tokenizer, model


# Dataset loading
def load_data(args, tokenizer):
    if args.train_data == "gsm8k":
        train_data = preprocess_gsm8k(
            split="train", tokenizer=tokenizer, max_length=args.max_length
        )
        eval_data = preprocess_gsm8k(
            split="test", tokenizer=tokenizer, max_length=args.max_length
        )
    elif args.train_data == "code-infill":
        train_data = code.preprocess_opc_coder(tokenizer, args.max_length)
        eval_data = code.preprocess_human_eval(tokenizer, args.max_length)
    else:
        data = load_dataset(args.train_data, split="train")
        train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)

    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset

 
# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps = 100,
        logging_steps = 10,
        save_steps = 100,
        save_safetensors=False,
        load_best_model_at_end=False,
        max_grad_norm=1.0,
        bf16=True,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else None,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )

    # setup the trainable parameters
    lora_params  = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    head_params  = [p for n, p in model.named_parameters() if "lora" not in n and p.requires_grad]

    # parameter count check
    num_lora_params = count_parameters(model.named_parameters(), key = 'lora')
    whole_trainable_params = count_parameters(model.named_parameters())
    print(f"Total trainable parameters: {whole_trainable_params:,}")
    print(f"  └─ LoRA adapter params          : {num_lora_params:,}")
    print(f"  └─ From-scratch params          : {whole_trainable_params - num_lora_params:,}")

    # Initialize Trainer with custom dLLMTrainer
    if args.variable_length:
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.1},
                {"params": head_params, "lr": args.lr, "weight_decay": 0.1}
            ],
        )

        if args.train_data in INFILL_DATASETS:
            infill_tokens = tokenizer.encode("<infill-boundary>")
            print("infill tokens: ", infill_tokens)
            trainer = dLLMVariableLengthTrainer(
                model=model,
                args=training_args,
                data_collator=dLLMVariableDataCollator(
                    tokenizer=tokenizer,
                    mask_token_id=126336,
                    max_length=args.max_length + 2 * len(infill_tokens),
                    low_discrepancy=True,
                    is_infill_task=True,
                    infill_tokens=infill_tokens
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(optimizer, None),
            )
        else:
            trainer = dLLMVariableLengthTrainer(
                model=model,
                args=training_args,
                data_collator=dLLMVariableDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length, low_discrepancy=True),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(optimizer, None),
            )
    else:
        if args.train_data in INFILL_DATASETS:
            instruction = tokenizer.encode("You are instructed to perform an infill task. Please fill in the missing parts of the code. The prefix and suffix are provided and delimited by <prefix> </prefix> and <suffix> </suffix> tokens.")
            prefix_delimiters = [tokenizer.encode("<prefix>"), tokenizer.encode("</prefix>")]
            suffix_delimiters = [tokenizer.encode("<suffix>"), tokenizer.encode("</suffix>")]
            total_length = args.max_length + len(prefix_delimiters[0]) + len(prefix_delimiters[1]) + len(suffix_delimiters[0]) + len(suffix_delimiters[1]) + len(instruction)
            trainer = dLLMTrainer(
                model=model,
                args=training_args,
                data_collator=dLLMDataCollator(
                    tokenizer=tokenizer,
                    mask_token_id=126336,
                    instruction_tokens=instruction,
                    prefix_delimiters=prefix_delimiters,
                    suffix_delimiters=suffix_delimiters,
                    max_length=total_length,
                    is_infill_task=True,
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        else:
            trainer = dLLMTrainer(
                model=model,
                args=training_args,
                data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(optimizer, None),
            )

        optimizer = torch.optim.AdamW(lora_params, lr = args.lora_lr, weight_decay = 0.1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.wandb and local_rank == 0:
        wandb.init(project="SFT-llada", name=args.job_name, entity="jaeyeon_kim-harvard-university")

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
