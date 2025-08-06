from lightning_modules import AnyOrderInsertionFlowModule, MaskedDiffusionModule
from sampling import (
    any_order_mask_insertion_euler_sampling,
    mdm_euler_sampling,
    any_order_mask_insertion_tau_leaping_sampling,
    mdm_tau_leaping_sampling,
)
from schedule import GeometricSchedule, LinearSchedule
from data.text import setup_tokeniser
import torch
import torch.nn as nn
import argparse
import json


torch.set_float32_matmul_precision("high")
torch.set_printoptions(threshold=10_000)

# Add argparse and remove hard-coded checkpoint_path
parser = argparse.ArgumentParser(description="Generate samples")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    required=True,
    help="Path to the model checkpoint file",
)
# Add argparse arguments
parser.add_argument(
    "--total_samples",
    "-n",
    type=int,
    default=1024,
    help="Total number of samples to generate",
)
parser.add_argument(
    "--model_type",
    choices=["flow", "mdm"],
    default="flow",
    help="Model type to use: 'flow' or 'mdm'",
)
parser.add_argument(
    "--output_file",
    "-o",
    type=str,
    default="generated_samples.json",
    help="Path to save generated samples JSON",
)
parser.add_argument(
    "--batch_size", "-b", type=int, help="Batch size; defaults to #GPUs or 1"
)
parser.add_argument(
    "--step_size",
    type=int,
    default=2048,
    help="Number of sampling steps",
)
parser.add_argument(
    "--sampler_type",
    type=str,
    default="euler",
    choices=["euler", "tau-leaping"],
)
args = parser.parse_args()
checkpoint_path = args.checkpoint_path
output_file = args.output_file

# Load chosen model
if args.model_type == "flow":
    model = AnyOrderInsertionFlowModule.load_from_checkpoint(checkpoint_path)
elif args.model_type == "mdm":
    model = MaskedDiffusionModule.load_from_checkpoint(checkpoint_path)
else:
    raise ValueError(f"Unknown model_type: {args.model_type}")
model.swap_to_ema()
model.eval()

# distribute model across GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

tokeniser = setup_tokeniser("gpt2")

# determine processing batch size
per_gpu_batch_size = 32
if args.batch_size:
    processing_batch_size = args.batch_size
elif torch.cuda.is_available():
    processing_batch_size = per_gpu_batch_size * torch.cuda.device_count()
else:
    processing_batch_size = per_gpu_batch_size

step_size = args.step_size
total_samples = args.total_samples

all_samples = []
all_raw_samples = []

sample_fn = None

match (args.model_type, args.sampler_type):
    case ("flow", "euler"):
        sample_fn = any_order_mask_insertion_euler_sampling
    case ("mdm", "euler"):
        sample_fn = mdm_euler_sampling
    case ("flow", "tau-leaping"):
        sample_fn = any_order_mask_insertion_tau_leaping_sampling
    case ("mdm", "tau-leaping"):
        sample_fn = mdm_tau_leaping_sampling


for _ in range(0, total_samples, processing_batch_size):
    samples, _ = sample_fn(
        model,
        steps=step_size,
        mask=model.interpolant.mask_token,
        pad=model.interpolant.pad_token,
        batch_size=processing_batch_size,
        max_length=model.interpolant.max_length,
        return_trace=False,
    )
    if args.model_type == "flow":
        all_raw_samples.extend(samples.cpu().tolist())
    else:
        all_raw_samples.extend(samples.cpu().tolist())
        # post-process mdm samples: replace tokens after first pad with pad
        pad_id = model.interpolant.pad_token
        B, L = samples.shape
        pos = torch.arange(L, device=samples.device).unsqueeze(0).expand(B, L)
        pad_mask = samples == pad_id
        idxs = torch.where(pad_mask, pos, torch.full_like(pos, L))
        first_idx = idxs.min(dim=1).values
        mask_after = pos > first_idx.unsqueeze(1)
        samples[mask_after] = pad_id

    # store raw token IDs
    # Decode and strip samples
    decoded_samples = tokeniser.batch_decode(samples, skip_special_tokens=True)
    all_samples.extend(decoded_samples)

with open(output_file, "w") as f:
    json.dump(all_samples, f, indent=2)
# also write raw token outputs
with open(output_file.replace(".json", "_raw.json"), "w") as f:
    json.dump(all_raw_samples, f, indent=2)
