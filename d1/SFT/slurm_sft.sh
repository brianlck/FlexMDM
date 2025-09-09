#!/bin/bash
#SBATCH --job-name=sft_gsm8k
#SBATCH --output=slurm_logs/sft_gsm8k/%j.out
#SBATCH --error=slurm_logs/sft_gsm8k/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Add timeout and retry environment variables
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT=60
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=INFO

# Activate conda environment
source /n/home01/cklee/.bashrc
conda activate d1

# Run the training script
srun python run_sft_robust.py $@