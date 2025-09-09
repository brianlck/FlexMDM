#!/bin/bash
#SBATCH --job-name=gen_gsm8k_samples
#SBATCH --account=albergo_lab
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=128GB
#SBATCH -C h200
#SBATCH --cpus-per-task=16
#SBATCH --time=01-00:00:00
#SBATCH --output=slurm_logs/gsm8k_sample/%j.out
#SBATCH --error=slurm_logs/gsm8k_sample/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com


export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=$SLURM_NODEID


python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    eval.py \
    --checkpoint_path /n/netscratch/sham_lab/Everyone/jay_brian/ckpt-for-paper/mdm-gsm8k/jay-gsm8k-mdm/checkpoint-9320 \
    --output_dir results/mdm-test \
    --alpha 15.0 \
    --max_window 32 \
    --diffusion_steps 1024 \
    --confidence_method top_prob \
    --use_sliding_window \
    --batch_size 1 \
    --dataset gsm8k


