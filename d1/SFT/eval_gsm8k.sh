#!/bin/bash
#SBATCH --job-name=test_sft_openwebtext
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=512GB
#SBATCH -C h100
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/sft_openwebtext/%j.out
#SBATCH --error=slurm_logs/sft_openwebtext/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com


export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=$SLURM_NODEID


python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    eval.py \
    --variable_length \
    --checkpoint_path /n/netscratch/albergo_lab/Lab/sft-gsm8k-checkpoints-2/llada-sft-gsm8k/checkpoint-2600/ \
    --output_dir results/gsm8k-2600 \
    --diffusion_steps 512 \
    --batch_size 16 \
    --dataset gsm8k
