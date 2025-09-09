#!/bin/bash
#SBATCH --job-name=gen_humaneval
#SBATCH --account=albergo_lab
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=128GB
#SBATCH -C h200
#SBATCH --cpus-per-task=16
#SBATCH --time=03-00:00:00
#SBATCH --output=slurm_logs/humaneval_sample/%j.out
#SBATCH --error=slurm_logs/humaneval_sample/%j.err
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
    eval_humaneval_infill.py \
    --checkpoint_path /n/netscratch/albergo_lab/Lab/brianlck/checkpoints/llada-varlen-code/llada-sft-varlen-code-infill/llada-sft-code-infill/last-checkpoint \
    --output_dir human_eval/var-len \
    --alpha 15.0 \
    --max_window 32 \
    --diffusion_steps 4096 \
    --confidence_method top_prob \
    --variable_length \
    --use_sliding_window \
    --batch_size 4 \

