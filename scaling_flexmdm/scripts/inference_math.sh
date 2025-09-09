#!/bin/bash
#SBATCH --job-name=test_sft_openwebtext
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/sft_openwebtext/%j.out
#SBATCH --error=slurm_logs/sft_openwebtext/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID


srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    eval_gsm8k.py \
    --variable_length \
    --checkpoint_path_variable /n/netscratch/albergo_lab/Lab/sft-gsm8k-checkpoints/llada-sft-gsm8k/checkpoint-9800/ \
    --output_dir results/gsm8k \
    --diffusion_steps 256 \
    --batch_size 8 \
    --dataset gsm8k
