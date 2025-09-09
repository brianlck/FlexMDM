#!/bin/bash
#SBATCH --job-name=test_sft_openwebtext
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=512GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/sft_openwebtext/%j.out
#SBATCH --error=slurm_logs/sft_openwebtext/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

module load cuda/12.4.1-fasrc01
conda activate d1

export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=$SLURM_NODEID

export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --ntasks-per-node=1 --gpus-per-task=4 \
  python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    flexmdm_transfer_openwebtext.py \
      --wandb \
      --variable_length \
      --low_discrepancy True
