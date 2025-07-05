#!/bin/bash
#SBATCH --job-name=openwebtext-mdm-linear
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200GB
#SBATCH --time=3-00:00:00
#SBATCH -o slurm_logs/openwebtext/job-%j.out
#SBATCH -e slurm_logs/openwebtext/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com
#SBATCH --signal=SIGUSR1@90

source /n/netscratch/albergo_lab/Everyone/brianlck/interpretable-flow/.venv/bin/activate

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=$SLURM_NODEID

srun python train.py --config-path config/openwebtext --config-name mdm