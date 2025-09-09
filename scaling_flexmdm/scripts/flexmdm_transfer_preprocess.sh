#!/bin/bash
#SBATCH --job-name=datamix
#SBATCH --account=albergo_lab
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/datamix/%j.out
#SBATCH --error=slurm_logs/datamix/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

python flexmdm_transfer_preprocess.py