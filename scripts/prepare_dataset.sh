#!/bin/bash
#SBATCH --job-name=load_dclm
#SBATCH --account=albergo_lab
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --mem=500GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=64      # Match num_proc
#SBATCH --tmp=100G              # Local scratch space
#SBATCH --output=slurm_logs/vlmdm/job-%j.out

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

python scripts/prepare_data.py