#!/bin/bash
#SBATCH --job-name=load_dclm
#SBATCH --account=albergo_lab
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/vlmdm/job-%j.out


export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache

python scripts/prepare_data.py