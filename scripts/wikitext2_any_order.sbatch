#!/bin/bash
#SBATCH --job-name=wikitext2_any_order
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/wikitext2/job-%j.out

python train.py --config-path config/wikitext2 --config-name any_order