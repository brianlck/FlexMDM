#!/bin/bash
#SBATCH --job-name=wikitex2
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm_logs/wikitext2/job-%j.out
#SBATCH -C a100

python train.py --config-path config/bracket --config-name any_order \
    training.loss_fn.insert=distribution