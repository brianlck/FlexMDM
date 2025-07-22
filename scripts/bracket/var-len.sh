#!/bin/bash
#SBATCH --job-name=bracket
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/bracket/job-%j.out
#SBATCH -e slurm_logs/bracket/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com
#SBATCH --signal=SIGUSR1@90

source /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/.venv/bin/activate

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

srun python train.py --config-path config/bracket --config-name any_order