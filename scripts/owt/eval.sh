#!/bin/bash
#SBATCH --job-name=openwebtext-evaluation
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=3-00:00:00
#SBATCH -o slurm_logs/openwebtext/job-%j.out
#SBATCH -e slurm_logs/openwebtext/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com
#SBATCH --signal=SIGUSR1@90

source /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/.venv/bin/activate

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

srun python evaluate_samples.py \
    --input-json /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/generated_samples_2048.json \
    --batch-size 32 \
    --length-plot-output /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/length_plot_var_len_2048.png \
