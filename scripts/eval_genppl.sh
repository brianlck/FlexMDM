#!/bin/bash
#SBATCH --job-name=eval_genppl
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/eval/job-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu


# define paths
source ~/.bashrc

conda deactivate
conda activate jay_vlmdm

# MDM
python evaluation.py --model mdm --num_samples 1024 --temperature 0.9 --genppl

python evaluation.py --model vlmdm --num_samples 1024 --temperature 0.9 --genppl