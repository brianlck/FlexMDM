#!/bin/bash
#SBATCH --job-name=vlmdm_test_run_linear
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=100GB
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/vlmdm/job-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu


# define paths
source ~/.bashrc

conda deactivate
conda activate jay_vlmdm

python train.py --insertion_schedule_type linear --len_loss_scheduler --wandb
