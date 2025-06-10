#!/bin/bash
#SBATCH --job-name=vlmdm_test_run_geo
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


python train.py --mask_schedule_type geometric --unmask_loss_type ce --len_predict_type expectation --len_loss_scheduler --wandb