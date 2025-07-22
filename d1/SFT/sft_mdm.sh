#!/bin/bash
#SBATCH --job-name=test_sft_mdm
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=100GB
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/sft/job-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu


# define paths
source ~/.bashrc

conda deactivate
conda activate jay_vlmdm
module load cuda/12.4.1-fasrc01


python -m torch.distributed.run --nproc_per_node=4 sft_train.py --wandb --job_name=llada-sft-gsm8k-mdm-longer