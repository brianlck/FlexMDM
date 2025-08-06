#!/bin/bash
#SBATCH --job-name=generate_samples_single
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB
#SBATCH --constraint h100
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/vlmdm/job-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com

source /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/.venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Configuration parameters - modify these as needed
MODEL="flow"
CKPT="/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/outputs/2025-07-29/00-17-21/checkpoints/openwebtext/any_order/20250729-001723/last.ckpt"
SAMPLER="tau-leaping"
STEP=1024
TOTAL_SAMPLES=1024
SUBDIR="new_linear"

echo "Running model=$MODEL sampler=$SAMPLER step=$STEP"

# Create output directory
mkdir -p "tmp/owt/${SAMPLER}"

# generate samples
# srun python generate_samples.py \
#     --checkpoint_path "${CKPT}" \
#     --total_samples "${TOTAL_SAMPLES}" \
#     --model_type "${MODEL}" \
#     --sampler_type "${SAMPLER}" \
#     --step_size "${STEP}" \
#     -o "tmp/owt/${SAMPLER}/${SUBDIR}/${MODEL}_generated_samples_${STEP}.json"

# evaluate samples
srun python evaluate_samples.py \
    --input-json "tmp/owt/${SAMPLER}/${SUBDIR}/${MODEL}_generated_samples_${STEP}.json" \
    --batch-size 32 \
    --results-output "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/tmp/owt/${SAMPLER}/${SUBDIR}/${MODEL}_eval_results_${STEP}.json" \
    --length-plot-output "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/tmp/owt/${SAMPLER}/${SUBDIR}/${MODEL}_length_plot_${STEP}.png" \
    --eval-mode "sentence" \
    --perplexity \
    --entropy \
    --mauve \
    --reference-perplexity

echo "Job completed successfully"
