#!/bin/bash
#SBATCH --job-name=generate_samples
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB
#SBATCH --constraint h100
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/vlmdm/job-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com
#SBATCH --array=15-19

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# define models, checkpoints, step sizes, and samplers
MODELS=(mdm flow)
CKPTS=(
  /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/outputs/2025-06-30/18-45-34/checkpoints/openwebtext/mdm/20250630-184537/last.ckpt
  /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/outputs/2025-06-28/10-53-14/checkpoints/openwebtext/any_order/20250628-105318/last.ckpt
)
STEP_SIZES=(128 256 1024 2048 4096)
SAMPLERS=(euler tau-leaping)

# compute which model, sampler, and step to run
TOTAL_STEPS=${#STEP_SIZES[@]}
TOTAL_SAMPLERS=${#SAMPLERS[@]}
TASK=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$(( TASK / (TOTAL_STEPS * TOTAL_SAMPLERS) ))
REM=$(( TASK % (TOTAL_STEPS * TOTAL_SAMPLERS) ))
SAMPLER_IDX=$(( REM / TOTAL_STEPS ))
STEP_IDX=$(( REM % TOTAL_STEPS ))

MODEL=${MODELS[MODEL_IDX]}
CKPT=${CKPTS[MODEL_IDX]}
SAMPLER=${SAMPLERS[SAMPLER_IDX]}
STEP=${STEP_SIZES[STEP_IDX]}

echo "Running model=$MODEL sampler=$SAMPLER step=$STEP (task $TASK)"

# generate samples
srun python generate_samples.py \
    --checkpoint_path "${CKPT}" \
    --total_samples 1024 \
    --model_type "${MODEL}" \
    --sampler_type "${SAMPLER}" \
    --step_size "${STEP}" \
    -o "tmp/owt/${SAMPLER}/${MODEL}_generated_samples_${STEP}.json"

# evaluate samples
srun python evaluate_samples.py \
    --input-json "tmp/owt/${SAMPLER}/${MODEL}_generated_samples_${STEP}.json" \
    --batch-size 32 \
    --results-output "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/tmp/owt/${SAMPLER}/${MODEL}_eval_results_${STEP}.json" \
    --length-plot-output "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/tmp/owt/${SAMPLER}/${MODEL}_length_plot_${STEP}.png"

