#!/bin/bash
#SBATCH --job-name=openwebtext-evaluation
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/openwebtext/job-%j.out
#SBATCH -e slurm_logs/openwebtext/job-%j.err
#SBATCH --constraint h100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brianlee.lck@gmail.com
#SBATCH --signal=SIGUSR1@90

source /n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/.venv/bin/activate

export HF_HOME=/n/netscratch/albergo_lab/Everyone/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Directory to search for JSON files
SEARCH_DIR="/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow"

# Directories to search
DIRS_TO_SEARCH=(
    "${SEARCH_DIR}"
    "${SEARCH_DIR}/tmp/owt/euler"
    "${SEARCH_DIR}/tmp/owt/tau-leaping"
)

# Loop through each directory
for dir in "${DIRS_TO_SEARCH[@]}"; do
    echo "Searching directory: $dir"
    
    # Find all JSON files matching the pattern and evaluate them
    for json_file in ${dir}/*_generated_samples_*.json; do
        if [ -f "$json_file" ]; then
            echo "Start processing: $json_file"
            
            # Extract filename without path and extension
            filename=$(basename "$json_file" .json)
            
            # Extract method and number from filename pattern {method}_generated_samples_{number}
            if [[ $filename =~ ^(.+)_generated_samples_([0-9]+)$ ]]; then
                method="${BASH_REMATCH[1]}"
                number="${BASH_REMATCH[2]}"
                
                # Create unique output filenames
                output_plot="${dir}/gpt_chunk_length_plot_${filename}.png"
                output_result="${dir}/gpt_chunk_${method}_eval_result_${number}.json"
                
                srun python evaluate_samples.py \
                    --input-json "$json_file" \
                    --batch-size 32 \
                    --length-plot-output "$output_plot" \
                    --results-output "$output_result" \
                    --eval-mode "chunk" \
                    --model-type "gpt2-xl"
                
                echo "Finished processing: $json_file"
            fi
        fi
    done
done
