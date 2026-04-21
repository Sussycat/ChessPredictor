#!/bin/bash

# --- SLURM DIRECTIVES ---
#SBATCH --job-name=dl_models      # Job name
#SBATCH --time=04:00:00           # Time limit (4 hours)
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=4         # CPUs for I/O operations
#SBATCH --mem=16G                 # Memory (16GB is usually plenty for downloading)
#SBATCH --partition=medium      # Partition name (CHANGE THIS to your cluster's partition, e.g., 'gpu', 'compute')
#SBATCH --job-name=download_models        # Give your job a name
#SBATCH --output=/home/nguye3hv/TextEE_Hung/logs/download_models_%j.txt # Output log file

# --- CONFIGURATION ---
BASE_DIR="/scratch/user/nguye3hv/models"

MODELS=(
    "meta-llama/Llama-3.2-11B-Vision-Instruct"   # Index 0
    "meta-llama/Llama-3.2-90B-Vision-Instruct"   # Index 1
    "HuggingFaceH4/zephyr-7b-alpha"              # Index 2
    "mistralai/Mixtral-8x7B-Instruct-v0.1"       # Index 3
    "Qwen/Qwen3.5-9B"                            # Index 4
)

# --- PREPARATION ---
echo "-----------------------------------------"
echo "🚀 Starting Slurm Model Download Job"
echo "📅 Date: $(date)"
echo "💻 Node: $(hostname)"
echo "-----------------------------------------"

# 1. Check and Create Directory
if [ -d "$BASE_DIR" ]; then
    echo "📂 Base Directory exists: $BASE_DIR"
else
    echo "📂 Directory not found. Creating: $BASE_DIR"
    mkdir -p "$BASE_DIR"
fi

# 2. Determine which models to process
# Note: sbatch passes arguments to the script if you run: sbatch script.sh 0 1
if [ $# -eq 0 ]; then
    echo "ℹ️  No specific indices provided. Downloading ALL models."
    TARGET_INDICES=("${!MODELS[@]}")
else
    echo "ℹ️  Indices provided via command line: $@"
    TARGET_INDICES=("$@")
fi

echo "-----------------------------------------"

# Load Proxy (Common on clusters)
if command -v module &> /dev/null; then
    echo "🌐 Loading WebProxy..."
    module load WebProxy
fi

# Activate env
source /sw/eb/sw/Miniconda3/23.10.0-1/bin/activate textee3

# --- DOWNLOAD LOOP ---
for i in "${TARGET_INDICES[@]}"; do
    
    # Check if input is a valid number
    if ! [[ "$i" =~ ^[0-9]+$ ]]; then
        echo "⚠️  Skipping invalid input: '$i' (Not a number)"
        continue
    fi

    # Check if index exists in the array
    if [ -z "${MODELS[$i]}" ]; then
        echo "⚠️  Skipping Index $i: No model defined at this index."
        continue
    fi

    model_id="${MODELS[$i]}"
    
    # REMOVE THE PREFIX
    model_name=${model_id##*/}
    target_folder="$BASE_DIR/$model_name"

    echo "-----------------------------------------"
    echo "Processing Index [$i]"
    echo "⬇️  Source:      $model_id"
    echo "📂 Destination: $target_folder"
    echo "-----------------------------------------"

    # Download using --local-dir
    # Note: Ensure 'hf' (huggingface-cli) is installed in your environment
    hf download "$model_id" \
        --local-dir "$target_folder"

    if [ $? -eq 0 ]; then
        echo "✅ Success: $model_name is ready."
    else
        echo "❌ Failed: $model_name"
    fi
done

echo "-----------------------------------------"
echo "🎉 Job finished."
echo "-----------------------------------------"