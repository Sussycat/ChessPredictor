#!/bin/bash

# --- CONFIGURATION ---
# We define the scratch location explicitly to avoid Home directory quota issues
SCRATCH_DIR="/scratch/user/nguye3hv"
ENV_NAME="chess"  # Create environment in Scratch
PYTHON_VERSION="3.10"

# --- 1. CACHE REDIRECTION (The Fix for "Disk Quota Exceeded") ---
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"

# Create these directories if they don't exist
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

# --- 2. PREPARATION ---
echo "-----------------------------------------"
echo "🚀 Starting Hybrid Reset (Conda + UV)"
echo "📂 Target Env Name: $ENV_NAME"
echo "📂 Cache: $SCRATCH_DIR/.cache"
echo "-----------------------------------------"

# Initialize Conda for the script (Crucial step!)
# This allows 'conda activate' to work inside a script
eval "$(conda shell.bash hook)"

# Attempt to load proxy (Works on Cluster, ignores errors on Laptop)
echo "🌐 Checking WebProxy..."
if command -v module &> /dev/null; then
    module load WebProxy
fi

# --- 3. DELETION ---
# Check if the environment name exists in the list of conda environments
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "🗑️  Deleting existing environment '$ENV_NAME'..."
    conda deactivate
    conda env remove -n "$ENV_NAME" -y
else
    echo "ℹ️  Environment '$ENV_NAME' not found. Skipping deletion."
fi

# --- 4. CREATION (Using Conda) ---
echo "🛠️  Creating Conda environment..."
# We use -n to name it (Conda decides the path based on config, usually ~/.conda/envs or the scratch location if configured globally)
conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y

# Activate the environment
echo "🔌 Activating Conda environment..."
conda activate "$ENV_NAME"

# --- 5. INSTALLATION ---
echo "⬇️ Installing uv..."
pip install uv --no-cache-dir

echo "⬇️ Installing requirements..."
uv pip install --no-cache -r requirements.txt

echo "✅ Done"