#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: Print each command before executing it (useful for debugging)
# set -x

# Define the project directory (optional)
PROJECT_DIR="/project_ghent/Mostafa/ActivityRecognition/DistInference"

# Navigate to the project directory
cd "$PROJECT_DIR"

echo "Activating the virtual environment..."
# Activate the virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Determine the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Number of GPUs available: $NUM_GPUS"

# Path to the training script
TRAIN_SCRIPT="src/DistInference/train_parallel.py"  # Adjust if your main script is named differently

# Path to the config file
CONFIG_PATH="config.yaml"  # Adjust the path if necessary

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching training on $NUM_GPUS GPUs using torchrun..."
    # Calculate the number of processes per node (typically the number of GPUs)
    PROC_PER_NODE=$NUM_GPUS

    # Launch the training script using torchrun for multi-GPU
    torchrun --nproc_per_node=$PROC_PER_NODE "$TRAIN_SCRIPT" --config "$CONFIG_PATH"
else
    echo "Launching training on a single GPU or CPU using python..."
    # Launch the training script using python for single-GPU or CPU
    python "$TRAIN_SCRIPT" --config "$CONFIG_PATH"
fi

echo "Script execution completed successfully."
