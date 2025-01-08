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

echo "Running the train script..."
# Run the Python script using the virtual environment's Python interpreter
python "$PROJECT_DIR/src/DistInference/train.py"

echo "Script execution completed successfully."
