#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

echo "Setup complete. Virtual environment 'venv' is activated."
