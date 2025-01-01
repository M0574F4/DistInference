# compute_global_stats.py

import os
import json
import yaml
import numpy as np
from scipy.signal import stft
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import logging

from get_fold_dataloaders import get_dataloaders_from_config


##### MAKE SURE YOU SET NORMALIZATION TO 'none' IN THE CONFIG BEFORE RUNNING THIS SCRIPT


def compute_global_stats(dataset):
    """
    Computes global statistics required for normalization.

    Parameters:
        dataset (RadarDataset): Instance of RadarDataset containing training data.

    Returns:
        dict: Dictionary containing 'linear_min', 'linear_max', 'db_mean', 'db_std'.
    """
    linear_min = np.inf
    linear_max = -np.inf
    db_sum = 0.0
    db_sq_sum = 0.0
    db_count = 0

    for idx in tqdm(range(len(dataset)), desc="Processing files"):
        try:
            magnitude, _ = dataset[idx]
        except ValueError as e:
            logging.warning(f"Skipping file at index {idx}: {e}")
            continue

        # Update linear min and max
        current_min = magnitude.min()
        current_max = magnitude.max()
        if current_min < linear_min:
            linear_min = current_min
        if current_max > linear_max:
            linear_max = current_max

        # Convert magnitude to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-6)  # Add epsilon to avoid log(0)

        # Update db statistics
        db_sum += magnitude_db.sum()
        db_sq_sum += (magnitude_db ** 2).sum()
        db_count += magnitude_db.numel()

    if db_count == 0:
        logging.error("No valid dB data found. Cannot compute statistics.")
        raise ValueError("No valid dB data found.")

    db_mean = db_sum / db_count
    db_std = np.sqrt((db_sq_sum / db_count) - (db_mean ** 2))

    global_stats = {
        'linear_min': float(linear_min),
        'linear_max': float(linear_max),
        'db_mean': float(db_mean),
        'db_std': float(db_std)
    }

    return global_stats

def main():

    config_path = '/project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/config.yaml'  # Path to your YAML config file
    _, _, _, train_dataset, _, _, fold_index = get_dataloaders_from_config(config_path)

    # Compute global statistics
    global_stats = compute_global_stats(train_dataset)

    logging.info("Computed Global Statistics:")
    logging.info(json.dumps(global_stats, indent=4))
    print("Computed Global Statistics:")
    print(json.dumps(global_stats, indent=4))

    # Save global_stats to a JSON file
    output_path = 'global_stats.json'  # Adjust the path if necessary
    with open(output_path, 'w') as f:
        json.dump(global_stats, f, indent=4)

    logging.info(f"Global statistics saved to {output_path}")
    print(f"Global statistics saved to {output_path}")

if __name__ == "__main__":
    main()
