import os
import json
import random
import yaml  # For reading YAML configuration files

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from scipy.signal import stft

# -------------
# Helper Functions
# -------------
def parse_filename_get_label(filename):
    """
    Map the first character (K) to an integer label (0..5) or similar.
    For example:
       '1' -> 0  # walking
       '2' -> 1  # sitting down
       '3' -> 2  # stand up
       ...
    Adjust as needed for your classification problem.
    """
    base = os.path.basename(filename)
    label_char = base[0]
    label_map = {
        '1': 0,  # walking
        '2': 1,  # sitting down
        '3': 2,  # stand up
        '4': 3,  # pick up an object
        '5': 4,  # drink water
        '6': 5,  # fall
    }
    return label_map.get(label_char, -1)  # -1 if not found

def parse_complex_line(line):
    """
    Convert a line into a Python complex number, 
    replacing 'i'/'I' with 'j'.
    """
    line = line.replace('i', 'j').replace('I', 'j')
    try:
        return complex(line)
    except ValueError:
        return None

# -------------
# The RadarDataset Class
# -------------
class RadarDataset(Dataset):
    def __init__(self, file_paths, transform=None, additive_noise_std=0.0, stft_settings=None,
                 normalization='db', global_stats=None, force_dim= None):
        """
        file_paths: list of .dat files
        transform: optional transform (e.g. a function or lambda)
        additive_noise_std: std dev for Gaussian noise, 0 means no noise
        stft_settings: dict containing 'window_size' and 'overlap'
        normalization: 'linear', 'db', or 'none'
        global_stats: dict containing 'mean' and 'std' if normalization requires it
        """
        self.file_paths = file_paths
        self.transform = transform
        self.additive_noise_std = additive_noise_std
        self.stft_settings = stft_settings
        self.normalization = normalization
        self.global_stats = global_stats
        self.force_dim = force_dim
        

        if self.normalization not in ['linear', 'db', 'none']:
            raise ValueError(f"Unsupported normalization type: {self.normalization}")

        if self.normalization == 'db' and self.global_stats is None:
            raise ValueError("global_stats must be provided for 'db' normalization.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = parse_filename_get_label(filepath)
        # --- Step 1: Load .dat file ---
        with open(filepath, 'r') as f:
            content = f.read().strip().split('\n')

        if len(content) < 5:
            raise ValueError(f"File {filepath} does not contain enough lines for metadata and data.")

        # Extract metadata (first 4 lines)
        try:
            carrier_freq_ghz = float(content[0])
            chirp_duration_ms = float(content[1])
            samples_per_beat_note = int(float(content[2]))
            bandwidth_mhz = float(content[3])
        except ValueError as e:
            raise ValueError(f"Error parsing metadata in file {filepath}: {e}")

        data_lines = content[4:]
        complex_samples = [parse_complex_line(line) for line in data_lines]
        # Filter out None entries
        complex_samples = [s for s in complex_samples if s is not None]
        complex_samples = np.array(complex_samples, dtype=np.complex64)

        total_samples = len(complex_samples)
        if samples_per_beat_note <= 0 or total_samples < samples_per_beat_note:
            raise ValueError(f"Invalid file {filepath}: samples_per_beat_note = {samples_per_beat_note}, total_samples = {total_samples}")

        # Optionally reshape to beat_notes
        total_beat_notes = total_samples // samples_per_beat_note
        complex_samples = complex_samples[: total_beat_notes * samples_per_beat_note]
        beat_notes = complex_samples.reshape((total_beat_notes, samples_per_beat_note))

        # --- Example: compute slow-time signal (average across each chirp) ---
        # For demonstration, let's assume samples_per_chirp = samples_per_beat_note
        # You can adapt logic for your real application
        num_chirps = total_samples // samples_per_beat_note
        complex_data = complex_samples.reshape((num_chirps, samples_per_beat_note))
        slow_time_signal = np.mean(complex_data, axis=1)
        # Remove DC
        slow_time_signal -= np.mean(slow_time_signal)

        # --- Step 2: STFT ---
        prf = 1.0 / (chirp_duration_ms * 1e-3)
        window_size = self.stft_settings['window_size']
        overlap = self.stft_settings['overlap']
        f_vals, t_vals, Zxx = stft(
            slow_time_signal,
            fs=prf,
            window='hann',
            nperseg=window_size,
            noverlap=overlap,
            nfft=window_size * 2,
            return_onesided=False
        )
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        # magnitude
        magnitude = np.abs(Zxx_shifted)

        # --- Step 3: Normalization ---
        if self.normalization == 'linear':
            # Normalize in the linear domain using global min and max
            if self.global_stats is not None:
                mag_min = self.global_stats['linear_min']
                mag_max = self.global_stats['linear_max']
                magnitude_normalized = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)
                magnitude_normalized = np.clip(magnitude_normalized, 0.0, 1.0)
            else:
                # Per-sample normalization (not recommended)
                magnitude_min = magnitude.min()
                magnitude_max = magnitude.max()
                magnitude_normalized = (magnitude - magnitude_min) / (magnitude_max - magnitude_min + 1e-10)
        elif self.normalization == 'db':
            # Convert to dB first
            magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add epsilon to avoid log(0)
            # Normalize using global mean and std
            magnitude_db_normalized = (magnitude_db - self.global_stats['db_mean']) / self.global_stats['db_std']
            # Optionally, clip to a range (e.g., -3 to +3)
            magnitude_db_normalized = np.clip(magnitude_db_normalized, -3.0, 3.0)
            # Scale to [0,1]
            magnitude_normalized = (magnitude_db_normalized + 3.0) / 6.0
            magnitude_normalized = np.clip(magnitude_normalized, 0.0, 1.0)
        else:
            # No normalization
            magnitude_normalized = magnitude
        # --- Step 4: Optional additive noise ---
        if self.additive_noise_std > 0:
            noise = np.random.normal(0, self.additive_noise_std, size=magnitude_normalized.shape)
            magnitude_normalized += noise
        if self.normalization != 'none':
            # Ensure the magnitude_normalized remains within [0, 1]
            magnitude_normalized = np.clip(magnitude_normalized, 0.0, 1.0)

        # --- Step 5: Convert to float32 ---
        magnitude_normalized = magnitude_normalized.astype(np.float32)

        # --- Step 6: (Optional) Apply any additional transformations ---
        if self.transform is not None:
            magnitude_normalized = self.transform(magnitude_normalized)

        # --- Step 7: Prepare the final tensor ---
        spectrogram_tensor = torch.tensor(magnitude_normalized).unsqueeze(0)  # Shape: (1, freq_bins, time_bins)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.force_dim is not None:
            _, _, X = spectrogram_tensor.size()
            if self.force_dim > X:
                repeat_factor = (self.force_dim + X - 1) // X
                spectrogram_tensor = spectrogram_tensor.repeat(1, 1, repeat_factor)[..., :self.force_dim]

        return spectrogram_tensor, label_tensor

def get_dataloaders(
    folds_json_path, 
    fold_index=0, 
    train_split_percentage=0.8, 
    batch_size_train=32, 
    batch_size_val=32, 
    batch_size_test=32,
    normalization='none',
    shuffle_train=True, 
    shuffle_val=False, 
    shuffle_test=False,
    additive_noise_std_train=0.0,
    additive_noise_std_val=0.0,
    additive_noise_std_test=0.0,
    global_stats='none',
    force_dim=None,
    num_workers_train=0,
    num_workers_val=0,
    num_workers_test=0,
    stft_settings=None
):
    """
    Given the path to a JSON with the k folds, pick one fold as test,
    the other folds as train. Then from train, take a fraction for
    train vs. validation, create Datasets and DataLoaders.

    The JSON file still contains .dat paths, but we look for matching
    .npz files in '/project_ghent/Mostafa/ActivityRecognition/preprocessed'.
    """

    # Where your preprocessed files are stored:
    PREPROCESSED_DIR = "/project_ghent/Mostafa/ActivityRecognition/preprocessed"

    # Helper to convert raw .dat path -> corresponding .npz path
    def dat_to_npz_path(dat_path, preprocessed_dir=PREPROCESSED_DIR):
        """
        Example:
          dat_path = '/some/folder/1_sample.dat'
          -> '/project_ghent/Mostafa/ActivityRecognition/preprocessed/1_sample.npz'
        """
        base = os.path.basename(dat_path)   # e.g. '1_sample.dat'
        base_no_ext, _ = os.path.splitext(base)  # e.g. '1_sample'
        npz_name = base_no_ext + ".npz"
        return os.path.join(preprocessed_dir, npz_name)

    # 1) Load the fold info
    with open(folds_json_path, "r") as f:
        folds = json.load(f)

    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index {fold_index} is out of range. Number of folds: {len(folds)}")

    # 2) Extract test files (.dat) from the chosen fold, train files from the rest
    test_dat_files = folds[fold_index]["test"]  # list of .dat paths
    train_dat_files = []
    for i, fold in enumerate(folds):
        if i != fold_index:
            train_dat_files.extend(fold["test"])  # from other folds

    # 3) Convert .dat -> .npz paths
    test_npz_files = [dat_to_npz_path(fp) for fp in test_dat_files]
    train_npz_files = [dat_to_npz_path(fp) for fp in train_dat_files]

    # 4) Split train_npz_files -> train + val
    random_seed = 42  # For reproducibility
    random.Random(random_seed).shuffle(train_npz_files)
    split_idx = int(len(train_npz_files) * train_split_percentage)
    train_subset_npz = train_npz_files[:split_idx]
    val_subset_npz = train_npz_files[split_idx:]

    # 5) Create Datasets (now pointing to .npz paths)
    train_dataset = RadarDataset(
        file_paths=train_subset_npz,
        transform=None,  # or some transform
        additive_noise_std=additive_noise_std_train,
        normalization=normalization,        # 'none', 'db', or 'linear'
        stft_settings=stft_settings,        # Not used if STFT is precomputed, kept for API compat
        global_stats=global_stats,
        force_dim=force_dim
    )
    val_dataset = RadarDataset(
        file_paths=val_subset_npz,
        transform=None,
        additive_noise_std=additive_noise_std_val,
        normalization=normalization,
        stft_settings=stft_settings,
        global_stats=global_stats,
        force_dim=force_dim
    )
    test_dataset = RadarDataset(
        file_paths=test_npz_files,
        transform=None,
        additive_noise_std=additive_noise_std_test,
        normalization=normalization,
        stft_settings=stft_settings,
        global_stats=global_stats,
        force_dim=force_dim
    )

    # 6) Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=shuffle_train, 
        num_workers=num_workers_train,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=shuffle_val, 
        num_workers=num_workers_val,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size_test, 
        shuffle=shuffle_test, 
        num_workers=num_workers_test,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def load_config(config_path):
    """
    Load YAML configuration file.

    Parameters:
        config_path (str): Path to the YAML config file.

    Returns:
        config (dict): Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    """
    Example usage:
      python dataset_and_dataloader.py --config config.yaml
    """
    parser = argparse.ArgumentParser(description="Radar Dataset and Dataloader Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract STFT settings
    stft_settings = config.get('stft', {'window_size': 128, 'overlap': 64})

    if normalization == 'none' or normalization ==  'db':
        global_stats = load_config("/project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/global_stats.json")
        print(global_stats)

    # Retrieve DataLoaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
        folds_json_path=config['folds_json'],
        fold_index=config['fold_index'],
        train_split_percentage=config['train_split_percentage'],
        batch_size_train=config['batch_sizes']['train'],
        batch_size_val=config['batch_sizes']['validation'],
        batch_size_test=config['batch_sizes']['test'],
        shuffle_train=config['shuffle']['train'],
        shuffle_val=config['shuffle']['validation'],
        shuffle_test=config['shuffle']['test'],
        additive_noise_std_train=config['additive_noise_std']['train'],
        additive_noise_std_val=config['additive_noise_std']['validation'],
        additive_noise_std_test=config['additive_noise_std']['test'],
        normalization=config['normalization'],
        global_stats=global_stats,
        force_dim=config['force_dim'],
        num_workers_train=config['num_workers']['train'],
        num_workers_val=config['num_workers']['validation'],
        num_workers_test=config['num_workers']['test'],
        stft_settings=stft_settings
    )

    print(f"Fold Index: {config['fold_index']}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Example iteration
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print("Batch index:", i, "Data shape:", batch_data.shape, "Labels:", batch_labels)
        # do something...
        break
