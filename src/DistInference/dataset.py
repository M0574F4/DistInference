# import os
# import json
# import random
# import yaml
# import torch
# import numpy as np
# import argparse

# from torch.utils.data import Dataset, DataLoader

# # -------------
# # Helper Functions
# # -------------
# def parse_filename_get_label(filename):
#     """
#     Same as before, but now 'filename' points to .npz (not .dat).
#     We'll reuse the logic (look at the first character).
#     """
#     base = os.path.basename(filename)
#     # Remove the .npz extension if needed
#     base_no_ext, _ = os.path.splitext(base)
#     label_char = base_no_ext[0]
#     label_map = {
#         '1': 0,  # walking
#         '2': 1,  # sitting down
#         '3': 2,  # stand up
#         '4': 3,  # pick up an object
#         '5': 4,  # drink water
#         '6': 5,  # fall
#     }
#     return label_map.get(label_char, -1)  # -1 if not found

# # -------------
# # The RadarDataset Class
# # -------------
# class RadarDataset(Dataset):
#     def __init__(
#         self, 
#         file_paths,              # list of .npz file paths
#         transform=None, 
#         additive_noise_std=0.0, 
#         stft_settings=None,      # <--- no longer used here, since STFT is done offline
#         normalization='none',    # if 'none', data is used as-is; if 'db' or 'linear', see note below
#         global_stats=None,       # used only if we still want to do runtime normalization
#         force_dim=None
#     ):
#         """
#         file_paths: list of .npz files
#         transform: optional transform (e.g. a function or lambda)
#         additive_noise_std: std dev for Gaussian noise (applied at runtime)
#         stft_settings: no effect here because STFT is already done offline,
#                        but we keep the parameter for compatibility
#         normalization: 'none', 'db', or 'linear' if you still want to do it at runtime.
#         global_stats: dictionary containing e.g. 'db_mean', 'db_std', 'linear_min', 'linear_max'.
#         force_dim: if we need to force the time dimension to a specific size
#         """
#         self.file_paths = file_paths
#         self.transform = transform
#         self.additive_noise_std = additive_noise_std
#         self.normalization = normalization
#         self.global_stats = global_stats
#         self.force_dim = force_dim

#         if self.normalization not in ['linear', 'db', 'none']:
#             raise ValueError(f"Unsupported normalization type: {self.normalization}")

#         if self.normalization != 'none' and self.global_stats is None:
#             raise ValueError("global_stats must be provided for runtime normalization.")

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         # (1) Load .npz
#         npz_path = self.file_paths[idx]
#         # data = np.load(npz_path)
#         data = np.load(npz_path, allow_pickle=True)

#         spectrogram = data['spectrogram']  # shape: (freq_bins, time_bins) float32
#         label = data['label'].item()       # or data['label'].astype(int), depending on how you saved

#         # (2) Optional: runtime normalization (if you didn't do it in preprocessing)
#         if self.normalization == 'db':
#             # The spectrogram is presumably in linear scale if you didn't do preproc normalization.
#             # Convert to dB, then standardize. But typically you'd do it offline.
#             magnitude_db = 20 * np.log10(spectrogram + 1e-10)
#             mean_db = self.global_stats['db_mean']
#             std_db  = self.global_stats['db_std']
#             magnitude_db_normalized = (magnitude_db - mean_db) / std_db
#             magnitude_db_normalized = np.clip(magnitude_db_normalized, -3.0, 3.0)
#             spectrogram = (magnitude_db_normalized + 3.0) / 6.0  # map [-3, 3] -> [0, 1]
#         elif self.normalization == 'linear':
#             # min/max scale
#             min_val = self.global_stats['linear_min']
#             max_val = self.global_stats['linear_max']
#             spectrogram = np.clip(spectrogram, min_val, max_val)
#             spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-10)

#         # (3) Optional: additive noise
#         if self.additive_noise_std > 0:
#             noise = np.random.normal(0, self.additive_noise_std, size=spectrogram.shape).astype(np.float32)
#             spectrogram = spectrogram + noise
#             # If the data is assumed to be in [0, 1], you can clip again
#             spectrogram = np.clip(spectrogram, 0.0, 1.0)

#         # (4) transform if any
#         if self.transform is not None:
#             spectrogram = self.transform(spectrogram)

#         # (5) to tensor
#         spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0)  # shape: (1, freq_bins, time_bins)
#         label_tensor = torch.tensor(label, dtype=torch.long)

#         # (6) optionally force dimension
#         if self.force_dim is not None:
#             _, _, X = spectrogram_tensor.size()
#             if self.force_dim > X:
#                 repeat_factor = (self.force_dim + X - 1) // X
#                 spectrogram_tensor = spectrogram_tensor.repeat(1, 1, repeat_factor)[..., :self.force_dim]

#         return spectrogram_tensor, label_tensor
# def get_dataloaders(
#     folds_json_path, 
#     fold_index=0, 
#     train_split_percentage=0.8, 
#     batch_size_train=32, 
#     batch_size_val=32, 
#     batch_size_test=32,
#     normalization='none',
#     shuffle_train=True, 
#     shuffle_val=False, 
#     shuffle_test=False,
#     additive_noise_std_train=0.0,
#     additive_noise_std_val=0.0,
#     additive_noise_std_test=0.0,
#     global_stats='none',
#     force_dim=None,
#     num_workers_train=0,
#     num_workers_val=0,
#     num_workers_test=0,
#     stft_settings=None
# ):
#     """
#     Given the path to a JSON with the k folds, pick one fold as test,
#     the other folds as train. Then from train, take a fraction for
#     train vs. validation, create Datasets and DataLoaders.

#     Parameters:
#         folds_json_path (str): Path to the JSON file containing fold splits.
#         fold_index (int): Index of the fold to use as the test set (0-based).
#         train_split_percentage (float): Percentage of training data to use for training (rest for validation).
#         batch_size_train (int): Batch size for training DataLoader.
#         batch_size_val (int): Batch size for validation DataLoader.
#         batch_size_test (int): Batch size for test DataLoader.
#         shuffle_train (bool): Whether to shuffle the training DataLoader.
#         shuffle_val (bool): Whether to shuffle the validation DataLoader.
#         shuffle_test (bool): Whether to shuffle the test DataLoader.
#         additive_noise_std_train (float): Std dev for additive noise in train set.
#         additive_noise_std_val (float): Std dev for additive noise in validation set.
#         additive_noise_std_test (float): Std dev for additive noise in test set.
#         num_workers_train (int): Number of workers for training DataLoader.
#         num_workers_val (int): Number of workers for validation DataLoader.
#         num_workers_test (int): Number of workers for test DataLoader.
#         stft_settings (dict): Dictionary containing 'window_size' and 'overlap' for STFT.

#     Returns:
#         train_loader (DataLoader): DataLoader for the training set.
#         val_loader (DataLoader): DataLoader for the validation set.
#         test_loader (DataLoader): DataLoader for the test set.
#     """
#     # 1) Load the fold info
#     with open(folds_json_path, "r") as f:
#         folds = json.load(f)

#     if fold_index < 0 or fold_index >= len(folds):
#         raise ValueError(f"fold_index {fold_index} is out of range. Number of folds: {len(folds)}")

#     # 2) Extract test files from the chosen fold, train files from the rest
#     test_files = folds[fold_index]["test"]
#     train_files = []
#     for i, fold in enumerate(folds):
#         if i != fold_index:
#             train_files.extend(fold["test"])  # Since in `five_fold_split.py`, "test" contains fold-specific test files
#     # 3) Split train_files -> train + val
#     random_seed = 42  # For reproducibility
#     random.Random(random_seed).shuffle(train_files)
#     split_idx = int(len(train_files) * train_split_percentage)
#     train_subset = train_files[:split_idx]
#     val_subset = train_files[split_idx:]

#     # 4) Create Datasets
#     train_dataset = RadarDataset(
#         file_paths=convert_paths_pathlib(train_subset),
#         transform=None,  # or some transform
#         additive_noise_std=additive_noise_std_train,
#         normalization=normalization,
#         stft_settings=stft_settings,
#         global_stats=global_stats,
#         force_dim=force_dim
#     )
#     val_dataset = RadarDataset(
#         file_paths=convert_paths_pathlib(val_subset),
#         transform=None,
#         additive_noise_std=additive_noise_std_val,
#         normalization=normalization,
#         stft_settings=stft_settings,
#         global_stats=global_stats,
#         force_dim=force_dim
#     )
#     test_dataset = RadarDataset(
#         file_paths=convert_paths_pathlib(test_files),
#         transform=None,
#         additive_noise_std=additive_noise_std_test,
#         normalization=normalization,
#         stft_settings=stft_settings,
#         global_stats=global_stats,
#         force_dim=force_dim
#     )

#     # 5) Create DataLoaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size_train, 
#         shuffle=shuffle_train, 
#         num_workers=num_workers_train,
#         pin_memory=True,
#         worker_init_fn=worker_init_fn
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size_val, 
#         shuffle=shuffle_val, 
#         num_workers=num_workers_val,
#         pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size_test, 
#         shuffle=shuffle_test, 
#         num_workers=num_workers_test,
#         pin_memory=True
#     )

#     return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# def load_config(config_path):
#     """
#     Load YAML configuration file.

#     Parameters:
#         config_path (str): Path to the YAML config file.

#     Returns:
#         config (dict): Dictionary containing configuration parameters.
#     """
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# from pathlib import Path
# def convert_paths_pathlib(paths):
#     new_paths = []
#     for path_str in paths:
#         path = Path(path_str)
        
#         # Replace the specific directory with 'preprocessed'
#         # Assuming '1 December 2017 Dataset' is always the parent directory
#         new_dir = path.parent.parent / 'preprocessed'
        
#         # Ensure the new directory exists (optional)
#         new_dir.mkdir(parents=True, exist_ok=True)
        
#         # Change the file extension from .dat to .npz
#         new_filename = path.stem + '.npz'
        
#         # Combine the new directory and new filename
#         new_full_path = new_dir / new_filename
#         new_paths.append(str(new_full_path))
    
#     return new_paths

# if __name__ == "__main__":
#     """
#     Example usage:
#       python dataset_and_dataloader.py --config config.yaml
#     """
#     parser = argparse.ArgumentParser(description="Radar Dataset and Dataloader Script")
#     parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
#     args = parser.parse_args()

#     # Load configuration
#     config = load_config(args.config)

#     # Extract STFT settings
#     stft_settings = config.get('stft', {'window_size': 128, 'overlap': 64})

#     if normalization == 'none' or normalization ==  'db':
#         global_stats = load_config("/project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/global_stats.json")
#         print(global_stats)

#     # Retrieve DataLoaders
#     train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
#         folds_json_path=config['folds_json'],
#         fold_index=config['fold_index'],
#         train_split_percentage=config['train_split_percentage'],
#         batch_size_train=config['batch_sizes']['train'],
#         batch_size_val=config['batch_sizes']['validation'],
#         batch_size_test=config['batch_sizes']['test'],
#         shuffle_train=config['shuffle']['train'],
#         shuffle_val=config['shuffle']['validation'],
#         shuffle_test=config['shuffle']['test'],
#         additive_noise_std_train=config['additive_noise_std']['train'],
#         additive_noise_std_val=config['additive_noise_std']['validation'],
#         additive_noise_std_test=config['additive_noise_std']['test'],
#         normalization=config['normalization'],
#         global_stats=global_stats,
#         force_dim=config['force_dim'],
#         num_workers_train=config['num_workers']['train'],
#         num_workers_val=config['num_workers']['validation'],
#         num_workers_test=config['num_workers']['test'],
#         stft_settings=stft_settings
#     )

#     print(f"Fold Index: {config['fold_index']}")
#     print(f"Number of training batches: {len(train_loader)}")
#     print(f"Number of validation batches: {len(val_loader)}")
#     print(f"Number of test batches: {len(test_loader)}")

#     # Example iteration
#     for i, (batch_data, batch_labels) in enumerate(train_loader):
#         print("Batch index:", i, "Data shape:", batch_data.shape, "Labels:", batch_labels)
#         # do something...
#         break

def worker_init_fn(worker_id):
    # Option A: Use your own scheme (seed + worker_id)
    worker_seed = (seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # If using GPU ops in worker (rare), you might also do:
    torch.cuda.manual_seed_all(worker_seed)




# dataset.py

import os
import json
import random
import yaml
import torch
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader

# -------------
# Helper Functions
# -------------
def parse_filename_get_label(filename):
    """
    Same as before, but now 'filename' points to .npz (not .dat).
    We'll reuse the logic (look at the first character).
    """
    base = os.path.basename(filename)
    # Remove the .npz extension if needed
    base_no_ext, _ = os.path.splitext(base)
    label_char = base_no_ext[0]
    label_map = {
        '1': 0,  # walking
        '2': 1,  # sitting down
        '3': 2,  # stand up
        '4': 3,  # pick up an object
        '5': 4,  # drink water
        '6': 5,  # fall
    }
    return label_map.get(label_char, -1)  # -1 if not found

# -------------
# The RadarDataset Class
# -------------
class RadarDataset(Dataset):
    def __init__(
        self, 
        file_paths,              # list of .npz file paths
        transform=None, 
        additive_noise_std=0.0, 
        stft_settings=None,      # <--- no longer used here, since STFT is done offline
        normalization='none',    # if 'none', data is used as-is; if 'db' or 'linear', see note below
        global_stats=None,       # used only if we still want to do runtime normalization
        force_dim=None
    ):
        """
        file_paths: list of .npz files
        transform: optional transform (e.g. a function or lambda)
        additive_noise_std: std dev for Gaussian noise (applied at runtime)
        stft_settings: no effect here because STFT is already done offline,
                       but we keep the parameter for compatibility
        normalization: 'none', 'db', or 'linear' if you still want to do it at runtime.
        global_stats: dictionary containing e.g. 'db_mean', 'db_std', 'linear_min', 'linear_max'.
        force_dim: if we need to force the time dimension to a specific size
        """
        self.file_paths = file_paths
        self.transform = transform
        self.additive_noise_std = additive_noise_std
        self.normalization = normalization
        self.global_stats = global_stats
        self.force_dim = force_dim

        if self.normalization not in ['linear', 'db', 'none']:
            raise ValueError(f"Unsupported normalization type: {self.normalization}")

        if self.normalization != 'none' and self.global_stats is None:
            raise ValueError("global_stats must be provided for runtime normalization.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # (1) Load .npz
        npz_path = self.file_paths[idx]
        data = np.load(npz_path, allow_pickle=True)

        spectrogram = data['spectrogram']  # shape: (freq_bins, time_bins) float32
        label = data['label'].item()       # or data['label'].astype(int), depending on how you saved

        # (2) Optional: runtime normalization (if you didn't do it in preprocessing)
        if self.normalization == 'db':
            # The spectrogram is presumably in linear scale if you didn't do preproc normalization.
            # Convert to dB, then standardize. But typically you'd do it offline.
            magnitude_db = 20 * np.log10(spectrogram + 1e-10)
            mean_db = self.global_stats['db_mean']
            std_db  = self.global_stats['db_std']
            magnitude_db_normalized = (magnitude_db - mean_db) / std_db
            magnitude_db_normalized = np.clip(magnitude_db_normalized, -3.0, 3.0)
            spectrogram = (magnitude_db_normalized + 3.0) / 6.0  # map [-3, 3] -> [0, 1]
        elif self.normalization == 'linear':
            # min/max scale
            min_val = self.global_stats['linear_min']
            max_val = self.global_stats['linear_max']
            spectrogram = np.clip(spectrogram, min_val, max_val)
            spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-10)

        # (3) Optional: additive noise
        if self.additive_noise_std > 0:
            noise = np.random.normal(0, self.additive_noise_std, size=spectrogram.shape).astype(np.float32)
            spectrogram = spectrogram + noise
            # If the data is assumed to be in [0, 1], you can clip again
            spectrogram = np.clip(spectrogram, 0.0, 1.0)

        # (4) transform if any
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        # (5) to tensor
        spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0)  # shape: (1, freq_bins, time_bins)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # (6) optionally force dimension
        if self.force_dim is not None:
            _, _, X = spectrogram_tensor.size()
            if self.force_dim > X:
                repeat_factor = (self.force_dim + X - 1) // X
                spectrogram_tensor = spectrogram_tensor.repeat(1, 1, repeat_factor)[..., :self.force_dim]

        return spectrogram_tensor, label_tensor

# def worker_init_fn(worker_id):
#     # Note: 'seed' should be passed or made global if used here
#     # Assuming 'seed' is defined globally or passed appropriately
#     # For example, you can modify the function to accept 'seed' as an argument
#     pass  # Implement if necessary

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

    Parameters:
        folds_json_path (str): Path to the JSON file containing fold splits.
        fold_index (int): Index of the fold to use as the test set (0-based).
        train_split_percentage (float): Percentage of training data to use for training (rest for validation).
        batch_size_train (int): Batch size for training DataLoader.
        batch_size_val (int): Batch size for validation DataLoader.
        batch_size_test (int): Batch size for test DataLoader.
        shuffle_train (bool): Whether to shuffle the training DataLoader.
        shuffle_val (bool): Whether to shuffle the validation DataLoader.
        shuffle_test (bool): Whether to shuffle the test DataLoader.
        additive_noise_std_train (float): Std dev for additive noise in train set.
        additive_noise_std_val (float): Std dev for additive noise in validation set.
        additive_noise_std_test (float): Std dev for additive noise in test set.
        num_workers_train (int): Number of workers for training DataLoader.
        num_workers_val (int): Number of workers for validation DataLoader.
        num_workers_test (int): Number of workers for test DataLoader.
        stft_settings (dict): Dictionary containing 'window_size' and 'overlap' for STFT.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    # 1) Load the fold info
    with open(folds_json_path, "r") as f:
        folds = json.load(f)

    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(f"fold_index {fold_index} is out of range. Number of folds: {len(folds)}")

    # 2) Extract test files from the chosen fold, train files from the rest
    test_files = folds[fold_index]["test"]
    train_files = []
    for i, fold in enumerate(folds):
        if i != fold_index:
            train_files.extend(fold["test"])  # Since in `five_fold_split.py`, "test" contains fold-specific test files

    # 3) Split train_files -> train + val
    random_seed = 42  # For reproducibility
    random.Random(random_seed).shuffle(train_files)
    split_idx = int(len(train_files) * train_split_percentage)
    train_subset = train_files[:split_idx]
    val_subset = train_files[split_idx:]

    # 4) Create Datasets
    train_dataset = RadarDataset(
        file_paths=convert_paths_pathlib(train_subset),
        transform=None,  # or some transform
        additive_noise_std=additive_noise_std_train,
        normalization=normalization,
        stft_settings=stft_settings,
        global_stats=global_stats,
        force_dim=force_dim
    )
    val_dataset = RadarDataset(
        file_paths=convert_paths_pathlib(val_subset),
        transform=None,
        additive_noise_std=additive_noise_std_val,
        normalization=normalization,
        stft_settings=stft_settings,
        global_stats=global_stats,
        force_dim=force_dim
    )
    test_dataset = RadarDataset(
        file_paths=convert_paths_pathlib(test_files),
        transform=None,
        additive_noise_std=additive_noise_std_test,
        normalization=normalization,
        stft_settings=stft_settings,
        global_stats=global_stats,
        force_dim=force_dim
    )

    # 5) Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=shuffle_train, 
        num_workers=num_workers_train,
        pin_memory=True,
        worker_init_fn=worker_init_fn
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

from pathlib import Path
def convert_paths_pathlib(paths):
    new_paths = []
    for path_str in paths:
        path = Path(path_str)
        
        # Replace the specific directory with 'preprocessed'
        # Assuming '1 December 2017 Dataset' is always the parent directory
        new_dir = path.parent.parent / 'preprocessed'
        
        # Ensure the new directory exists (optional)
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Change the file extension from .dat to .npz
        new_filename = path.stem + '.npz'
        
        # Combine the new directory and new filename
        new_full_path = new_dir / new_filename
        new_paths.append(str(new_full_path))
    
    return new_paths

if __name__ == "__main__":
    """
    Example usage:
      python dataset.py --config config.yaml
    """
    parser = argparse.ArgumentParser(description="Radar Dataset and Dataloader Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract STFT settings
    stft_settings = config.get('stft', {'window_size': 128, 'overlap': 64})

    # Corrected the reference to normalization
    if config['normalization'] in ['none', 'db']:
        global_stats = load_config("/project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/global_stats.json")
        print(global_stats)
    elif config['normalization'] == 'linear':
        global_stats = load_config("/project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/global_stats.json")
        print(global_stats)
    else:
        global_stats = 'none'

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
        break  # Remove this to iterate through the entire DataLoader
