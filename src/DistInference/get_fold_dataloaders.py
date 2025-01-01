import argparse
import yaml
from dataset import get_dataloaders

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

def get_dataloaders_from_config(config_path):
    """
    Get DataLoaders based on the provided configuration file.

    Parameters:
        config_path (str): Path to the YAML config file.

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects.
    """
    # Load configuration
    config = load_config(config_path)

    # Extract STFT settings
    print(config)
    stft_settings = config['stft']

    if config['normalization'] == 'linear' or config['normalization'] ==  'db':
        global_stats = load_config("global_stats.json")
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
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, config['fold_index']

def main():
    parser = argparse.ArgumentParser(description="Get DataLoaders for a specific fold using configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # Get DataLoaders and fold index
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, fold_index = get_dataloaders_from_config(args.config)

    print(f"Fold Index: {fold_index}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Example iteration
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        print("Batch index:", i, "Data shape:", batch_data.shape, "Labels:", batch_labels)
        # Perform training step...
        break  # Remove this to iterate through the entire DataLoader

if __name__ == "__main__":
    main()