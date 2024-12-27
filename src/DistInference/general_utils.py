import torch
import numpy as np
from scipy.stats import entropy
import sys

def analyze_input(input_variable, description=None):
    if description is None:
        description = 'No Description Passed!'
    
    if isinstance(input_variable, torch.Tensor):
        num_elements = input_variable.numel()
        dtype_size = input_variable.element_size()  # Size of each element in bytes
        total_size_bytes = num_elements * dtype_size
        
        unique_values = torch.unique(input_variable)
        num_unique_values = unique_values.numel()
        
        # Calculate entropy
        if input_variable.dtype in (torch.int32, torch.int64, torch.uint8, torch.int16):
            counts = torch.bincount(input_variable.flatten().int())
            probs = counts / num_elements
            entropy_value = entropy(probs.cpu().numpy(), base=2) if num_elements > 1 else 0
        else:
            entropy_value = str("Only Defined for Integers!")

        print(f"{description}:")
        print(f"  - Shape: {input_variable.shape}")
        print(f"  - Dtype: {input_variable.dtype}")
        if input_variable.numel()>0:
            print(f"  - Value range: {input_variable.min().item()} to {input_variable.max().item()}")
            print(f"  - Unique values as a fraction of total: {num_unique_values / num_elements:.4f}")
        print(f"  - Number of elements: {num_elements}")
        print(f"  - Number of unique values: {num_unique_values}")
        print(f"  - Entropy: {entropy_value:} bits per element")
        print(f"  - A few sample values: {input_variable.flatten()[:5].tolist()}")
        print(f"  - Size: {total_size_bytes / (1024 ** 2):.4f} MB")
        print()
    
    elif isinstance(input_variable, np.ndarray):
        num_elements = input_variable.size
        dtype_size = input_variable.itemsize  # Size of each element in bytes
        total_size_bytes = num_elements * dtype_size
        
        unique_values = np.unique(input_variable)
        num_unique_values = unique_values.size
        
        # Calculate entropy
        counts = np.bincount(input_variable.flatten().astype(int))
        probs = counts / num_elements
        entropy_value = entropy(probs, base=2) if num_elements > 1 else 0

        print(f"{description}:")
        print(f"  - Shape: {input_variable.shape}")
        print(f"  - Dtype: {input_variable.dtype}")
        print(f"  - Value range: {input_variable.min()} to {input_variable.max()}")
        print(f"  - Number of elements: {num_elements}")
        print(f"  - Number of unique values: {num_unique_values}")
        print(f"  - Unique values as a fraction of total: {num_unique_values / num_elements:.4f}")
        print(f"  - Entropy: {entropy_value:.4f} bits per element")
        print(f"  - A few sample values: {input_variable.flatten()[:5].tolist()}")
        print(f"  - Size: {total_size_bytes / (1024 ** 2):.4f} MB")
        print()
    
    elif isinstance(input_variable, list):
        num_elements = len(input_variable)
        
        # Check if the list contains tensors
        if num_elements > 0 and isinstance(input_variable[0], torch.Tensor):
            # Handle list of tensors
            total_size_bytes = sum(tensor.nelement() * tensor.element_size() for tensor in input_variable)
            device = input_variable[0].device
            dtype = input_variable[0].dtype
            shapes = [tensor.shape for tensor in input_variable]
            
            print(f"{description}:")
            print(f"  - Type: list of tensors")
            print(f"  - Number of tensors: {num_elements}")
            print(f"  - Device: {device}")
            print(f"  - Dtype: {dtype}")
            print(f"  - Shapes: {shapes}")
            if num_elements > 0:
                print(f"  - First tensor sample values: {input_variable[0].flatten()[:5].tolist()}")
            print(f"  - Total Size: {total_size_bytes / (1024 ** 2):.4f} MB")
        else:
            # Handle regular list (non-tensor)
            try:
                dtype_size = np.array(input_variable).dtype.itemsize if num_elements > 0 else 0
                total_size_bytes = num_elements * dtype_size
            except:
                # If conversion to numpy array fails, use a rough estimate
                total_size_bytes = sys.getsizeof(input_variable)
            
            print(f"{description}:")
            print(f"  - Type: list")
            print(f"  - Length: {num_elements}")
            if num_elements > 0:
                print(f"  - Value range: {min(input_variable)} to {max(input_variable)}")
                print(f"  - A few sample values: {input_variable[:5]}")
            print(f"  - Estimated Size: {total_size_bytes / (1024 ** 2):.4f} MB")
        
        print()
    
    elif isinstance(input_variable, bytes):
        length_bytes = len(input_variable)
        length_bits = length_bytes * 8  # Convert bytes to bits
        
        # Calculate entropy
        counts = np.bincount(np.frombuffer(input_variable, dtype=np.uint8))
        probs = counts / length_bytes
        entropy_value = entropy(probs, base=2) if length_bytes > 1 else 0

        print(f"{description}:")
        print(f"  - Type: bytes")
        print(f"  - Length: {length_bytes} bytes")
        print(f"  - Equivalent bits: {length_bits} bits")
        print(f"  - Entropy: {entropy_value:.4f} bits per element")
        print(f"  - A few sample bytes: {list(input_variable[:5])}")
        print()

    elif isinstance(input_variable, str):
        length_bits = len(input_variable)
        length_bytes = length_bits // 8

        print(f"{description}:")
        print(f"  - Type: bit sequence (string)")
        print(f"  - Length: {length_bits} bits")
        print(f"  - Equivalent bytes: {length_bytes} bytes")
        print(f"  - A few sample bits: {input_variable[:40]}")  # Show first 40 bits for brevity
        print()
        
    elif isinstance(input_variable, dict):
        num_elements = len(input_variable)
        
        # Estimate size (this is a rough estimate and may not be accurate for all types of dictionary values)
        total_size_bytes = sys.getsizeof(input_variable)
        
        key_types = set(type(k).__name__ for k in input_variable.keys())
        value_types = set(type(v).__name__ for v in input_variable.values())

        print(f"{description}:")
        print(f"  - Type: dictionary")
        print(f"  - Number of key-value pairs: {num_elements}")
        print(f"  - Key types: {', '.join(key_types)}")
        print(f"  - Value types: {', '.join(value_types)}")
        if num_elements > 0:
            sample_keys = list(input_variable.keys())[:3]
            print(f"  - A few sample keys: {sample_keys}")
        print(f"  - Estimated Size: {total_size_bytes / (1024 ** 2):.4f} MB")
        print()
        
def histogram_plotter(tensor, bins=50, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plot a histogram of the values in a PyTorch tensor.
    
    Parameters:
        tensor (torch.Tensor): The input tensor.
        bins (int): Number of bins for the histogram. Default is 50.
        title (str): Title of the histogram plot. Default is "Tensor Histogram".
        xlabel (str): Label for the x-axis. Default is "Value".
        ylabel (str): Label for the y-axis. Default is "Frequency".
    """
    # Flatten the tensor to 1D
    tensor_flat = tensor.flatten().cpu().numpy()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(tensor_flat, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()



import torch
import torch.nn as nn
from typing import Optional

def count_parameters(module: nn.Module) -> tuple:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_grad_status(module: nn.Module) -> str:
    if any(p.requires_grad for p in module.parameters()):
        return "grad_ON" if all(p.requires_grad for p in module.parameters()) else "grad_PARTIAL"
    return "grad_OFF"

def print_module_tree(
    module: nn.Module,
    max_depth: int = -1,
    prefix: str = '',
    is_last: bool = True,
    module_name: Optional[str] = None,
    current_depth: int = 0
) -> None:
    """
    Recursively prints the tree structure of PyTorch modules up to a specified depth,
    including detailed parameter count and gradient status.
    
    Args:
    - module (nn.Module): The PyTorch module to print
    - max_depth (int): Maximum depth to print. -1 means no limit.
    - prefix (str): The current prefix string (used for recursion)
    - is_last (bool): Whether this is the last child of its parent
    - module_name (Optional[str]): The name of the current module (if any)
    - current_depth (int): The current depth in the module tree
    """
    connector = "└── " if is_last else "├── "
    new_prefix = prefix + ("    " if is_last else "│   ")

    total_params, trainable_params = count_parameters(module)
    grad_status = get_grad_status(module)
    
    if grad_status == "grad_PARTIAL":
        module_info = (f"{module.__class__.__name__} "
                       f"(total: {total_params:.1e}, "
                       f"trainable: {trainable_params:.1e}, "
                       f"non-trainable: {total_params - trainable_params:.1e} params, "
                       f"{grad_status})")
    else:
        module_info = f"{module.__class__.__name__} ({total_params:.1e} params, {grad_status})"
    
    if module_name:
        print(f"{prefix}{connector}{module_name}: {module_info}")
    else:
        print(f"{prefix}{connector}{module_info}")
    
    if max_depth != -1 and current_depth >= max_depth:
        if list(module.children()):
            print(f"{new_prefix}...")
        return

    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_module_tree(
            child,
            max_depth,
            new_prefix,
            is_last_child,
            name,
            current_depth + 1
        )

def print_module_summary(
    module: nn.Module,
    depth: int = -1
) -> None:
    """
    Prints a summary of the module tree with detailed parameter counts and gradient status.
    
    Args:
    - module (nn.Module): The PyTorch module to summarize
    - max_depth (int): Maximum depth to print. -1 means no limit.
    """
    print(f"Module Summary (depth={depth}):")
    print_module_tree(module, depth)
    
    total_params, trainable_params = count_parameters(module)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")


import torch.nn as nn

def set_module_grad(model: nn.Module, module_names: list, requires_grad: bool = False):
    """
    Sets the requires_grad attribute for specified modules in the model in-place.
    This function can freeze specific modules in an unfrozen model,
    or unfreeze specific modules in a frozen model.
    
    Args:
    - model (nn.Module): The model containing the modules to be modified
    - module_names (list): List of module names to be frozen/unfrozen (e.g., ["encoder", "decoder"])
    - requires_grad (bool): If True, enables gradients. If False, disables gradients (freezes the module)
    
    Returns:
    - model (nn.Module): The modified model
    - modified (list): List of successfully modified module names
    """
    modified = []

    def set_grad_recursive(module, name, should_modify):
        for param in module.parameters(recurse=False):
            if should_modify:
                param.requires_grad = requires_grad
            else:
                param.requires_grad = not requires_grad
        for child_name, child_module in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            child_should_modify = should_modify or any(full_child_name.startswith(m) for m in module_names)
            set_grad_recursive(child_module, full_child_name, child_should_modify)

    # Apply the gradient setting recursively from the root
    set_grad_recursive(model, "", False)

    # Check which modules were modified
    for name in module_names:
        try:
            module = model
            for part in name.split('.'):
                module = getattr(module, part)
            
            if any(param.requires_grad == requires_grad for param in module.parameters()):
                modified.append(name)
                print(f"{'Unfrozen' if requires_grad else 'Frozen'}: {name}")
            else:
                print(f"Module {name} was already {'unfrozen' if requires_grad else 'frozen'}")
        except AttributeError:
            print(f"Module not found: {name}")

    # Print overall model grad state
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model overall: {trainable_params}/{total_params} parameters are trainable")

    return model, modified

import zlib
class IndicesTX:
    def __init__(self, quant_bits=13, is_compress=None, bit_error_rate=None, seed=42):
        # Remove the super().__init__() call
        self.quant_bits = quant_bits  # Uncomment this line
        self.bit_error_rate = bit_error_rate
        self.seed = seed
        self.is_compress = is_compress
        self.quantizer = integer_quantizer()

    def transmit(self, indices):
        bit_seq = self.quantizer.quantize(indices, quant_bits=self.quant_bits)
        N_bits = len(bit_seq)
        if self.is_compress is not None:
            byte_seq = bit_string_to_bytes(bit_seq)
            byte_seq_compressed = zlib.compress(byte_seq)
            byte_seq_decompressed = zlib.decompress(byte_seq_compressed)
            bit_seq_tx = bytes_to_bit_string(byte_seq_decompressed)
            N_bits_compressed = 8 * len(byte_seq_compressed)
        else:
            bit_seq_tx = bit_seq
            N_bits_compressed = N_bits
            
        if self.bit_error_rate is not None:
            bit_seq_rx = flip_bits(bit_seq_tx, flip_ratio=self.bit_error_rate, seed=self.seed)
        else:
            bit_seq_rx = bit_seq_tx
            
        indices_hat = self.quantizer.dequantize(bit_seq_rx, quant_bits=self.quant_bits)
        return indices_hat, N_bits, N_bits_compressed

def flip_bits_in_bytes_seq(data_bytes, ber):
    """
    Simulates the transmission of a bytes sequence over an AWGN channel with a given BER.

    Parameters:
    data_bytes (bytes): The original byte sequence to be transmitted.
    ber (float): The bit error rate (probability of a bit being flipped).

    Returns:
    bytes: The byte sequence after transmission with noise.
    """
    # Convert bytes to a binary string
    data_bits = ''.join(f'{byte:08b}' for byte in data_bytes)  # Each byte as an 8-bit binary string

    # Convert the binary string into a list of characters for easy manipulation
    noisy_bits = list(data_bits)

    # Total number of bits
    total_bits = len(noisy_bits)

    # Determine how many bits to flip based on BER
    num_flips = np.random.binomial(n=total_bits, p=ber)

    # Randomly choose which bits to flip
    flip_indices = np.random.choice(total_bits, num_flips, replace=False)

    # Flip the selected bits
    for idx in flip_indices:
        noisy_bits[idx] = '0' if noisy_bits[idx] == '1' else '1'

    # Reassemble the noisy bits back into bytes
    noisy_bits_string = ''.join(noisy_bits)

    # Convert the binary string back to bytes
    noisy_data_bytes = bytes(int(noisy_bits_string[i:i+8], 2) for i in range(0, len(noisy_bits_string), 8))

    return noisy_data_bytes
    
def flip_bits(bit_sequence: str, flip_ratio: float, seed: int = 42) -> str:
    # Set the seed for reproducibility
    random.seed(seed)

    # Validate input
    if not all(bit in '01' for bit in bit_sequence):
        raise ValueError("Input must be a string of 0s and 1s")
    if not 0 <= flip_ratio <= 1:
        raise ValueError("Flip ratio must be between 0 and 1")
    
    # Convert string to list for easier manipulation
    bits = list(bit_sequence)
    
    # Calculate number of bits to flip
    num_bits_to_flip = int(len(bits) * (flip_ratio))
    
    # Randomly select indices to flip
    indices_to_flip = random.sample(range(len(bits)), num_bits_to_flip)
    
    # Flip selected bits
    for index in indices_to_flip:
        bits[index] = '0' if bits[index] == '1' else '1'
    
    # Convert back to string and return
    return ''.join(bits)
    
import os
import shutil

def save_experiment_files(timestamp, target_dir, log_dir):
    # Create a folder named timestamp in log_dir
    # timestamp_folder = os.path.join(log_dir, timestamp)
    ckpt_folder = os.path.join(log_dir, 'checkpoint')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)

    # List of file extensions to copy
    extensions = ['py', 'yaml']

    # Iterate over all files in the target directory
    for file_name in os.listdir(target_dir):
        source_path = os.path.join(target_dir, file_name)
        # Check if the file has one of the desired extensions
        if os.path.isdir(source_path):
            # Only copy if it is named 'configs'
            if file_name == 'configs':
                target_path = os.path.join(log_dir, file_name)
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        # Otherwise, check if the file has one of the desired extensions
        elif any(file_name.endswith(f'.{ext}') for ext in extensions):
            target_path = os.path.join(log_dir, file_name)
            shutil.copy2(source_path, target_path)

    # print(f"Files copied to {timestamp_folder}")

import pickle
def save_dict_pickle(dictionary, experiment_name, base_path="/project_ghent/Mostafa/image/ImageTransmission/notebooks/tmp"):
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{experiment_name}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f)

def save_results_csv(results, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, results)) + '\n')


def fully_flatten_config(config):
    items = {}
    for k, v in config.items():
        if isinstance(v, dict):
            items.update(fully_flatten_config(v))
        else:
            items[k] = v
    return items

def initialize_wandb(YAML_config, timestamp):
    import wandb
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        wandb.login(key="abd082d717ec4951f1e4b8500e72172c5a1984e3")
        # Start a new W&B run
        dict_config = fully_flatten_config(YAML_config)
        wandb.init(project=YAML_config['wandb_settings']['project_name'], name=timestamp, config=dict_config)
    
        # # Log any file (optional)
        # wandb.save(log_dir)


import random
def set_seeds(seed, device='cuda'):
    np.random.seed(seed)
    
    if device=='cuda':
        torch.manual_seed(seed)
    random.seed(seed)
    
    # If using CUDA
    if device=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bit_string_to_bytes(bit_string: str) -> bytes:
    # Ensure the length is a multiple of 8 by padding with zeros
    padded_bit_string = bit_string.zfill((8 - len(bit_string) % 8) % 8 + len(bit_string))
    
    # Convert each group of 8 bits to a byte and combine into a bytes object
    byte_sequence = bytes(int(padded_bit_string[i:i + 8], 2) for i in range(0, len(padded_bit_string), 8))
    
    return byte_sequence
def bytes_to_bit_string(byte_sequence: bytes) -> str:
    # Convert each byte to an 8-bit binary string and concatenate them
    bit_string = ''.join(f'{byte:08b}' for byte in byte_sequence)
    return bit_string



import matplotlib.pyplot as plt
import math
def grid_image_show(image_list, grid_shape=None, titles=None, scale=1.0):
    if grid_shape is None:
        total_images = len(image_list)
        grid_n = math.ceil(math.sqrt(total_images))
        grid_shape = (grid_n, grid_n)
    
    rows, cols = grid_shape
    base_size = 15  # Base size in inches
    fig = plt.figure(figsize=(base_size * scale, base_size * rows / cols * scale))
    
    for i, img in enumerate(image_list):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        if len(img.shape) == 4:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            if img.shape[0] == 1 or img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        elif len(img.shape) != 2:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None, aspect='equal')
        ax.axis('off')
        
        if titles is not None:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()