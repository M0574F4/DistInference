############################################################
# evaluation_utils.py
############################################################

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# External utilities (make sure these imports work in your environment)
from DistInference.general_utils import load_config
from DistInference.get_fold_dataloaders import get_dataloaders_from_config
from DistInference.models import My_Model

############################################################
# 1. Utility Functions
############################################################
def set_seed(seed: int) -> None:
    """
    Set the seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels, num_classes):
    """
    Compute accuracy and row-normalized confusion matrix (in percentages).

    Each row of the confusion matrix sums to 100.0.
    
    Args:
        preds (list or np.array): Predicted labels.
        labels (list or np.array): True labels.
        num_classes (int): Number of classes.

    Returns:
        accuracy (float): Accuracy score.
        cm (np.array): Confusion matrix, where each row sums to 100.0.
    """
    # Accuracy
    accuracy = np.mean(np.array(preds) == np.array(labels))

    # Absolute confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes))).astype(float)

    # Row-normalize to get percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for any row that might be all zeros
    row_sums[row_sums == 0] = 1.0  
    cm = (cm / row_sums) * 100.0
    
    return accuracy, cm


def plot_and_save_normalized_confusion_matrix(
    cm, 
    class_names, 
    title, 
    filename=None, 
    line_width=1
):
    """
    Plot, display, and optionally save the confusion matrix, 
    which is assumed to already be row-normalized in percentage form.
    
    Args:
        cm (np.array): Confusion matrix with each row summing to 100.0.
        class_names (list): List of class names.
        title (str): Title for the plot.
        filename (str): Filename to save the plot (if provided).
        line_width (float): Width of the lines in the heatmap.
    
    Returns:
        matplotlib.figure.Figure: The figure object of the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6));
    sns.heatmap(
        cm, 
        annot=True,       # Show values in the cells
        fmt='.2f',        # Format to 2 decimal places
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        linecolor='black', 
        linewidths=line_width,
        ax=ax
    )
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names, rotation=45)
    plt.tight_layout()

    if filename:
        fig.savefig(filename)
    # plt.show()
    
    return fig;


############################################################
# 2. Centralized Evaluation
############################################################
def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluate the model on a given dataloader (centralized).

    Args:
        model (nn.Module): The pretrained model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): Device to run the evaluation on.
        num_classes (int): Number of classes.

    Returns:
        accuracy (float): Accuracy score.
        cm (np.array): Row-normalized confusion matrix (in %).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)

            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                raise KeyError("Model output does not contain 'logits' key.")

            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy, cm = compute_metrics(all_preds, all_labels, num_classes)
    return accuracy, cm


############################################################
# 3. Distributed Inference Models (Edge/Server Split)
############################################################
class EdgeModel(nn.Module):
    """
    Splits the original model into two parts:
    - Edge: partial backbone
    - The rest remains for the server
    """

    def __init__(self, original_model):
        super(EdgeModel, self).__init__()
        # Example assumes original_model.backbone is a ResNet-like architecture
        self.backbone1 = nn.Sequential(
            original_model.backbone.conv1,
            original_model.backbone.bn1,
            original_model.backbone.act1,
            original_model.backbone.maxpool,
            original_model.backbone.layer1,
            original_model.backbone.layer2,
            original_model.backbone.layer3,
            nn.Sequential(*list(original_model.backbone.layer4.children())[:1])  # e.g. layer4[0]
        )
        self.backbone2 = nn.Sequential(
            *list(original_model.backbone.layer4.children())[1:],  # layer4[1] onward
            original_model.backbone.global_pool,
            original_model.backbone.fc
        )

    def forward(self, x):
        x = self.backbone1(x)
        return self.backbone2(x)


class ServerModel(nn.Module):
    """
    Server-side model that receives the output from EdgeModel.
    """

    def __init__(self, original_model):
        super(ServerModel, self).__init__()
        self.classifier = original_model.classifier
        self.loss_fn = original_model.loss_fn  # if needed

    def forward(self, x):
        x = self.classifier(x)
        return x


############################################################
# 4. Quantization / Dequantization (Optional)
############################################################
def quantize_tensor(bit_budget, tensor):
    """
    Quantizes a 1D tensor (shape (1, L)) to fit within the specified bit budget, 
    including space for metadata.

    Args:
        bit_budget (int): Total number of bits available (metadata included).
        tensor (torch.Tensor): Input tensor of shape (1, L).

    Returns:
        dict: { 'min_val', 'max_val', 'quantized_values', 'bits_per_element', 'shape' }
    """
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError("Tensor must have shape (1, L).")

    L = tensor.shape[1]
    metadata_bits = 32 + 32 + 32  # float32 for min, max and int32 for shape
    remaining_bits = bit_budget - metadata_bits
    if remaining_bits <= 0:
        raise ValueError("Bit budget too small to store metadata.")

    bits_per_element = remaining_bits // L
    if bits_per_element < 1:
        raise ValueError("Bit budget too small to allocate at least 1 bit per element.")

    # Limit bits_per_element to a maximum (32 in this example)
    bits_per_element = min(bits_per_element, 32)
    levels = 2 ** bits_per_element

    min_val = float(tensor.min())
    max_val = float(tensor.max())

    if min_val == max_val:
        # All elements the same
        quantized = torch.zeros(L, dtype=torch.uint8)
        return {
            'min_val': min_val,
            'max_val': max_val,
            'quantized_values': quantized,
            'bits_per_element': bits_per_element,
            'shape': tuple(tensor.shape)
        }

    # Normalize to [0, 1] and quantize
    normalized = (tensor - min_val) / (max_val - min_val)
    quantized = torch.floor(normalized * (levels - 1)).to(torch.int32)

    if bits_per_element <= 8:
        quantized = quantized.to(torch.uint8)
    elif bits_per_element <= 16:
        quantized = quantized.to(torch.int16)
    elif bits_per_element <= 32:
        quantized = quantized.to(torch.int32)
    else:
        raise ValueError(f"Unsupported bits_per_element: {bits_per_element}")

    return {
        'min_val': min_val,
        'max_val': max_val,
        'quantized_values': quantized,
        'bits_per_element': bits_per_element,
        'shape': tuple(tensor.shape)
    }


def dequantize_tensor(quantized_data):
    """
    Dequantizes the tensor from its quantized dictionary representation.

    Args:
        quantized_data (dict): { 'min_val', 'max_val', 'quantized_values', 
                                 'bits_per_element', 'shape' }

    Returns:
        torch.Tensor: Dequantized tensor with the original shape.
    """
    min_val = quantized_data['min_val']
    max_val = quantized_data['max_val']
    quantized = quantized_data['quantized_values']
    bits_per_element = quantized_data['bits_per_element']
    shape = quantized_data['shape']

    if min_val == max_val:
        # All elements the same
        return torch.full(shape, min_val, dtype=torch.float32)

    levels = 2 ** bits_per_element
    normalized = quantized.to(torch.float32) / (levels - 1)
    tensor = normalized * (max_val - min_val) + min_val
    return tensor.view(shape)


############################################################
# 5. Distributed Evaluation Pipeline
############################################################
def evaluate_distributed_model_pipeline(
    edge_model,
    server_model,
    dataloader,
    edge_device,
    server_device,
    num_classes,
    num_bits=None,
    use_bit_budget=False,
    total_bits=None,
    p_mask=0,
    mask_strategy='last',  # 'norm' or 'last'
    feature_importance=None
):
    """
    Evaluate a split (Edge/Server) model on the given dataloader.

    Args:
        edge_model (nn.Module): Model running on the edge.
        server_model (nn.Module): Model running on the server.
        dataloader (DataLoader): DataLoader for evaluation dataset.
        edge_device (torch.device or str): Device for edge model.
        server_device (torch.device or str): Device for server model.
        num_classes (int): Number of classes.
        num_bits (int, optional): Bits for quantization (if not using bit budget).
        use_bit_budget (bool): Whether to use a total bit budget approach.
        total_bits (int, optional): Total bits if use_bit_budget=True.
        p_mask (float): Percentage of features to mask out.
        mask_strategy (str): 'last' or 'norm'. 
            'last' -> Zero out the last p_mask% of features.
            'norm' -> Use feature_importance to identify important features.
        feature_importance (torch.Tensor, optional): Feature importance if using mask_strategy='norm'.

    Returns:
        accuracy (float): Accuracy score.
        cm (np.array): Row-normalized confusion matrix (in %).
    """
    # import numpy as np, matplotlib.pyplot as plt;  plt.plot(np.sort(feature_importance.detach().cpu()), np.arange(1, len(feature_importance)+1)/len(feature_importance), '.'); plt.show()

    edge_model.eval()
    server_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(edge_device), batch[1].to(edge_device)
            # Edge-side forward
            edge_output = edge_model(inputs)
            # print(torch.min(feature_importance), torch.max(feature_importance))

            # Masking logic
            if p_mask > 0:
                if mask_strategy == 'last':
                    # Zero out last p_mask% of features
                    cutoff = int(edge_output.shape[1] * (1 - p_mask / 100.0))
                    mask = torch.ones_like(edge_output)
                    mask[:, cutoff:] = 0
                    server_input = edge_output * mask
                elif mask_strategy == 'norm':
                    # Zero out features by importance
                    num_features = edge_output.shape[1]
                    k = int(num_features * (p_mask / 100.0))
                    
                    # Handle edge cases where k=0 or k=num_features
                    k = max(1, min(k, num_features))
                    
                    # Get indices of the k least important features
                    _, indices = torch.topk(feature_importance.abs(), k, largest=False)
                    # print(feature_importance[indices])
                    # print(torch.min(feature_importance), torch.max(feature_importance))


                    # Create mask: set least important features to zero
                    mask_indices = torch.ones_like(feature_importance)
                    mask_indices[indices] = 0
                    mask_indices = mask_indices.unsqueeze(0)
                    
                    server_input = edge_output * mask_indices

                else:
                    server_input = edge_output
            else:
                server_input = edge_output

            # (Optional) Quantization with total bit budget
            if use_bit_budget and total_bits is not None:
                quantized_data = quantize_tensor(total_bits, server_input)
                server_input = dequantize_tensor(quantized_data)

            # Move server_input to server_device
            server_input = server_input.to(server_device)

            # Server-side forward
            server_output = server_model(server_input)
            if server_output.ndim == 1:
                server_output = server_output.unsqueeze(0)

            predictions = torch.argmax(server_output, dim=-1)
            if predictions.ndim == 0:
                predictions = predictions.unsqueeze(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy, cm = compute_metrics(all_preds, all_labels, num_classes)
    return accuracy, cm


############################################################
# 6. Main Entry Point for Evaluation
############################################################
def evaluate_timestamped_model(
    model_timestamp: str,
    config_path: str,
    distributed: bool = False,
    # plotting args
    do_plot: bool = False,
    plot_filename: str = None,
    # distributed-inference args
    num_bits=None,
    use_bit_budget=False,
    total_bits=None,
    p_mask=0,
    mask_strategy='last',
    feature_importance=None,
    feature_importance_strategy = None

):
    """
    Orchestrate loading a timestamped model, retrieving data loaders,
    and performing either centralized or distributed inference.

    Args:
        model_timestamp (str): The timestamp (directory name) containing the model weights.
        config_path (str): Path to your config.yaml.
        distributed (bool): If True, perform edge/server distributed inference. 
                            Otherwise, do a standard (centralized) inference.
        do_plot (bool): Whether to plot and show the confusion matrix.
        plot_filename (str): If provided, saves the confusion matrix to this file.
        num_bits (int, optional): Bits for quantization (if not using bit budget).
        use_bit_budget (bool): Whether to use a total bit budget approach.
        total_bits (int, optional): Total bits if use_bit_budget=True.
        p_mask (float): Percentage of features to mask out.
        mask_strategy (str): 'last' or 'norm'. 
        feature_importance (torch.Tensor, optional): Feature importance if using mask_strategy='norm'.

    Returns:
        dict: Dictionary containing evaluation metrics and confusion matrix
              for both validation and test sets, e.g.:
              {
                  "val_accuracy": ...,
                  "val_confusion_matrix": ...,
                  "test_accuracy": ...,
                  "test_confusion_matrix": ...
              }
    """
    # 1. Load config and set seed
    config = load_config(config_path)
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)

    # 2. Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load model
    model = My_Model(config)
    model_weights_path = os.path.join(
        "/project_ghent/Mostafa/ActivityRecognition/DistInference/trained_model", 
        model_timestamp,
        "model.safetensors"
    )
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Get dataloaders
    (
        train_loader, 
        val_loader, 
        test_loader, 
        _,
        val_dataset, 
        test_dataset,
        _
    ) = get_dataloaders_from_config(config_path)

    num_classes = config['model']['num_classes']

    # 5. Evaluate model
    if not distributed:
        # Centralized
        test_accuracy, test_cm = evaluate_model(model, test_loader, device, num_classes)
    else:
        # Distributed
        edge_model = EdgeModel(model)
        server_model = ServerModel(model)
        if feature_importance_strategy == "norm2":
            feature_importance = torch.norm(server_model.classifier.weight.data, p=2, dim=0)
        elif feature_importance_strategy == "max_abs":
            feature_importance = torch.max(torch.abs(server_model.classifier.weight.data), dim=0).values
            # print(server_model.classifier.weight.data.shape, feature_importance.shape)
            # print(feature_importance)
        edge_device = device
        server_device = device
        edge_model.to(edge_device)
        server_model.to(server_device)


        # Evaluate on test set
        test_accuracy, test_cm = evaluate_distributed_model_pipeline(
            edge_model=edge_model, 
            server_model=server_model, 
            dataloader=test_loader, 
            edge_device=edge_device, 
            server_device=server_device,
            num_classes=num_classes,
            num_bits=num_bits,
            use_bit_budget=use_bit_budget,
            total_bits=total_bits,
            p_mask=p_mask,
            mask_strategy=mask_strategy,
            feature_importance=feature_importance
        )

    # 6. Optionally plot the test confusion matrix
    if do_plot:
        # Example label map; adjust as needed
        label_map = {
            0: 'walking',
            1: 'sitting down',
            2: 'stand up',
            3: 'pick up an object',
            4: 'drink water',
            5: 'fall'
        }
        class_names = [label_map[i] for i in range(num_classes)]

        fig = plot_and_save_normalized_confusion_matrix(
            test_cm,
            class_names,
            title="Confusion Matrix (Test Set)",
            filename=plot_filename
        );

    # 7. Return results
    return {
        "test_accuracy": test_accuracy,
        "test_confusion_matrix": test_cm,
        "fig": fig
};
