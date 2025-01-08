# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EdgeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', input_channels=1, pretrained=False):
        super(EdgeModel, self).__init__()
        # Load EfficientNet without the classifier
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=input_channels,
            num_classes=0,          # Removes the classification head
            global_pool='avg'       # Ensures global average pooling is applied
        )
        
    def forward(self, x):
        latent = self.model.forward_features(x)              # Shape: (batch, 1280, H, W)
        latent = F.adaptive_avg_pool2d(latent, (1, 1))        # Shape: (batch, 1280, 1, 1)
        latent = torch.flatten(latent, 1)                    # Shape: (batch, 1280)
        return latent

class ServerModel(nn.Module):
    def __init__(self, feature_dim, num_classes=6):
        super(ServerModel, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, latent):
        logits = self.classifier(latent)
        return logits

import torch
import torch.nn as nn
import timm
import random

class My_Model(nn.Module):
    def __init__(self, config):
        super(My_Model, self).__init__()
        """
        Initializes the My_Model using a ResNet backbone from the timm library.

        Args:
            config (dict): Configuration dictionary containing model parameters.
                - config['model']['name'] (str): Name of the ResNet model (e.g., 'resnet50').
                - config['model']['input_channels'] (int): Number of input channels (e.g., 1 for grayscale).
                - config['model']['num_classes'] (int): Number of output classes (e.g., 6).
                - config['model']['pretrained'] (bool): Whether to use pretrained weights.
                - config['model']['learn_order_of_importance'] (bool): Whether to enable importance-based masking.
                - config['model']['max_p'] (float): Maximum percentage of features to randomly mask (0 <= max_p <= 1).
        """
        # Load a pre-trained ResNet model without the classification head
        self.backbone = timm.create_model(
            model_name=config['model']['name'],                   # e.g., 'resnet50'
            pretrained=config['model'].get('pretrained', False),  # Use pretrained weights if specified
            in_chans=config['model']['input_channels'],           # 1 for single-channel input
            num_classes=0,                                        # Removes the classification head
            global_pool='avg'                                     # Ensures global average pooling is applied
        )

        # Verify the number of features from the backbone
        self.expected_num_features = self.backbone.num_features
        print(f"Backbone '{config['model']['name']}' outputs {self.expected_num_features} features.")

        # Define the classification head
        self.classifier = nn.Linear(self.expected_num_features, config['model']['num_classes'])

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Optional parameters for importance-based masking
        self.learn_order_of_importance = config['model'].get('learn_order_of_importance', False)
        self.max_p = config['model'].get('max_p', 0.0)  # 0.0 means no masking by default
        self.mask_distribution=config['model'].get('mask_distribution', 'uniform')
        self.exponential_distribution_alpha=config['model'].get('exponential_distribution_alpha', 1)
        

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        """
        Forward pass of the My_Model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            labels (torch.Tensor, optional): Target labels of shape (batch_size,). Defaults to None.

        Returns:
            dict: A dictionary containing:
                - 'logits' (torch.Tensor): Raw output scores from the classifier.
                - 'loss' (torch.Tensor, optional): Computed loss if labels are provided.
        """
        x = input_ids  # Map 'input_ids' to 'x'
        features = self.backbone(x)   # Shape: (batch_size, num_features)

        # --- Importance-based masking (training only) ---
        if self.learn_order_of_importance and self.training and self.max_p > 0:
            num_features = features.size(1)
            max_k = int(self.max_p * num_features)

            if max_k > 0:
                if self.mask_distribution == 'uniform':
                    # Uniform distribution: sample k uniformly from [0, max_k]
                    sampled_k = random.randint(0, max_k)
                elif self.mask_distribution == 'exponential':
                    # Exponential decay distribution: P(k) ~ exp(-alpha * k)
                    k_values = torch.arange(0, max_k + 1, device=features.device, dtype=torch.float32)
                    # Compute unnormalized probabilities
                    probs = torch.exp(-self.exponential_distribution_alpha * k_values)
                    # Normalize probabilities
                    probs /= probs.sum()

                    # Sample k using torch.multinomial
                    sampled_k_tensor = torch.multinomial(probs, num_samples=1)
                    sampled_k = sampled_k_tensor.item()
                else:
                    raise ValueError(f"Unsupported mask_distribution: {self.mask_distribution}")

                if sampled_k > 0:
                    # Zero out the last k features along dimension=1
                    features[..., -sampled_k:] = 0
        logits = self.classifier(features)  # Shape: (batch_size, num_classes)
        output = {'logits': logits}

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output['loss'] = loss

        return output
