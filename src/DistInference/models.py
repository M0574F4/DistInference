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
        """
        # Load a pre-trained ResNet model without the classification head
        self.backbone = timm.create_model(
            model_name=config['model']['name'],                  # e.g., 'resnet50'
            pretrained=config['model'].get('pretrained', False),# Use pretrained weights if specified
            in_chans=config['model']['input_channels'],          # 1 for single-channel input
            num_classes=0,                                       # Removes the classification head
            global_pool='avg'                                    # Ensures global average pooling is applied
        )

        # Verify the number of features from the backbone
        expected_num_features = self.backbone.num_features
        print(f"Backbone '{config['model']['name']}' outputs {expected_num_features} features.")

        # Define the classification head
        self.classifier = nn.Linear(expected_num_features, config['model']['num_classes'])

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
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
        logits = self.classifier(features)  # Shape: (batch_size, num_classes)

        output = {'logits': logits}

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output['loss'] = loss

        return output