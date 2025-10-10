"""Modality-specific encoders for clinical data"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    """Encoder for medical imaging modalities (CT, PET, MRI)"""

    def __init__(self, in_channels=1, feat_dim=2048, pretrained=True):
        super().__init__()

        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)

        # Modify first conv layer for single-channel medical images
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove final FC layer
        self.backbone.fc = nn.Identity()

        self.feat_dim = feat_dim

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            features: (batch_size, feat_dim)
        """
        return self.backbone(x)


class ClinicalEncoder(nn.Module):
    """Encoder for tabular clinical data"""

    def __init__(self, input_dim, hidden_dims=[256, 512, 1024], feat_dim=1024, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final projection to feature dimension
        layers.append(nn.Linear(prev_dim, feat_dim))

        self.encoder = nn.Sequential(*layers)
        self.feat_dim = feat_dim

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            features: (batch_size, feat_dim)
        """
        return self.encoder(x)


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder combining multiple modality encoders"""

    def __init__(self, modality_configs):
        """
        Args:
            modality_configs: dict mapping modality names to encoder configs
                e.g., {'ct': {'type': 'image', ...}, 'clinical': {'type': 'clinical', ...}}
        """
        super().__init__()

        self.modality_names = list(modality_configs.keys())
        self.encoders = nn.ModuleDict()

        for modality, config in modality_configs.items():
            if config['type'] == 'image':
                self.encoders[modality] = ImageEncoder(
                    in_channels=config.get('in_channels', 1),
                    feat_dim=config.get('feat_dim', 2048),
                    pretrained=config.get('pretrained', True)
                )
            elif config['type'] == 'clinical':
                self.encoders[modality] = ClinicalEncoder(
                    input_dim=config['input_dim'],
                    hidden_dims=config.get('hidden_dims', [256, 512, 1024]),
                    feat_dim=config.get('feat_dim', 1024),
                    dropout=config.get('dropout', 0.3)
                )
            else:
                raise ValueError(f"Unknown encoder type: {config['type']}")

    def forward(self, modality_inputs):
        """
        Args:
            modality_inputs: dict mapping modality names to input tensors
        Returns:
            features: dict mapping modality names to feature tensors
        """
        features = {}
        for modality, data in modality_inputs.items():
            if modality in self.encoders:
                features[modality] = self.encoders[modality](data)
        return features

    def get_feat_dims(self):
        """Returns dictionary of feature dimensions for each modality"""
        return {name: encoder.feat_dim for name, encoder in self.encoders.items()}
