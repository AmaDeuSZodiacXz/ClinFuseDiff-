"""Fusion modules for combining multimodal features"""

import torch
import torch.nn as nn
from .diffusion import GaussianDiffusion


class AttentionFusion(nn.Module):
    """Attention-based fusion for multimodal features"""

    def __init__(self, feat_dims, fusion_dim=512, num_heads=8, dropout=0.1):
        """
        Args:
            feat_dims: dict of {modality: feature_dim}
            fusion_dim: dimension of fused representation
        """
        super().__init__()

        self.modality_names = list(feat_dims.keys())
        self.fusion_dim = fusion_dim

        # Projection layers for each modality
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in feat_dims.items()
        })

        # Multi-head attention for cross-modal fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, features_dict, available_modalities=None):
        """
        Args:
            features_dict: dict of {modality: (batch_size, feat_dim)}
            available_modalities: list of available modality names (optional)
        Returns:
            fused: (batch_size, fusion_dim)
        """
        if available_modalities is None:
            available_modalities = list(features_dict.keys())

        # Project all features to fusion dimension
        projected = []
        for modality in available_modalities:
            if modality in features_dict:
                feat = self.projections[modality](features_dict[modality])
                projected.append(feat.unsqueeze(1))  # (B, 1, fusion_dim)

        if len(projected) == 0:
            raise ValueError("No modalities available for fusion")

        # Stack into sequence: (B, num_modalities, fusion_dim)
        modality_seq = torch.cat(projected, dim=1)

        # Self-attention across modalities
        attn_out, _ = self.attention(modality_seq, modality_seq, modality_seq)

        # Average pooling across modalities
        fused = attn_out.mean(dim=1)  # (B, fusion_dim)

        # Layer norm
        fused = self.norm(fused)

        return fused


class DiffusionFusion(nn.Module):
    """Diffusion-based fusion module"""

    def __init__(
        self,
        feat_dims,
        fusion_dim=512,
        num_timesteps=1000,
        beta_schedule='linear',
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        """
        Args:
            feat_dims: dict of {modality: feature_dim}
            fusion_dim: dimension of fused representation
            num_timesteps: number of diffusion timesteps
            beta_schedule: noise schedule type
        """
        super().__init__()

        self.modality_names = list(feat_dims.keys())
        self.fusion_dim = fusion_dim

        # Projection layers for each modality
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim)
            for name, dim in feat_dims.items()
        })

        # Conditioning network: combines available modalities
        self.condition_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Diffusion model
        self.diffusion = GaussianDiffusion(
            input_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule
        )

    def prepare_condition(self, features_dict, available_modalities=None):
        """Prepare conditioning from available modalities"""
        if available_modalities is None:
            available_modalities = list(features_dict.keys())

        # Project and average available modalities
        projected = []
        for modality in available_modalities:
            if modality in features_dict:
                feat = self.projections[modality](features_dict[modality])
                projected.append(feat)

        if len(projected) == 0:
            return None

        # Average pooling
        condition = torch.stack(projected, dim=0).mean(dim=0)  # (B, fusion_dim)

        # Process through conditioning network
        condition = self.condition_net(condition)  # (B, hidden_dim)

        return condition

    def forward(self, features_dict, available_modalities=None, mode='train'):
        """
        Args:
            features_dict: dict of {modality: (batch_size, feat_dim)}
            available_modalities: list of available modality names
            mode: 'train' or 'sample'
        Returns:
            fused: (batch_size, fusion_dim) or loss during training
        """
        # Prepare conditioning from available modalities
        condition = self.prepare_condition(features_dict, available_modalities)

        if mode == 'train':
            # During training, use ground truth fusion as target
            # Here we use average of all projected features as "ground truth"
            projected = []
            for modality in self.modality_names:
                if modality in features_dict:
                    feat = self.projections[modality](features_dict[modality])
                    projected.append(feat)

            if len(projected) == 0:
                raise ValueError("No modalities available")

            x_0 = torch.stack(projected, dim=0).mean(dim=0)  # (B, fusion_dim)

            # Compute diffusion loss
            loss = self.diffusion(x_0, condition)
            return loss

        else:  # mode == 'sample'
            # Sample fused representation through reverse diffusion
            batch_size = next(iter(features_dict.values())).shape[0]
            device = next(iter(features_dict.values())).device

            fused = self.diffusion.sample(
                shape=(batch_size, self.fusion_dim),
                condition=condition,
                device=device
            )
            return fused


class ClinFuseDiffModel(nn.Module):
    """Complete ClinFuseDiff model with encoders, fusion, and prediction head"""

    def __init__(
        self,
        modality_configs,
        fusion_config,
        predictor_config,
        use_diffusion=True
    ):
        super().__init__()

        from .encoders import MultiModalEncoder

        # Encoders
        self.encoders = MultiModalEncoder(modality_configs)
        feat_dims = self.encoders.get_feat_dims()

        # Fusion module
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.fusion = DiffusionFusion(feat_dims=feat_dims, **fusion_config)
        else:
            self.fusion = AttentionFusion(feat_dims=feat_dims, **fusion_config)

        # Prediction head
        fusion_dim = fusion_config.get('fusion_dim', 512)
        hidden_dims = predictor_config.get('hidden_dims', [256, 128])
        num_classes = predictor_config.get('num_classes', 2)
        dropout = predictor_config.get('dropout', 0.3)

        layers = []
        prev_dim = fusion_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.predictor = nn.Sequential(*layers)

    def forward(self, modality_inputs, available_modalities=None, mode='train'):
        """
        Args:
            modality_inputs: dict of {modality: input_tensor}
            available_modalities: list of available modality names
            mode: 'train' or 'inference'
        Returns:
            outputs: dict containing predictions and losses
        """
        # Encode modalities
        features = self.encoders(modality_inputs)

        outputs = {}

        # Fusion
        if self.use_diffusion:
            if mode == 'train':
                # During training, compute diffusion loss
                fusion_loss = self.fusion(features, available_modalities, mode='train')
                outputs['fusion_loss'] = fusion_loss

                # Also sample for prediction
                fused = self.fusion(features, available_modalities, mode='sample')
            else:
                # During inference, only sample
                fused = self.fusion(features, available_modalities, mode='sample')
        else:
            fused = self.fusion(features, available_modalities)

        # Prediction
        logits = self.predictor(fused)
        outputs['logits'] = logits
        outputs['fused_features'] = fused

        return outputs
