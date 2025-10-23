"""
Lesion Segmentation Head for CVPR 2026 CLIN-FuseDiff++

Frozen pretrained segmentation network used to:
1. Extract lesion boundaries from fused images
2. Compute boundary-aware metrics during guided diffusion
3. Provide gradient guidance for lesion preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


class LesionSegmentationHead(nn.Module):
    """
    Frozen lesion segmentation network

    Used in Algorithm 1, Line 7: Ŝ ← LesionHead(F)

    Architecture options:
    1. 3D U-Net (train from scratch on BraTS)
    2. nnU-Net (use pretrained weights)
    3. Transfer from TotalSegmentator
    """

    def __init__(
        self,
        architecture: str = "unet3d",
        in_channels: int = 1,
        num_classes: int = 1,  # Binary lesion segmentation
        pretrained_path: Optional[str] = None,
        frozen: bool = True,
        use_sigmoid: bool = True
    ):
        """
        Args:
            architecture: 'unet3d', 'nnunet', or 'totalseg'
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of output classes (1 for binary)
            pretrained_path: Path to pretrained weights (.pth file)
            frozen: If True, freeze all parameters
            use_sigmoid: Apply sigmoid to output (for probability maps)
        """
        super().__init__()

        self.architecture = architecture
        self.frozen = frozen
        self.use_sigmoid = use_sigmoid

        # Build network based on architecture
        if architecture == "unet3d":
            from .unet3d import UNet3D
            self.model = UNet3D(
                in_channels=in_channels,
                out_channels=num_classes,
                base_channels=32,  # Smaller than fusion U-Net
                channel_mult=[1, 2, 4, 8],
                num_res_blocks=2,
                attention_resolutions=[],  # No attention for efficiency
                dropout=0.0
            )
        elif architecture == "nnunet":
            # TODO: Load nnU-Net architecture
            raise NotImplementedError("nnU-Net support coming soon")
        elif architecture == "totalseg":
            # TODO: Adapt TotalSegmentator for lesion segmentation
            raise NotImplementedError("TotalSegmentator adaptation coming soon")
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        # Freeze if requested
        if frozen:
            self.freeze()

    def load_pretrained(self, path: str):
        """Load pretrained weights"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {path}")

        print(f"Loading lesion head weights from: {path}")
        checkpoint = torch.load(path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded pretrained lesion segmentation head")

    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print("✓ Lesion head frozen (no gradient updates)")

    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        print("✓ Lesion head unfrozen (trainable)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image (B, 1, D, H, W)

        Returns:
            Lesion probability map (B, 1, D, H, W)
        """
        # Ensure in eval mode if frozen
        if self.frozen:
            self.eval()

        with torch.set_grad_enabled(not self.frozen):
            logits = self.model(x)

        # Apply sigmoid for probability
        if self.use_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        else:
            return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary lesion mask

        Args:
            x: Input image (B, 1, D, H, W)
            threshold: Probability threshold for binary mask

        Returns:
            Binary mask (B, 1, D, H, W)
        """
        probs = self.forward(x)
        mask = (probs > threshold).float()
        return mask


class SimpleLesionHead(nn.Module):
    """
    Simple lesion segmentation head (for prototyping/testing)

    Use this when pretrained weights are not available.
    Can be trained from scratch on BraTS or similar datasets.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 1
    ):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.dec3 = self._conv_block(base_channels * 12, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 6, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 3, base_channels)

        # Output
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

        # Pooling/upsampling
        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        # Output
        out = self.out_conv(d1)
        return torch.sigmoid(out)


def create_lesion_head(
    config: dict,
    pretrained_path: Optional[str] = None
) -> LesionSegmentationHead:
    """
    Factory function to create lesion segmentation head from config

    Args:
        config: Configuration dict with keys:
            - architecture: 'unet3d', 'nnunet', or 'simple'
            - frozen: Whether to freeze parameters
            - pretrained: Path to pretrained weights
        pretrained_path: Override config pretrained path

    Returns:
        LesionSegmentationHead instance
    """
    architecture = config.get('architecture', 'unet3d')
    frozen = config.get('frozen', True)

    if pretrained_path is None:
        pretrained_path = config.get('pretrained', None)

    if architecture == 'simple':
        # Use simple head (for prototyping)
        return SimpleLesionHead(
            in_channels=1,
            base_channels=32,
            num_classes=1
        )
    else:
        # Use full lesion head
        return LesionSegmentationHead(
            architecture=architecture,
            in_channels=1,
            num_classes=1,
            pretrained_path=pretrained_path,
            frozen=frozen
        )


# Helper function for training lesion head from scratch
def train_lesion_head_on_brats(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4
):
    """
    Train lesion segmentation head on BraTS dataset

    This is a standalone training script for creating pretrained weights.
    Run this separately before training the full fusion model.

    Args:
        data_dir: BraTS dataset directory
        output_dir: Where to save trained weights
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # TODO: Implement BraTS training script
    # This would be a separate training pipeline:
    # 1. Load BraTS dataset (T1, T1ce, T2, FLAIR → lesion mask)
    # 2. Create LesionSegmentationHead
    # 3. Train with Dice + BCE loss
    # 4. Save best checkpoint
    raise NotImplementedError(
        "Training script for lesion head not yet implemented. "
        "Please train separately on BraTS or use pretrained nnU-Net weights."
    )


if __name__ == "__main__":
    # Test lesion head
    print("Testing LesionSegmentationHead...")

    # Create simple head (no pretrained weights)
    head = SimpleLesionHead(in_channels=1, base_channels=16, num_classes=1)

    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)  # Smaller test size
    y = head(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print("✓ Lesion head test passed!")

    # Test with full head (will fail without pretrained weights)
    try:
        full_head = LesionSegmentationHead(
            architecture="unet3d",
            pretrained_path=None,
            frozen=False
        )
        y2 = full_head(x)
        print(f"Full head output shape: {y2.shape}")
    except Exception as e:
        print(f"Full head test skipped (expected): {e}")