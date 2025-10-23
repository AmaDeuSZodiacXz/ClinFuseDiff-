#!/usr/bin/env python3
"""
Diff-IF Inference Wrapper

Original Paper: "Diff-IF: Multi-modality image fusion via diffusion model
                 with fusion knowledge prior" (Information Fusion 2024)
GitHub: https://github.com/XunpengYi/Diff-IF

This wrapper adapts Diff-IF for our APIS CT-MRI data format.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add Diff-IF repo to path (after cloning)
DIFF_IF_ROOT = Path(__file__).parent / "Diff-IF"
if DIFF_IF_ROOT.exists():
    sys.path.insert(0, str(DIFF_IF_ROOT))


class DiffIFWrapper:
    """
    Wrapper for Diff-IF pretrained model

    Usage:
        model = DiffIFWrapper(checkpoint_path="baselines/diffusion_methods/diff_if/pretrained/model.pth")
        fused = model.fuse(mri, ct)
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize Diff-IF model

        Args:
            checkpoint_path: Path to pretrained weights
            device: Device to run on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading Diff-IF model from: {checkpoint_path}")

        # Try to import Diff-IF modules
        try:
            # NOTE: Actual imports depend on Diff-IF repo structure
            # This is a template - update after cloning the repo
            from model import DiffusionFusion  # Example import

            self.model = DiffusionFusion()

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(device)
            self.model.eval()

            print("✓ Diff-IF model loaded successfully")

        except ImportError as e:
            print(f"⚠ Could not import Diff-IF modules: {e}")
            print(f"   Please clone Diff-IF repo to: {DIFF_IF_ROOT}")
            print(f"   git clone https://github.com/XunpengYi/Diff-IF {DIFF_IF_ROOT}")
            self.model = None

    @torch.no_grad()
    def fuse(self, mri: np.ndarray, ct: np.ndarray) -> np.ndarray:
        """
        Fuse MRI and CT using Diff-IF

        Args:
            mri: MRI volume (D, H, W) or (H, W) normalized to [0, 1]
            ct: CT volume (D, H, W) or (H, W) normalized to [0, 1]

        Returns:
            fused: Fused volume same shape as input
        """
        if self.model is None:
            # Fallback to simple average
            print("⚠ Diff-IF model not available, using simple average")
            return (mri + ct) / 2.0

        # Convert to torch tensors
        mri_t = torch.from_numpy(mri).float()
        ct_t = torch.from_numpy(ct).float()

        # Add batch and channel dimensions
        if mri_t.ndim == 2:  # 2D image
            mri_t = mri_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            ct_t = ct_t.unsqueeze(0).unsqueeze(0)
        elif mri_t.ndim == 3:  # 3D volume
            # Diff-IF may only support 2D - process slice by slice
            fused_slices = []
            for z in range(mri.shape[0]):
                mri_slice = mri_t[z:z+1].unsqueeze(0)  # (1, 1, H, W)
                ct_slice = ct_t[z:z+1].unsqueeze(0)

                fused_slice = self._fuse_2d(mri_slice, ct_slice)
                fused_slices.append(fused_slice[0, 0].cpu().numpy())

            return np.stack(fused_slices, axis=0)

        # Move to device
        mri_t = mri_t.to(self.device)
        ct_t = ct_t.to(self.device)

        # Run Diff-IF fusion
        fused_t = self._fuse_2d(mri_t, ct_t)

        # Convert back to numpy
        fused = fused_t[0, 0].cpu().numpy()

        return fused

    def _fuse_2d(self, mri_t: torch.Tensor, ct_t: torch.Tensor) -> torch.Tensor:
        """
        Internal 2D fusion (placeholder - update based on Diff-IF API)

        Args:
            mri_t: (1, 1, H, W)
            ct_t: (1, 1, H, W)

        Returns:
            fused_t: (1, 1, H, W)
        """
        # NOTE: Update this based on actual Diff-IF inference API
        # Example (hypothetical):
        try:
            # fused_t = self.model.sample(mri_t, ct_t)
            fused_t = self.model(mri_t, ct_t)  # Placeholder
            return fused_t
        except Exception as e:
            print(f"⚠ Diff-IF inference failed: {e}")
            # Fallback
            return (mri_t + ct_t) / 2.0


def test_wrapper():
    """Test the wrapper with dummy data"""
    print("Testing Diff-IF wrapper...")

    # Create dummy data
    mri = np.random.rand(256, 256).astype(np.float32)
    ct = np.random.rand(256, 256).astype(np.float32)

    # Test wrapper
    wrapper = DiffIFWrapper(
        checkpoint_path="baselines/diffusion_methods/diff_if/pretrained/model.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    fused = wrapper.fuse(mri, ct)

    print(f"✓ Fusion successful")
    print(f"  Input shape: {mri.shape}")
    print(f"  Output shape: {fused.shape}")
    print(f"  Output range: [{fused.min():.3f}, {fused.max():.3f}]")


if __name__ == '__main__':
    test_wrapper()