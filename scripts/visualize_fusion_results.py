#!/usr/bin/env python3
"""
Visualize fusion results from validation during training.

This script saves fused images, MRI, CT, and comparison views
for each validation batch at specified epochs.

Usage:
    # Add to config: save_vis_every: 5  # Save every 5 epochs
    # Or call manually during validation
"""

import torch
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict

def normalize_for_display(img: np.ndarray, percentile_clip: tuple = (1, 99)) -> np.ndarray:
    """
    Normalize image for display

    Args:
        img: Image array
        percentile_clip: Percentile range for clipping

    Returns:
        Normalized image in [0, 1]
    """
    if percentile_clip:
        vmin, vmax = np.percentile(img, percentile_clip)
        img = np.clip(img, vmin, vmax)
    else:
        vmin, vmax = img.min(), img.max()

    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    return img


def save_fusion_visualization(
    fused: torch.Tensor,
    mri: torch.Tensor,
    ct: torch.Tensor,
    case_id: str,
    epoch: int,
    output_dir: Path,
    brain_mask: Optional[torch.Tensor] = None,
    bone_mask: Optional[torch.Tensor] = None,
    lesion_mask: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Save visualization of fusion results

    Args:
        fused: Fused image (B, 1, D, H, W)
        mri: MRI image (B, 1, D, H, W)
        ct: CT image (B, 1, D, H, W)
        case_id: Case identifier
        epoch: Current epoch number
        output_dir: Output directory
        brain_mask: Brain ROI mask (optional)
        bone_mask: Bone ROI mask (optional)
        lesion_mask: Lesion mask (optional)
        metrics: Metrics dictionary (optional)
    """
    # Convert to numpy and squeeze
    fused_np = fused[0, 0].cpu().numpy()  # (D, H, W)
    mri_np = mri[0, 0].cpu().numpy()
    ct_np = ct[0, 0].cpu().numpy()

    # Select middle slices for visualization
    d = fused_np.shape[0]
    slice_idxs = [d // 4, d // 2, 3 * d // 4]  # 25%, 50%, 75%

    # Create output directory
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # === Figure 1: Main comparison (Fused vs MRI vs CT) ===
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Epoch {epoch} - {case_id}', fontsize=16, fontweight='bold')

    for i, slice_idx in enumerate(slice_idxs):
        # MRI
        ax = axes[i, 0]
        mri_slice = normalize_for_display(mri_np[slice_idx])
        ax.imshow(mri_slice, cmap='gray')
        ax.set_title(f'MRI - Slice {slice_idx}/{d}')
        ax.axis('off')

        # CT
        ax = axes[i, 1]
        ct_slice = normalize_for_display(ct_np[slice_idx])
        ax.imshow(ct_slice, cmap='gray')
        ax.set_title(f'CT - Slice {slice_idx}/{d}')
        ax.axis('off')

        # Fused
        ax = axes[i, 2]
        fused_slice = normalize_for_display(fused_np[slice_idx])
        ax.imshow(fused_slice, cmap='gray')
        ax.set_title(f'Fused - Slice {slice_idx}/{d}')
        ax.axis('off')

    # Add metrics if provided
    if metrics:
        metrics_text = "Metrics:\n"
        for k, v in metrics.items():
            if not np.isnan(v):
                metrics_text += f"  {k}: {v:.4f}\n"
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
                verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(epoch_dir / f"{case_id}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # === Figure 2: ROI Masks Overlay ===
    if brain_mask is not None or bone_mask is not None or lesion_mask is not None:
        fig, axes = plt.subplots(len(slice_idxs), 4, figsize=(20, 5 * len(slice_idxs)))
        fig.suptitle(f'Epoch {epoch} - {case_id} - ROI Masks', fontsize=16, fontweight='bold')

        for i, slice_idx in enumerate(slice_idxs):
            fused_slice = normalize_for_display(fused_np[slice_idx])

            # Fused image
            axes[i, 0].imshow(fused_slice, cmap='gray')
            axes[i, 0].set_title(f'Fused - Slice {slice_idx}')
            axes[i, 0].axis('off')

            # Brain mask overlay
            axes[i, 1].imshow(fused_slice, cmap='gray')
            if brain_mask is not None:
                brain_np = brain_mask[0, 0, slice_idx].cpu().numpy()
                axes[i, 1].imshow(brain_np, cmap='Reds', alpha=0.3 * (brain_np > 0))
            axes[i, 1].set_title('Brain ROI')
            axes[i, 1].axis('off')

            # Bone mask overlay
            axes[i, 2].imshow(fused_slice, cmap='gray')
            if bone_mask is not None:
                bone_np = bone_mask[0, 0, slice_idx].cpu().numpy()
                axes[i, 2].imshow(bone_np, cmap='Blues', alpha=0.3 * (bone_np > 0))
            axes[i, 2].set_title('Bone ROI')
            axes[i, 2].axis('off')

            # Lesion mask overlay
            axes[i, 3].imshow(fused_slice, cmap='gray')
            if lesion_mask is not None:
                lesion_np = lesion_mask[0, 0, slice_idx].cpu().numpy()
                axes[i, 3].imshow(lesion_np, cmap='Greens', alpha=0.5 * (lesion_np > 0))
            axes[i, 3].set_title('Lesion Mask')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(epoch_dir / f"{case_id}_masks.png", dpi=150, bbox_inches='tight')
        plt.close()

    # === Save NIfTI files for detailed inspection ===
    nifti_dir = epoch_dir / "nifti"
    nifti_dir.mkdir(exist_ok=True)

    # Save fused image as NIfTI
    fused_nii = nib.Nifti1Image(fused_np, affine=np.eye(4))
    nib.save(fused_nii, nifti_dir / f"{case_id}_fused.nii.gz")

    print(f"✓ Saved visualization: {epoch_dir / case_id}")


def save_validation_samples(
    trainer,
    val_loader,
    epoch: int,
    num_samples: int = 3
):
    """
    Save visualization samples during validation

    Args:
        trainer: FusionTrainer instance
        val_loader: Validation dataloader
        epoch: Current epoch
        num_samples: Number of validation samples to visualize
    """
    output_dir = trainer.experiment_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving {num_samples} validation visualizations...")

    trainer.model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break

            case_id = batch['case_id'][0] if 'case_id' in batch else f"case_{batch_idx}"

            mri = batch['mri'].to(trainer.device)
            ct = batch['ct'].to(trainer.device)
            brain_mask = batch.get('brain_mask', None)
            bone_mask = batch.get('bone_mask', None)
            lesion_mask = batch.get('lesion_mask', None)

            if brain_mask is not None:
                brain_mask = brain_mask.to(trainer.device)
            if bone_mask is not None:
                bone_mask = bone_mask.to(trainer.device)
            if lesion_mask is not None:
                lesion_mask = lesion_mask.to(trainer.device)

            # Generate fused image
            fused = trainer.model.sample(
                mri=mri,
                ct=ct,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_mask=lesion_mask,
                lesion_head=trainer.lesion_head,
                sampling_timesteps=20,
                verbose=False
            )

            # Compute metrics
            metrics = trainer.roi_metrics.compute_all_metrics(
                fused=fused,
                mri=mri,
                ct=ct,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_pred=lesion_mask,
                lesion_gt=lesion_mask
            )

            # Save visualization
            save_fusion_visualization(
                fused=fused,
                mri=mri,
                ct=ct,
                case_id=case_id,
                epoch=epoch,
                output_dir=output_dir,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_mask=lesion_mask,
                metrics=metrics
            )

    print(f"✓ Visualizations saved to: {output_dir / f'epoch_{epoch:03d}'}")


if __name__ == '__main__':
    print("This script is meant to be imported and used during training.")
    print("See fusion_trainer.py for integration example.")