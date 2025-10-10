"""Clinical Composite ROI Loss Functions (Equation 2 from Proposal)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class ClinicalROILoss(nn.Module):
    """
    Implements Equation 2 from CVPR 2026 proposal:

    L_ROI = α·(1 - SSIM(F, IM | M_brain)) +
            β·(1 - SSIM(F, IC | M_bone)) +
            γ·[λ1·Dice + λ2·NSD_τ + λ3·HD95]

    where:
    - F: Fused image
    - IM: MRI reference
    - IC: CT reference
    - M_brain, M_bone, M_les: ROI masks
    - S*: Ground truth lesion segmentation
    - Ŝ(F): Predicted lesion from fused image
    """

    def __init__(
        self,
        alpha=1.0,      # Brain region weight
        beta=1.0,       # Bone region weight
        gamma=2.0,      # Lesion region weight
        lambda_dice=1.0,
        lambda_nsd=1.0,
        lambda_hd95=0.5,
        tolerance_mm=2.0  # τ for NSD metric
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_dice = lambda_dice
        self.lambda_nsd = lambda_nsd
        self.lambda_hd95 = lambda_hd95
        self.tolerance_mm = tolerance_mm

    def forward(
        self,
        fused,
        mri,
        ct,
        brain_mask,
        bone_mask,
        lesion_pred=None,
        lesion_gt=None,
        voxel_spacing=(1.0, 1.0, 1.0)
    ):
        """
        Compute total ROI loss

        Args:
            fused: Fused image (B, 1, D, H, W)
            mri: MRI reference (B, 1, D, H, W)
            ct: CT reference (B, 1, D, H, W)
            brain_mask: Brain ROI (B, 1, D, H, W)
            bone_mask: Bone ROI (B, 1, D, H, W)
            lesion_pred: Predicted lesion segmentation (B, 1, D, H, W)
            lesion_gt: Ground truth lesion (B, 1, D, H, W)
            voxel_spacing: Voxel spacing for NSD/HD95 (tuple)

        Returns:
            Total loss and loss dict
        """
        losses = {}

        # L_brain: 1 - SSIM(F, IM | M_brain)
        if brain_mask is not None and mri is not None:
            l_brain = 1 - roi_ssim(fused, mri, brain_mask)
            losses['brain'] = l_brain.item()
            total_loss = self.alpha * l_brain
        else:
            total_loss = 0.0

        # L_bone: 1 - SSIM(F, IC | M_bone)
        if bone_mask is not None and ct is not None:
            l_bone = 1 - roi_ssim(fused, ct, bone_mask)
            losses['bone'] = l_bone.item()
            total_loss = total_loss + self.beta * l_bone

        # L_les: λ1·Dice + λ2·NSD + λ3·HD95
        if lesion_pred is not None and lesion_gt is not None:
            l_dice = dice_loss(lesion_pred, lesion_gt)
            l_nsd = nsd_loss(lesion_pred, lesion_gt, self.tolerance_mm, voxel_spacing)
            l_hd95 = hd95_loss(lesion_pred, lesion_gt, voxel_spacing)

            l_lesion = (
                self.lambda_dice * l_dice +
                self.lambda_nsd * l_nsd +
                self.lambda_hd95 * l_hd95
            )

            losses['lesion_dice'] = l_dice.item()
            losses['lesion_nsd'] = l_nsd.item()
            losses['lesion_hd95'] = l_hd95.item()
            losses['lesion_total'] = l_lesion.item()

            total_loss = total_loss + self.gamma * l_lesion

        losses['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, losses


def roi_ssim(pred, target, mask, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM within ROI mask

    Args:
        pred: Predicted image (B, C, D, H, W)
        target: Target image (B, C, D, H, W)
        mask: ROI mask (B, 1, D, H, W)

    Returns:
        SSIM value (scalar)
    """
    # Apply mask
    pred_masked = pred * mask
    target_masked = target * mask

    # Count valid pixels
    n_pixels = mask.sum() + 1e-8

    # Compute means
    mu_pred = pred_masked.sum() / n_pixels
    mu_target = target_masked.sum() / n_pixels

    # Compute variances
    pred_centered = (pred_masked - mu_pred * mask)
    target_centered = (target_masked - mu_target * mask)

    sigma_pred = (pred_centered ** 2).sum() / n_pixels
    sigma_target = (target_centered ** 2).sum() / n_pixels
    sigma_pred_target = (pred_centered * target_centered).sum() / n_pixels

    # SSIM formula
    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2)

    ssim = numerator / (denominator + 1e-8)

    return ssim.clamp(0, 1)


def roi_fsim(pred, target, mask):
    """
    Feature Similarity Index (FSIM) within ROI

    Simplified version using gradient magnitude
    """
    # Sobel filters for 3D gradient
    def compute_gradient_magnitude(x):
        # Simple gradient approximation
        gx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        gy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        gz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

        # Pad to match original size
        gx = F.pad(gx, (0, 1, 0, 0, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1, 0, 0))
        gz = F.pad(gz, (0, 0, 0, 0, 0, 1))

        grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        return grad_mag

    # Compute gradients
    grad_pred = compute_gradient_magnitude(pred)
    grad_target = compute_gradient_magnitude(target)

    # Apply mask
    grad_pred_masked = grad_pred * mask
    grad_target_masked = grad_target * mask

    # FSIM similarity
    T = 0.85  # Threshold
    similarity = (2 * grad_pred_masked * grad_target_masked + T) / \
                 (grad_pred_masked**2 + grad_target_masked**2 + T)

    fsim = (similarity * mask).sum() / (mask.sum() + 1e-8)

    return fsim.clamp(0, 1)


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for segmentation

    Args:
        pred: Predicted segmentation (B, 1, D, H, W) - probabilities [0, 1]
        target: Ground truth (B, 1, D, H, W) - binary {0, 1}

    Returns:
        Dice loss (scalar)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice


def normalized_surface_dice(pred, target, tolerance_mm=2.0, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Normalized Surface Dice (NSD)

    Measures boundary agreement within tolerance

    Args:
        pred: Predicted segmentation (B, 1, D, H, W)
        target: Ground truth (B, 1, D, H, W)
        tolerance_mm: Tolerance in mm
        voxel_spacing: Voxel spacing (mm)

    Returns:
        NSD score [0, 1]
    """
    # Binarize predictions
    pred_binary = (pred > 0.5).float().cpu().numpy()
    target_binary = (target > 0.5).float().cpu().numpy()

    # Extract surfaces (boundaries)
    pred_surface = extract_surface(pred_binary[0, 0])
    target_surface = extract_surface(target_binary[0, 0])

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_surface, sampling=voxel_spacing)
    target_dist = distance_transform_edt(1 - target_surface, sampling=voxel_spacing)

    # Points within tolerance
    pred_within_tolerance = (pred_dist[target_surface > 0] <= tolerance_mm).mean()
    target_within_tolerance = (target_dist[pred_surface > 0] <= tolerance_mm).mean()

    # NSD = average of both directions
    nsd = (pred_within_tolerance + target_within_tolerance) / 2.0

    return torch.tensor(nsd, device=pred.device)


def nsd_loss(pred, target, tolerance_mm=2.0, voxel_spacing=(1.0, 1.0, 1.0)):
    """NSD loss: 1 - NSD"""
    nsd = normalized_surface_dice(pred, target, tolerance_mm, voxel_spacing)
    return 1 - nsd


def hausdorff_95(pred, target, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    95th percentile Hausdorff Distance

    Args:
        pred: Predicted segmentation (B, 1, D, H, W)
        target: Ground truth (B, 1, D, H, W)
        voxel_spacing: Voxel spacing (mm)

    Returns:
        HD95 distance (mm)
    """
    # Binarize
    pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]
    target_binary = (target > 0.5).float().cpu().numpy()[0, 0]

    # Extract surfaces
    pred_surface = extract_surface(pred_binary)
    target_surface = extract_surface(target_binary)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return torch.tensor(100.0, device=pred.device)  # Large penalty

    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_surface, sampling=voxel_spacing)
    target_dist = distance_transform_edt(1 - target_surface, sampling=voxel_spacing)

    # Distances from pred surface to target
    distances_pred_to_target = pred_dist[target_surface > 0]
    # Distances from target surface to pred
    distances_target_to_pred = target_dist[pred_surface > 0]

    # 95th percentile
    hd95 = max(
        np.percentile(distances_pred_to_target, 95),
        np.percentile(distances_target_to_pred, 95)
    )

    return torch.tensor(hd95, device=pred.device)


def hd95_loss(pred, target, voxel_spacing=(1.0, 1.0, 1.0), max_hd=100.0):
    """
    HD95 loss: normalized by max_hd

    Returns value in [0, 1]
    """
    hd95 = hausdorff_95(pred, target, voxel_spacing)
    return torch.clamp(hd95 / max_hd, 0, 1)


def extract_surface(binary_mask):
    """
    Extract surface (boundary) from binary mask

    Args:
        binary_mask: Binary mask (D, H, W)

    Returns:
        Surface mask (D, H, W)
    """
    from scipy.ndimage import binary_erosion

    # Erode by 1 voxel
    eroded = binary_erosion(binary_mask)

    # Surface = original - eroded
    surface = binary_mask.astype(np.float32) - eroded.astype(np.float32)

    return surface


class MultiPresetROILoss(nn.Module):
    """
    ROI Loss with disease-specific presets

    Presets:
    - brain_tumor: α=1.5, β=0.5, γ=2.0 (emphasize brain tissue)
    - bone_tumor: α=0.5, β=2.0, γ=2.0 (emphasize bone)
    - metastasis: α=1.0, β=1.0, γ=3.0 (strong lesion preservation)
    """

    PRESETS = {
        'default': {'alpha': 1.0, 'beta': 1.0, 'gamma': 2.0},
        'brain_tumor': {'alpha': 1.5, 'beta': 0.5, 'gamma': 2.0},
        'bone_tumor': {'alpha': 0.5, 'beta': 2.0, 'gamma': 2.0},
        'metastasis': {'alpha': 1.0, 'beta': 1.0, 'gamma': 3.0},
    }

    def __init__(self, preset='default', **kwargs):
        super().__init__()

        # Load preset
        preset_params = self.PRESETS.get(preset, self.PRESETS['default'])

        # Override with kwargs
        params = {**preset_params, **kwargs}

        self.loss_fn = ClinicalROILoss(**params)

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

    def set_preset(self, preset):
        """Change preset dynamically"""
        preset_params = self.PRESETS.get(preset, self.PRESETS['default'])
        self.loss_fn.alpha = preset_params['alpha']
        self.loss_fn.beta = preset_params['beta']
        self.loss_fn.gamma = preset_params['gamma']
