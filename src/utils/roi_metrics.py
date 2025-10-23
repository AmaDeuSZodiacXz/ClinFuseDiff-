"""
ROI-Aware Metrics Suite for CLIN-FuseDiff++

Primary Metrics (Section 4 of Proposal):
- Lesion ROI: Dice, NSD@τmm, HD95
- Brain ROI: SSIM/FSIM (F vs. MRI)
- Bone ROI: PSNR/SSIM (F vs. CT)

Secondary Metrics:
- Global: PSNR/SSIM/FSIM/FMI

Calibration Metrics:
- ECE (Expected Calibration Error)
- Brier Score
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from typing import Dict, Tuple, Optional


class ROIMetrics:
    """
    Complete ROI-aware metrics suite for image fusion evaluation
    """

    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0), tolerance_mm=2.0):
        """
        Args:
            voxel_spacing: Voxel spacing in mm (D, H, W)
            tolerance_mm: Tolerance for NSD metric
        """
        self.voxel_spacing = voxel_spacing
        self.tolerance_mm = tolerance_mm

    def compute_all_metrics(
        self,
        fused,
        mri,
        ct,
        brain_mask=None,
        bone_mask=None,
        lesion_pred=None,
        lesion_gt=None
    ) -> Dict[str, float]:
        """
        Compute all ROI-aware metrics

        Args:
            fused: Fused image (B, 1, D, H, W) or (D, H, W)
            mri: MRI reference (same shape as fused)
            ct: CT reference (same shape as fused)
            brain_mask: Brain ROI mask (same shape)
            bone_mask: Bone ROI mask (same shape)
            lesion_pred: Predicted lesion segmentation (same shape)
            lesion_gt: Ground truth lesion (same shape)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Ensure tensors
        if not isinstance(fused, torch.Tensor):
            fused = torch.from_numpy(fused).float()
        if not isinstance(mri, torch.Tensor):
            mri = torch.from_numpy(mri).float()
        if not isinstance(ct, torch.Tensor):
            ct = torch.from_numpy(ct).float()

        # Add batch and channel dims if needed
        if fused.ndim == 3:
            fused = fused.unsqueeze(0).unsqueeze(0)
            mri = mri.unsqueeze(0).unsqueeze(0)
            ct = ct.unsqueeze(0).unsqueeze(0)

        # === PRIMARY ROI METRICS ===

        # Brain ROI: SSIM/FSIM with MRI
        if brain_mask is not None:
            if not isinstance(brain_mask, torch.Tensor):
                brain_mask = torch.from_numpy(brain_mask).float()
            if brain_mask.ndim == 3:
                brain_mask = brain_mask.unsqueeze(0).unsqueeze(0)

            metrics['brain_ssim'] = self.roi_ssim(fused, mri, brain_mask).item()
            metrics['brain_fsim'] = self.roi_fsim(fused, mri, brain_mask).item()

        # Bone ROI: PSNR/SSIM with CT
        if bone_mask is not None:
            if not isinstance(bone_mask, torch.Tensor):
                bone_mask = torch.from_numpy(bone_mask).float()
            if bone_mask.ndim == 3:
                bone_mask = bone_mask.unsqueeze(0).unsqueeze(0)

            metrics['bone_psnr'] = self.roi_psnr(fused, ct, bone_mask).item()
            metrics['bone_ssim'] = self.roi_ssim(fused, ct, bone_mask).item()

        # Lesion ROI: Dice, NSD, HD95
        if lesion_pred is not None and lesion_gt is not None:
            if not isinstance(lesion_pred, torch.Tensor):
                lesion_pred = torch.from_numpy(lesion_pred).float()
            if not isinstance(lesion_gt, torch.Tensor):
                lesion_gt = torch.from_numpy(lesion_gt).float()

            if lesion_pred.ndim == 3:
                lesion_pred = lesion_pred.unsqueeze(0).unsqueeze(0)
                lesion_gt = lesion_gt.unsqueeze(0).unsqueeze(0)

            metrics['lesion_dice'] = self.dice_score(lesion_pred, lesion_gt).item()
            metrics['lesion_nsd'] = self.normalized_surface_dice(
                lesion_pred, lesion_gt, self.tolerance_mm
            ).item()
            metrics['lesion_hd95'] = self.hausdorff_95(lesion_pred, lesion_gt).item()

        # === SECONDARY GLOBAL METRICS ===

        metrics['global_psnr'] = self.psnr(fused, mri).item()
        metrics['global_ssim'] = self.ssim(fused, mri).item()
        metrics['global_fsim'] = self.fsim(fused, mri).item()

        return metrics

    # ==================== ROI-Specific Metrics ====================

    def roi_ssim(self, pred, target, mask):
        """
        SSIM within ROI

        FIXED: Extract only ROI pixels for statistics computation
        Previous bug: Masked multiplication included zeros in mean/variance
        """
        # Flatten and extract only ROI pixels
        mask_flat = mask.view(-1) > 0.5

        if mask_flat.sum() == 0:
            return torch.tensor(0.0)

        pred_roi = pred.view(-1)[mask_flat]
        target_roi = target.view(-1)[mask_flat]

        # Compute statistics on ROI pixels only
        mu_pred = pred_roi.mean()
        mu_target = target_roi.mean()

        sigma_pred = pred_roi.var()
        sigma_target = target_roi.var()
        sigma_pred_target = ((pred_roi - mu_pred) * (target_roi - mu_target)).mean()

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2) + 1e-8)

        return ssim.clamp(0, 1)

    def roi_fsim(self, pred, target, mask):
        """Feature Similarity Index within ROI"""
        def gradient_magnitude(x):
            gx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
            gy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
            gz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

            gx = F.pad(gx, (0, 1, 0, 0, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1, 0, 0))
            gz = F.pad(gz, (0, 0, 0, 0, 0, 1))

            return torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)

        grad_pred = gradient_magnitude(pred) * mask
        grad_target = gradient_magnitude(target) * mask

        T = 0.85
        similarity = (2 * grad_pred * grad_target + T) / \
                     (grad_pred**2 + grad_target**2 + T)

        fsim = (similarity * mask).sum() / (mask.sum() + 1e-8)
        return fsim.clamp(0, 1)

    def roi_psnr(self, pred, target, mask):
        """PSNR within ROI"""
        pred_masked = pred * mask
        target_masked = target * mask

        mse = ((pred_masked - target_masked) ** 2).sum() / (mask.sum() + 1e-8)
        max_val = target.max()

        psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
        return psnr

    # ==================== Lesion Boundary Metrics ====================

    def dice_score(self, pred, target, smooth=1e-8):
        """
        Dice coefficient with proper handling of edge cases

        Cases:
        1. Both empty (no lesion): Return NaN (not 1.0!)
        2. One empty, one not: Return 0.0 (complete mismatch)
        3. Both non-empty: Return normal Dice
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Check if both are empty (no lesion case)
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()

        if pred_sum < 1e-6 and target_sum < 1e-6:
            # Both empty - no lesion to evaluate
            return torch.tensor(float('nan'))

        if pred_sum < 1e-6 or target_sum < 1e-6:
            # Only one is empty - complete mismatch
            return torch.tensor(0.0)

        # Normal Dice computation
        intersection = (pred_flat * target_flat).sum()
        union = pred_sum + target_sum

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    def normalized_surface_dice(self, pred, target, tolerance_mm):
        """
        Normalized Surface Dice with tolerance

        Edge cases:
        - Both empty: Return NaN (no surface to compare)
        - One empty: Return 0.0 (complete mismatch)
        """
        pred_np = (pred > 0.5).float().cpu().numpy()[0, 0]
        target_np = (target > 0.5).float().cpu().numpy()[0, 0]

        # Check if both empty
        if pred_np.sum() < 1e-6 and target_np.sum() < 1e-6:
            return torch.tensor(float('nan'))

        pred_surface = self._extract_surface(pred_np)
        target_surface = self._extract_surface(target_np)

        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return torch.tensor(0.0)

        pred_dist = distance_transform_edt(1 - pred_surface, sampling=self.voxel_spacing)
        target_dist = distance_transform_edt(1 - target_surface, sampling=self.voxel_spacing)

        pred_within = (pred_dist[target_surface > 0] <= tolerance_mm).mean()
        target_within = (target_dist[pred_surface > 0] <= tolerance_mm).mean()

        nsd = (pred_within + target_within) / 2.0
        return torch.tensor(nsd)

    def hausdorff_95(self, pred, target):
        """
        95th percentile Hausdorff Distance

        Edge cases:
        - Both empty: Return NaN (no surface to compare)
        - One empty: Return 100.0 mm (worst possible, complete mismatch)
        """
        pred_np = (pred > 0.5).float().cpu().numpy()[0, 0]
        target_np = (target > 0.5).float().cpu().numpy()[0, 0]

        # Check if both empty
        if pred_np.sum() < 1e-6 and target_np.sum() < 1e-6:
            return torch.tensor(float('nan'))

        pred_surface = self._extract_surface(pred_np)
        target_surface = self._extract_surface(target_np)

        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return torch.tensor(100.0)

        pred_dist = distance_transform_edt(1 - pred_surface, sampling=self.voxel_spacing)
        target_dist = distance_transform_edt(1 - target_surface, sampling=self.voxel_spacing)

        d_pred_to_target = pred_dist[target_surface > 0]
        d_target_to_pred = target_dist[pred_surface > 0]

        hd95 = max(np.percentile(d_pred_to_target, 95),
                   np.percentile(d_target_to_pred, 95))

        return torch.tensor(hd95)

    @staticmethod
    def _extract_surface(binary_mask):
        """Extract boundary from binary mask"""
        eroded = binary_erosion(binary_mask)
        return binary_mask.astype(np.float32) - eroded.astype(np.float32)

    # ==================== Global Metrics ====================

    def psnr(self, pred, target):
        """Global PSNR"""
        mse = ((pred - target) ** 2).mean()
        max_val = target.max()
        psnr = 20 * torch.log10(max_val / (torch.sqrt(mse) + 1e-8))
        return psnr

    def ssim(self, pred, target, window_size=11):
        """Global SSIM"""
        mu_pred = pred.mean()
        mu_target = target.mean()

        sigma_pred = ((pred - mu_pred) ** 2).mean()
        sigma_target = ((target - mu_target) ** 2).mean()
        sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2) + 1e-8)

        return ssim.clamp(0, 1)

    def fsim(self, pred, target):
        """Global FSIM"""
        # Use roi_fsim with full mask
        full_mask = torch.ones_like(pred)
        return self.roi_fsim(pred, target, full_mask)


class CalibrationMetrics:
    """Uncertainty calibration metrics"""

    @staticmethod
    def expected_calibration_error(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidences: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE)

        Args:
            predictions: Predicted classes (N,)
            targets: Ground truth classes (N,)
            confidences: Prediction confidences [0, 1] (N,)
            n_bins: Number of calibration bins

        Returns:
            ECE value
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Points in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    @staticmethod
    def brier_score(
        probabilities: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Brier Score for calibration

        Args:
            probabilities: Predicted probabilities (N, C)
            targets: Ground truth one-hot (N, C)

        Returns:
            Brier score (lower is better)
        """
        return ((probabilities - targets) ** 2).mean().item()


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as readable table"""
    lines = ["=" * 60]
    lines.append("ROI-Aware Metrics Report")
    lines.append("=" * 60)

    if any(k.startswith('brain_') for k in metrics):
        lines.append("\nBrain ROI (vs. MRI):")
        for k, v in metrics.items():
            if k.startswith('brain_'):
                lines.append(f"  {k:20s}: {v:.4f}")

    if any(k.startswith('bone_') for k in metrics):
        lines.append("\nBone ROI (vs. CT):")
        for k, v in metrics.items():
            if k.startswith('bone_'):
                lines.append(f"  {k:20s}: {v:.4f}")

    if any(k.startswith('lesion_') for k in metrics):
        lines.append("\nLesion Boundary Metrics:")
        for k, v in metrics.items():
            if k.startswith('lesion_'):
                lines.append(f"  {k:20s}: {v:.4f}")

    if any(k.startswith('global_') for k in metrics):
        lines.append("\nGlobal Metrics:")
        for k, v in metrics.items():
            if k.startswith('global_'):
                lines.append(f"  {k:20s}: {v:.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)
