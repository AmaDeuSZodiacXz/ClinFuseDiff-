#!/usr/bin/env python3
"""
Registration-Aware Robustness Testing for ClinFuseDiff++
Implements synthetic warp stress testing (±1-3mm) per Proposal §3.3
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional, Dict


def generate_synthetic_warp(
    shape: Tuple[int, int, int],
    max_displacement_mm: float = 2.0,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smoothness_sigma: float = 5.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate smooth random displacement field for stress testing

    Args:
        shape: (D, H, W) volume shape
        max_displacement_mm: Maximum displacement in millimeters
        voxel_spacing: Voxel spacing (D, H, W) in mm
        smoothness_sigma: Gaussian smoothing sigma for displacement field
        random_seed: Random seed for reproducibility

    Returns:
        displacement_field: (3, D, H, W) displacement in voxels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    D, H, W = shape

    # Generate random displacement field
    displacement = np.random.randn(3, D, H, W)

    # Convert max displacement from mm to voxels
    max_disp_voxels = np.array([
        max_displacement_mm / voxel_spacing[0],
        max_displacement_mm / voxel_spacing[1],
        max_displacement_mm / voxel_spacing[2]
    ])

    # Apply gaussian smoothing to each dimension
    for i in range(3):
        displacement[i] = gaussian_filter(displacement[i], sigma=smoothness_sigma)

        # Normalize to [-1, 1] then scale to max displacement
        displacement[i] = displacement[i] / (np.abs(displacement[i]).max() + 1e-8)
        displacement[i] = displacement[i] * max_disp_voxels[i]

    return displacement


def apply_displacement_field(
    image: np.ndarray,
    displacement: np.ndarray,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0
) -> np.ndarray:
    """
    Apply displacement field to image

    Args:
        image: (D, H, W) image volume
        displacement: (3, D, H, W) displacement field in voxels
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        mode: How to handle out-of-bounds ('constant', 'nearest', 'reflect')
        cval: Constant value for out-of-bounds if mode='constant'

    Returns:
        warped_image: (D, H, W) warped image
    """
    D, H, W = image.shape

    # Create coordinate grids
    d_coords, h_coords, w_coords = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )

    # Apply displacement
    d_warped = d_coords + displacement[0]
    h_warped = h_coords + displacement[1]
    w_warped = w_coords + displacement[2]

    # Stack coordinates
    coords = np.array([d_warped, h_warped, w_warped])

    # Apply warp using scipy map_coordinates
    warped = map_coordinates(image, coords, order=order, mode=mode, cval=cval)

    return warped


def stress_test_registration(
    ct: np.ndarray,
    mri: np.ndarray,
    displacement_range_mm: Tuple[float, float] = (1.0, 3.0),
    num_samples: int = 5,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate multiple mis-registered versions for robustness testing

    Args:
        ct: (D, H, W) CT volume
        mri: (D, H, W) MRI volume (reference)
        displacement_range_mm: (min, max) displacement range
        num_samples: Number of stress test samples to generate
        voxel_spacing: Voxel spacing in mm
        random_seed: Random seed

    Returns:
        dict with:
            - 'ct_warped': List of warped CT volumes
            - 'displacements': List of displacement fields
            - 'displacement_magnitudes': List of actual max displacements used
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    ct_warped_list = []
    displacement_list = []
    magnitude_list = []

    min_disp, max_disp = displacement_range_mm

    for i in range(num_samples):
        # Sample displacement magnitude
        if num_samples == 1:
            mag = max_disp
        else:
            mag = min_disp + (max_disp - min_disp) * i / (num_samples - 1)

        magnitude_list.append(mag)

        # Generate displacement field
        displacement = generate_synthetic_warp(
            shape=ct.shape,
            max_displacement_mm=mag,
            voxel_spacing=voxel_spacing,
            smoothness_sigma=5.0,
            random_seed=random_seed + i if random_seed is not None else None
        )

        # Apply to CT
        ct_warped = apply_displacement_field(
            ct, displacement, order=1, mode='constant', cval=ct.min()
        )

        ct_warped_list.append(ct_warped)
        displacement_list.append(displacement)

    return {
        'ct_warped': ct_warped_list,
        'displacements': displacement_list,
        'displacement_magnitudes': magnitude_list
    }


def compute_registration_error_map(
    displacement: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Compute per-voxel registration error magnitude in mm

    Args:
        displacement: (3, D, H, W) displacement field in voxels
        voxel_spacing: Voxel spacing in mm

    Returns:
        error_map: (D, H, W) error magnitude in mm
    """
    # Convert displacement from voxels to mm
    displacement_mm = displacement.copy()
    displacement_mm[0] *= voxel_spacing[0]
    displacement_mm[1] *= voxel_spacing[1]
    displacement_mm[2] *= voxel_spacing[2]

    # Compute magnitude
    error_map = np.sqrt(np.sum(displacement_mm ** 2, axis=0))

    return error_map


def evaluate_with_registration_stress(
    model,
    ct: torch.Tensor,
    mri: torch.Tensor,
    brain_mask: Optional[torch.Tensor] = None,
    bone_mask: Optional[torch.Tensor] = None,
    lesion_gt: Optional[torch.Tensor] = None,
    displacement_range_mm: Tuple[float, float] = (1.0, 3.0),
    num_stress_samples: int = 5,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    metric_fn=None
) -> Dict:
    """
    Evaluate model robustness under synthetic mis-registration

    Args:
        model: Fusion model
        ct: (1, 1, D, H, W) CT tensor
        mri: (1, 1, D, H, W) MRI tensor
        brain_mask: Optional brain mask
        bone_mask: Optional bone mask
        lesion_gt: Optional lesion ground truth
        displacement_range_mm: Range of synthetic displacements
        num_stress_samples: Number of stress test samples
        voxel_spacing: Voxel spacing
        metric_fn: Function to compute metrics (takes fused, mri, ct, masks)

    Returns:
        dict with metrics for each stress level
    """
    ct_np = ct[0, 0].cpu().numpy()
    mri_np = mri[0, 0].cpu().numpy()

    # Generate stress test samples
    stress_data = stress_test_registration(
        ct_np, mri_np,
        displacement_range_mm=displacement_range_mm,
        num_samples=num_stress_samples,
        voxel_spacing=voxel_spacing
    )

    results = []

    # Evaluate on each stress sample
    for idx, (ct_warped, disp, mag) in enumerate(zip(
        stress_data['ct_warped'],
        stress_data['displacements'],
        stress_data['displacement_magnitudes']
    )):
        # Convert back to tensor
        ct_warped_tensor = torch.from_numpy(ct_warped).float().unsqueeze(0).unsqueeze(0).to(ct.device)

        # Run model
        with torch.no_grad():
            fused = model(ct_warped_tensor, mri)

        # Compute metrics if provided
        if metric_fn is not None:
            metrics = metric_fn(
                fused, mri, ct_warped_tensor,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_gt=lesion_gt
            )
        else:
            metrics = {}

        # Compute registration error statistics
        error_map = compute_registration_error_map(disp, voxel_spacing)

        results.append({
            'displacement_mm': mag,
            'mean_error_mm': error_map.mean(),
            'max_error_mm': error_map.max(),
            'metrics': metrics
        })

    return {
        'stress_levels': stress_data['displacement_magnitudes'],
        'results': results
    }


def analyze_robustness_trends(results: Dict) -> Dict[str, np.ndarray]:
    """
    Analyze how metrics degrade with increasing mis-registration

    Args:
        results: Output from evaluate_with_registration_stress

    Returns:
        dict with per-metric degradation curves
    """
    stress_levels = results['stress_levels']
    results_list = results['results']

    # Extract all metric keys
    if len(results_list) > 0 and 'metrics' in results_list[0]:
        metric_keys = results_list[0]['metrics'].keys()
    else:
        return {}

    # Build degradation curves
    degradation = {}
    for key in metric_keys:
        values = [r['metrics'].get(key, np.nan) for r in results_list]
        degradation[key] = np.array(values)

    degradation['stress_levels_mm'] = np.array(stress_levels)

    return degradation


def compute_robustness_score(
    degradation: Dict[str, np.ndarray],
    metric_name: str,
    threshold: float = 0.9
) -> float:
    """
    Compute robustness score: maximum stress level before metric drops below threshold

    Args:
        degradation: Output from analyze_robustness_trends
        metric_name: Name of metric to evaluate
        threshold: Threshold ratio (e.g., 0.9 = 90% of baseline)

    Returns:
        max_stress_mm: Maximum stress level where metric >= threshold * baseline
    """
    if metric_name not in degradation:
        return 0.0

    values = degradation[metric_name]
    stress_levels = degradation['stress_levels_mm']

    if len(values) == 0:
        return 0.0

    baseline = values[0]  # Assume first is no stress (or minimal)

    # Find where metric drops below threshold
    threshold_value = threshold * baseline

    robust_levels = stress_levels[values >= threshold_value]

    if len(robust_levels) > 0:
        return robust_levels.max()
    else:
        return 0.0