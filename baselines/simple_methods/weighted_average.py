#!/usr/bin/env python3
"""
Weighted Average Fusion Baseline

ROI-adaptive weighted fusion: F = α·MRI + β·CT
where α and β are optimized per ROI to maximize SSIM

Usage:
    python baselines/simple_methods/weighted_average.py \
        --data-dir data/apis/preproc \
        --split test \
        --optimize-weights \
        --output work/results/baselines/weighted
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from scipy.optimize import minimize

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fusion_dataset import load_splits
from src.utils.roi_metrics import ROIMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted Average Fusion Baseline")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--splits-dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--optimize-weights', action='store_true',
                        help='Optimize α and β to maximize SSIM (otherwise use 0.5/0.5)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='MRI weight (if not optimizing)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='CT weight (if not optimizing)')
    return parser.parse_args()


def weighted_fusion(mri: np.ndarray, ct: np.ndarray,
                   alpha: float = 0.5, beta: float = 0.5,
                   brain_mask: np.ndarray = None,
                   bone_mask: np.ndarray = None) -> np.ndarray:
    """
    ROI-adaptive weighted fusion

    Args:
        mri: MRI volume (D, H, W)
        ct: CT volume (D, H, W)
        alpha: MRI weight
        beta: CT weight
        brain_mask: Brain ROI mask (favor MRI here)
        bone_mask: Bone ROI mask (favor CT here)

    Returns:
        fused: Weighted fused volume
    """
    # Default: balanced weights
    fused = alpha * mri + beta * ct

    # ROI-adaptive: increase MRI weight in brain, CT weight in bone
    if brain_mask is not None and bone_mask is not None:
        fused = np.zeros_like(mri)

        # Brain region: 80% MRI, 20% CT
        brain_mask_bool = brain_mask > 0.5
        fused[brain_mask_bool] = 0.8 * mri[brain_mask_bool] + 0.2 * ct[brain_mask_bool]

        # Bone region: 20% MRI, 80% CT
        bone_mask_bool = bone_mask > 0.5
        fused[bone_mask_bool] = 0.2 * mri[bone_mask_bool] + 0.8 * ct[bone_mask_bool]

        # Other regions: balanced
        other_mask = ~(brain_mask_bool | bone_mask_bool)
        fused[other_mask] = alpha * mri[other_mask] + beta * ct[other_mask]

    return fused


def optimize_weights_for_case(mri: np.ndarray, ct: np.ndarray,
                               brain_mask: np.ndarray = None) -> tuple:
    """
    Optimize α and β to maximize SSIM in brain region

    Returns:
        (alpha_opt, beta_opt)
    """
    from skimage.metrics import structural_similarity as ssim

    def objective(weights):
        alpha, beta = weights
        # Constraint: α + β = 1
        beta = 1.0 - alpha

        fused = alpha * mri + beta * ct

        # Compute SSIM in brain region (if available)
        if brain_mask is not None and brain_mask.sum() > 100:
            mask_bool = brain_mask > 0.5
            fused_roi = fused[mask_bool]
            mri_roi = mri[mask_bool]

            # SSIM requires 2D, so we compute per-slice and average
            ssim_scores = []
            for z in range(mri.shape[0]):
                if mask_bool[z].sum() > 10:  # Enough pixels in this slice
                    try:
                        score = ssim(mri_roi, fused_roi, data_range=1.0)
                        ssim_scores.append(score)
                    except:
                        pass

            if ssim_scores:
                return -np.mean(ssim_scores)  # Negative because we minimize

        # Fallback: maximize SSIM globally
        try:
            score = ssim(mri, fused, data_range=1.0)
            return -score
        except:
            return 0.0

    # Optimize
    result = minimize(
        objective,
        x0=[0.5],  # Start with balanced
        bounds=[(0.1, 0.9)],  # α must be in [0.1, 0.9]
        method='L-BFGS-B'
    )

    alpha_opt = result.x[0]
    beta_opt = 1.0 - alpha_opt

    return alpha_opt, beta_opt


def load_case(case_dir: Path):
    """Load case data"""
    ct = nib.load(case_dir / 'ct.nii.gz').get_fdata()
    mri = nib.load(case_dir / 'mri.nii.gz').get_fdata()

    brain_mask = None
    bone_mask = None
    lesion_mask = None

    if (case_dir / 'brain_mask.nii.gz').exists():
        brain_mask = nib.load(case_dir / 'brain_mask.nii.gz').get_fdata()
    if (case_dir / 'bone_mask.nii.gz').exists():
        bone_mask = nib.load(case_dir / 'bone_mask.nii.gz').get_fdata()
    if (case_dir / 'lesion_mask.nii.gz').exists():
        lesion_mask = nib.load(case_dir / 'lesion_mask.nii.gz').get_fdata()

    return {
        'ct': ct,
        'mri': mri,
        'brain_mask': brain_mask,
        'bone_mask': bone_mask,
        'lesion_mask': lesion_mask
    }


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1]"""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        return (volume - vmin) / (vmax - vmin)
    return volume


def main():
    args = parse_args()

    print("=" * 70)
    print("Weighted Average Fusion Baseline")
    print("=" * 70)
    print()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    if args.splits_dir:
        splits_dir = Path(args.splits_dir)
    else:
        splits_dir = data_dir.parent / 'splits'

    print(f"Loading splits from: {splits_dir}")
    splits = load_splits(splits_dir)
    case_ids = splits[args.split]
    print(f"✓ Loaded {len(case_ids)} cases for '{args.split}' split")

    if args.optimize_weights:
        print("✓ Weight optimization enabled")
    else:
        print(f"✓ Using fixed weights: α={args.alpha}, β={args.beta}")
    print()

    # Initialize metrics
    roi_metrics = ROIMetrics()
    all_case_metrics = []
    all_weights = []

    # Process each case
    print("Processing cases...")
    for case_id in tqdm(case_ids):
        case_dir = data_dir / case_id

        if not case_dir.exists():
            print(f"⚠ Skipping {case_id}: directory not found")
            continue

        # Load data
        data = load_case(case_dir)

        # Normalize
        mri_norm = normalize_volume(data['mri'])
        ct_norm = normalize_volume(data['ct'])

        # Optimize or use fixed weights
        if args.optimize_weights:
            alpha, beta = optimize_weights_for_case(
                mri_norm, ct_norm, data['brain_mask']
            )
            all_weights.append({'case_id': case_id, 'alpha': alpha, 'beta': beta})
        else:
            alpha, beta = args.alpha, args.beta

        # Fuse with weighted average
        fused = weighted_fusion(
            mri_norm, ct_norm, alpha, beta,
            data['brain_mask'], data['bone_mask']
        )

        # Compute metrics
        import torch
        fused_t = torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0)
        mri_t = torch.from_numpy(mri_norm).float().unsqueeze(0).unsqueeze(0)
        ct_t = torch.from_numpy(ct_norm).float().unsqueeze(0).unsqueeze(0)

        brain_mask_t = None
        bone_mask_t = None
        lesion_gt_t = None

        if data['brain_mask'] is not None:
            brain_mask_t = torch.from_numpy(data['brain_mask']).float().unsqueeze(0).unsqueeze(0)
        if data['bone_mask'] is not None:
            bone_mask_t = torch.from_numpy(data['bone_mask']).float().unsqueeze(0).unsqueeze(0)
        if data['lesion_mask'] is not None:
            lesion_gt_t = torch.from_numpy(data['lesion_mask']).float().unsqueeze(0).unsqueeze(0)

        metrics = roi_metrics.compute_all(
            fused=fused_t,
            mri=mri_t,
            ct=ct_t,
            brain_mask=brain_mask_t,
            bone_mask=bone_mask_t,
            lesion_pred=None,
            lesion_gt=lesion_gt_t
        )

        metrics['case_id'] = case_id
        metrics['alpha'] = alpha
        metrics['beta'] = beta
        all_case_metrics.append(metrics)

        # Save fused image if requested
        if args.save_images:
            images_dir = output_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            fused_nii = nib.Nifti1Image(fused, affine=np.eye(4))
            nib.save(fused_nii, images_dir / f"{case_id}_fused.nii.gz")

    # Save results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    # Save per-case metrics
    df = pd.DataFrame(all_case_metrics)
    csv_path = output_dir / 'per_case_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved per-case metrics: {csv_path}")

    # Save weights if optimized
    if args.optimize_weights and all_weights:
        weights_df = pd.DataFrame(all_weights)
        weights_path = output_dir / 'optimized_weights.csv'
        weights_df.to_csv(weights_path, index=False)
        print(f"✓ Saved optimized weights: {weights_path}")
        print(f"  Mean α: {weights_df['alpha'].mean():.3f} ± {weights_df['alpha'].std():.3f}")
        print(f"  Mean β: {weights_df['beta'].mean():.3f} ± {weights_df['beta'].std():.3f}")

    # Compute aggregate statistics
    aggregate_metrics = {}
    metric_keys = [k for k in all_case_metrics[0].keys() if k not in ['case_id', 'alpha', 'beta']]

    for key in metric_keys:
        values = [m[key] for m in all_case_metrics if key in m and m[key] is not None]
        if values:
            aggregate_metrics[key + '/mean'] = float(np.mean(values))
            aggregate_metrics[key + '/std'] = float(np.std(values))
            aggregate_metrics[key + '/median'] = float(np.median(values))
            aggregate_metrics[key + '/min'] = float(np.min(values))
            aggregate_metrics[key + '/max'] = float(np.max(values))

    json_path = output_dir / 'aggregate_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"✓ Saved aggregate metrics: {json_path}")

    # Print summary
    print()
    print("Key Metrics (mean ± std):")
    primary_metrics = [
        'lesion/dice', 'lesion/nsd', 'lesion/hd95',
        'brain/ssim', 'brain/fsim',
        'bone/psnr', 'bone/ssim',
        'global/psnr', 'global/ssim'
    ]

    for metric in primary_metrics:
        mean_key = metric + '/mean'
        std_key = metric + '/std'
        if mean_key in aggregate_metrics:
            mean_val = aggregate_metrics[mean_key]
            std_val = aggregate_metrics.get(std_key, 0)
            print(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    print()
    print("=" * 70)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()