#!/usr/bin/env python3
"""
Simple Average Fusion Baseline

Trivial baseline: F = (MRI + CT) / 2

Usage:
    python baselines/simple_methods/average.py \
        --data-dir data/apis/preproc \
        --split test \
        --output work/results/baselines/average
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fusion_dataset import load_splits
from src.utils.roi_metrics import ROIMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Average Fusion Baseline")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to preprocessed data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--splits-dir', type=str, default=None,
                        help='Path to splits directory (default: data_dir/../splits)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--save-images', action='store_true',
                        help='Save fused images')
    return parser.parse_args()


def simple_average_fusion(mri: np.ndarray, ct: np.ndarray) -> np.ndarray:
    """
    Simplest possible fusion: arithmetic mean

    Args:
        mri: MRI volume (D, H, W)
        ct: CT volume (D, H, W)

    Returns:
        fused: Fused volume (D, H, W)
    """
    return (mri + ct) / 2.0


def load_case(case_dir: Path):
    """Load case data"""
    ct = nib.load(case_dir / 'ct.nii.gz').get_fdata()
    mri = nib.load(case_dir / 'mri.nii.gz').get_fdata()

    # Load masks if available
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
    print("Simple Average Fusion Baseline")
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
    print()

    # Initialize metrics
    roi_metrics = ROIMetrics()
    all_case_metrics = []

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

        # Fuse: simple average
        fused = simple_average_fusion(mri_norm, ct_norm)

        # Compute metrics
        # Convert to torch format for metrics
        import torch
        fused_t = torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
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

        # Compute ROI metrics
        metrics = roi_metrics.compute_all(
            fused=fused_t,
            mri=mri_t,
            ct=ct_t,
            brain_mask=brain_mask_t,
            bone_mask=bone_mask_t,
            lesion_pred=None,  # No lesion prediction for simple average
            lesion_gt=lesion_gt_t
        )

        metrics['case_id'] = case_id
        all_case_metrics.append(metrics)

        # Save fused image if requested
        if args.save_images:
            images_dir = output_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            fused_nii = nib.Nifti1Image(fused, affine=np.eye(4))
            nib.save(fused_nii, images_dir / f"{case_id}_fused.nii.gz")

    # Aggregate results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    # Save per-case metrics
    df = pd.DataFrame(all_case_metrics)
    csv_path = output_dir / 'per_case_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved per-case metrics: {csv_path}")

    # Compute aggregate statistics
    aggregate_metrics = {}
    metric_keys = [k for k in all_case_metrics[0].keys() if k != 'case_id']

    for key in metric_keys:
        values = [m[key] for m in all_case_metrics if key in m and m[key] is not None]
        if values:
            aggregate_metrics[key + '/mean'] = float(np.mean(values))
            aggregate_metrics[key + '/std'] = float(np.std(values))
            aggregate_metrics[key + '/median'] = float(np.median(values))
            aggregate_metrics[key + '/min'] = float(np.min(values))
            aggregate_metrics[key + '/max'] = float(np.max(values))

    # Save aggregate metrics
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