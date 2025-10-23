#!/usr/bin/env python3
"""
Evaluation script for CLIN-FuseDiff++ (CVPR 2026)

Evaluate trained model on test set with comprehensive ROI-aware metrics.
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fusion_dataset import ImageFusionDataset, load_splits
from src.models.unet3d import ImageFusionDiffusion
from src.models.roi_guided_diffusion import ROIGuidedDiffusion
from src.models.lesion_head import create_lesion_head
from src.utils.roi_metrics import ROIMetrics
from src.utils.uncertainty import compute_calibration_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIN-FuseDiff++ on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='work/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save fused images'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of diffusion samples per case (for uncertainty)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict, checkpoint_path: str, device: str):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Create model architecture
    model_config = config['model']
    fusion_model = ImageFusionDiffusion(
        in_channels=model_config['unet3d']['in_channels'],
        cond_channels=model_config['unet3d']['cond_channels'],
        out_channels=model_config['unet3d']['out_channels'],
        base_channels=model_config['unet3d']['base_channels'],
        channel_mult=model_config['unet3d']['channel_mult'],
        num_res_blocks=model_config['unet3d']['num_res_blocks'],
        attention_resolutions=model_config['unet3d']['attention_resolutions'],
        dropout=model_config['unet3d']['dropout']
    )

    diffusion_config = model_config['diffusion']
    roi_config = config['roi_guidance']

    model = ROIGuidedDiffusion(
        image_fusion_model=fusion_model,
        num_timesteps=diffusion_config['num_timesteps'],
        beta_schedule=diffusion_config['beta_schedule'],
        alpha=roi_config['alpha'],
        beta=roi_config['beta'],
        gamma=roi_config['gamma'],
        lambda_dice=roi_config['lesion_weights']['dice'],
        lambda_nsd=roi_config['lesion_weights']['nsd'],
        lambda_hd95=roi_config['lesion_weights']['hd95'],
        eta=roi_config['eta'],
        eta_u=roi_config['eta_u']
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    # Load lesion head if available
    lesion_config = model_config.get('lesion_head', {})
    if lesion_config.get('enabled', False):
        lesion_head = create_lesion_head(
            config=lesion_config,
            pretrained_path=lesion_config.get('pretrained', None)
        )
        lesion_head = lesion_head.to(device)
        lesion_head.eval()
        print("✓ Lesion head loaded")
    else:
        lesion_head = None

    return model, lesion_head


def create_dataloader(config: dict, split: str, batch_size: int):
    """Create evaluation dataloader"""
    data_config = config['data']

    # Load splits
    splits_dir = Path(data_config['splits_dir'])
    splits = load_splits(splits_dir)

    if split not in splits:
        raise ValueError(f"Split '{split}' not found in {splits_dir}")

    # Create dataset
    dataset = ImageFusionDataset(
        data_dir=data_config['data_dir'],
        cases=splits[split],
        normalize=True,
        cache_in_memory=False
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✓ Created dataloader for '{split}' split:")
    print(f"  Cases: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")

    return dataloader


@torch.no_grad()
def evaluate(
    model,
    lesion_head,
    dataloader,
    device,
    num_samples=5,
    save_images=False,
    output_dir=None
):
    """
    Run evaluation on dataloader

    Args:
        model: ROI-guided diffusion model
        lesion_head: Lesion segmentation head
        dataloader: Evaluation dataloader
        device: Device to use
        num_samples: Number of samples for uncertainty estimation
        save_images: Whether to save fused images
        output_dir: Output directory

    Returns:
        results: Dict with per-case and aggregate metrics
    """
    model.eval()
    roi_metrics = ROIMetrics()

    all_case_metrics = []
    all_case_ids = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, batch in enumerate(pbar):
        ct = batch['ct'].to(device)
        mri = batch['mri'].to(device)
        brain_mask = batch.get('brain_mask', None)
        bone_mask = batch.get('bone_mask', None)
        lesion_mask = batch.get('lesion_mask', None)
        case_id = batch['case_id']

        if brain_mask is not None:
            brain_mask = brain_mask.to(device)
        if bone_mask is not None:
            bone_mask = bone_mask.to(device)
        if lesion_mask is not None:
            lesion_mask = lesion_mask.to(device)

        # Generate multiple samples for uncertainty estimation
        fused_samples = []
        for _ in range(num_samples):
            fused = model.sample(
                batch_size=ct.shape[0],
                shape=(1, *ct.shape[2:]),  # (C, D, H, W)
                mri=mri,
                ct=ct,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_gt=lesion_mask,
                lesion_head=lesion_head
            )
            fused_samples.append(fused)

        # Stack samples
        fused_samples = torch.stack(fused_samples, dim=0)  # (num_samples, B, C, D, H, W)

        # Use mean for evaluation
        fused_mean = fused_samples.mean(dim=0)

        # Compute uncertainty (std across samples)
        fused_std = fused_samples.std(dim=0)

        # Predict lesion from fused image
        lesion_pred = None
        if lesion_head is not None:
            lesion_pred = lesion_head(fused_mean)

        # Compute metrics
        batch_metrics = roi_metrics.compute_all(
            fused=fused_mean,
            mri=mri,
            ct=ct,
            brain_mask=brain_mask,
            bone_mask=bone_mask,
            lesion_pred=lesion_pred,
            lesion_gt=lesion_mask
        )

        # Add uncertainty metrics
        batch_metrics['uncertainty/mean'] = fused_std.mean().item()
        batch_metrics['uncertainty/max'] = fused_std.max().item()

        # Compute calibration metrics (ECE, Brier) for lesion region
        if lesion_mask is not None:
            # Convert to numpy
            fused_samples_np = fused_samples.cpu().numpy()  # (num_samples, B, C, D, H, W)
            lesion_mask_np = lesion_mask.cpu().numpy()  # (B, C, D, H, W)

            # For each case in batch
            for b in range(ct.shape[0]):
                # Get samples for this case
                case_samples = fused_samples_np[:, b, 0]  # (num_samples, D, H, W)
                case_mask = lesion_mask_np[b, 0]  # (D, H, W)

                # Ground truth: use MRI in lesion region (or average of MRI/CT)
                case_mri = mri[b, 0].cpu().numpy()  # (D, H, W)

                # Compute calibration metrics
                try:
                    calib_metrics = compute_calibration_metrics(
                        ensemble_samples=case_samples,
                        ground_truth=case_mri,
                        mask=case_mask,
                        n_bins=10
                    )
                    batch_metrics['calibration/ece'] = calib_metrics['ece']
                    batch_metrics['calibration/brier'] = calib_metrics['brier']
                except Exception as e:
                    # Skip if calibration fails
                    batch_metrics['calibration/ece'] = None
                    batch_metrics['calibration/brier'] = None

        # Store results
        for i, cid in enumerate(case_id):
            case_metrics = {k: v for k, v in batch_metrics.items()}
            case_metrics['case_id'] = cid
            all_case_metrics.append(case_metrics)
            all_case_ids.append(cid)

        # Save images if requested
        if save_images and output_dir is not None:
            import nibabel as nib

            output_dir = Path(output_dir) / 'images'
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, cid in enumerate(case_id):
                # Save fused image
                fused_img = fused_mean[i, 0].cpu().numpy()
                nib.save(
                    nib.Nifti1Image(fused_img, affine=np.eye(4)),
                    output_dir / f"{cid}_fused.nii.gz"
                )

                # Save uncertainty map
                uncertainty_img = fused_std[i, 0].cpu().numpy()
                nib.save(
                    nib.Nifti1Image(uncertainty_img, affine=np.eye(4)),
                    output_dir / f"{cid}_uncertainty.nii.gz"
                )

                # Save lesion prediction if available
                if lesion_pred is not None:
                    lesion_img = lesion_pred[i, 0].cpu().numpy()
                    nib.save(
                        nib.Nifti1Image(lesion_img, affine=np.eye(4)),
                        output_dir / f"{cid}_lesion_pred.nii.gz"
                    )

        # Update progress
        pbar.set_postfix({
            'SSIM_brain': f"{batch_metrics.get('brain/ssim', 0):.3f}",
            'Dice': f"{batch_metrics.get('lesion/dice', 0):.3f}"
        })

    # Aggregate metrics
    aggregate_metrics = {}
    if all_case_metrics:
        # Get all metric keys (excluding case_id)
        metric_keys = [k for k in all_case_metrics[0].keys() if k != 'case_id']

        for key in metric_keys:
            values = [m[key] for m in all_case_metrics if key in m and m[key] is not None]
            if values:
                aggregate_metrics[key + '/mean'] = float(np.mean(values))
                aggregate_metrics[key + '/std'] = float(np.std(values))
                aggregate_metrics[key + '/median'] = float(np.median(values))
                aggregate_metrics[key + '/min'] = float(np.min(values))
                aggregate_metrics[key + '/max'] = float(np.max(values))

    results = {
        'per_case': all_case_metrics,
        'aggregate': aggregate_metrics,
        'num_cases': len(all_case_ids)
    }

    return results


def save_results(results: dict, output_dir: Path):
    """Save evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-case metrics as CSV
    if results['per_case']:
        df = pd.DataFrame(results['per_case'])
        csv_path = output_dir / 'per_case_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved per-case metrics: {csv_path}")

    # Save aggregate metrics as JSON
    json_path = output_dir / 'aggregate_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results['aggregate'], f, indent=2)
    print(f"✓ Saved aggregate metrics: {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70)
    print(f"Number of cases: {results['num_cases']}")
    print("\nKey Metrics (mean ± std):")

    # Primary metrics
    primary = [
        'lesion/dice',
        'lesion/nsd',
        'lesion/hd95',
        'brain/ssim',
        'brain/fsim',
        'bone/psnr',
        'bone/ssim',
        'calibration/ece',
        'calibration/brier',
        'uncertainty/mean'
    ]

    for metric in primary:
        mean_key = metric + '/mean'
        std_key = metric + '/std'
        if mean_key in results['aggregate']:
            mean_val = results['aggregate'][mean_key]
            std_val = results['aggregate'].get(std_key, 0)
            print(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    print("=" * 70)


def main():
    args = parse_args()

    print("=" * 70)
    print("CLIN-FuseDiff++ Evaluation (CVPR 2026)")
    print("=" * 70)
    print("")

    # Load config
    config = load_config(args.config)
    print(f"✓ Loaded config: {args.config}")

    # Load model
    model, lesion_head = load_model(config, args.checkpoint, args.device)

    # Create dataloader
    dataloader = create_dataloader(config, args.split, args.batch_size)
    print("")

    # Run evaluation
    print("=" * 70)
    print(f"Evaluating on '{args.split}' split...")
    print("=" * 70)
    print("")

    results = evaluate(
        model=model,
        lesion_head=lesion_head,
        dataloader=dataloader,
        device=args.device,
        num_samples=args.num_samples,
        save_images=args.save_images,
        output_dir=args.output
    )

    # Save results
    print("")
    save_results(results, args.output)

    print("")
    print("=" * 70)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()