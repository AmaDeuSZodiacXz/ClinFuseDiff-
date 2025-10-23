#!/usr/bin/env python3
"""
Training script for CLIN-FuseDiff++ (CVPR 2026)

ROI-Aware Guided Diffusion for Medical Image Fusion
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fusion_dataset import ImageFusionDataset, load_splits
from src.models.unet3d import ImageFusionDiffusion
from src.models.roi_guided_diffusion import ROIGuidedDiffusion
from src.models.lesion_head import create_lesion_head
from src.training.fusion_trainer import FusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CLIN-FuseDiff++ for medical image fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train.py --config configs/cvpr2026/train_roi.yaml

  # Train with specific disease preset
  python train.py --config configs/cvpr2026/train_roi.yaml --preset stroke

  # Train with custom ROI weights
  python train.py --config configs/cvpr2026/train_roi.yaml --alpha 1.5 --beta 0.5 --gamma 2.0

  # Resume from checkpoint
  python train.py --config configs/cvpr2026/train_roi.yaml --resume work/experiments/clinfusediff_cvpr2026/checkpoints/last.pth
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--preset',
        type=str,
        choices=['default', 'stroke', 'brain_tumor', 'bone_tumor', 'metastasis'],
        help='Disease-specific preset (overrides config ROI weights)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        help='Brain ROI weight (overrides config)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        help='Bone ROI weight (overrides config)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        help='Lesion ROI weight (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on (default: cuda if available)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of data loader workers (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_preset(config: dict, preset: str):
    """Apply disease-specific preset to config"""
    if preset in config['disease_presets']:
        preset_config = config['disease_presets'][preset]
        config['roi_guidance']['alpha'] = preset_config['alpha']
        config['roi_guidance']['beta'] = preset_config['beta']
        config['roi_guidance']['gamma'] = preset_config['gamma']
        print(f"✓ Applied preset: {preset}")
        print(f"  Description: {preset_config['description']}")
        print(f"  α (brain): {preset_config['alpha']}")
        print(f"  β (bone): {preset_config['beta']}")
        print(f"  γ (lesion): {preset_config['gamma']}")
    else:
        print(f"WARNING: Unknown preset '{preset}', using config values")


def override_config(config: dict, args):
    """Override config with command-line arguments"""
    if args.alpha is not None:
        config['roi_guidance']['alpha'] = args.alpha
        print(f"✓ Overriding α (brain): {args.alpha}")

    if args.beta is not None:
        config['roi_guidance']['beta'] = args.beta
        print(f"✓ Overriding β (bone): {args.beta}")

    if args.gamma is not None:
        config['roi_guidance']['gamma'] = args.gamma
        print(f"✓ Overriding γ (lesion): {args.gamma}")

    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.wandb:
        config['experiment']['use_wandb'] = True

    if args.no_wandb:
        config['experiment']['use_wandb'] = False


def create_dataloaders(config: dict):
    """Create train and validation dataloaders"""
    data_config = config['data']

    # Load splits
    splits_dir = Path(data_config['splits_dir'])
    if not splits_dir.exists():
        raise FileNotFoundError(
            f"Splits directory not found: {splits_dir}\n"
            "Please run: python scripts/make_splits.py"
        )

    splits = load_splits(splits_dir)

    # Create datasets
    train_dataset = ImageFusionDataset(
        data_dir=data_config['data_dir'],
        cases=splits['train'],
        normalize=True,
        cache_in_memory=False  # Too large for memory
    )

    val_dataset = ImageFusionDataset(
        data_dir=data_config['data_dir'],
        cases=splits['val'],
        normalize=True,
        cache_in_memory=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True),
        prefetch_factor=data_config.get('prefetch_factor', 2)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config.get('pin_memory', True)
    )

    print(f"✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} cases, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} cases, {len(val_loader)} batches")

    return train_loader, val_loader


def create_model(config: dict, device: str):
    """Create model components"""
    print("Creating model...")

    # Create image fusion diffusion model (U-Net + encoders)
    model_config = config['model']
    fusion_model = ImageFusionDiffusion(
        image_channels=model_config['unet3d']['in_channels'],
        cond_dim=model_config['unet3d']['cond_channels'] // 2,  # Each encoder outputs cond_dim
        unet_base_channels=model_config['unet3d']['base_channels'],
        time_emb_dim=256,
        channel_multipliers=tuple(model_config['unet3d']['channel_mult']),
        attention_resolutions=tuple(model_config['unet3d']['attention_resolutions']),
        num_res_blocks=model_config['unet3d']['num_res_blocks']
    ).to(device)

    # Wrap with ROI-guided diffusion
    diffusion_config = model_config['diffusion']
    roi_config = config['roi_guidance']

    roi_model = ROIGuidedDiffusion(
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
        eta_u=roi_config['eta_u'],
        use_uncertainty_modulation=config['uncertainty']['enabled'],
        kappa=config['uncertainty']['kappa']
    ).to(device)

    # Create lesion segmentation head
    lesion_config = model_config.get('lesion_head', {})
    if lesion_config.get('enabled', False):
        print("Creating lesion segmentation head...")
        lesion_head = create_lesion_head(
            config=lesion_config,
            pretrained_path=lesion_config.get('pretrained', None)
        )
        print("✓ Lesion head created (frozen)")
    else:
        print("⚠ Lesion head disabled (lesion boundary guidance will not be used)")
        lesion_head = None

    # Count parameters
    total_params = sum(p.numel() for p in roi_model.parameters())
    trainable_params = sum(p.numel() for p in roi_model.parameters() if p.requires_grad)

    print(f"✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return roi_model, lesion_head


def main():
    args = parse_args()

    print("=" * 70)
    print("CLIN-FuseDiff++ Training (CVPR 2026)")
    print("ROI-Aware Guided Diffusion for Medical Image Fusion")
    print("=" * 70)
    print("")

    # Load configuration
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Apply preset if specified
    if args.preset:
        apply_preset(config, args.preset)

    # Override with command-line arguments
    override_config(config, args)

    # Set random seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed: {seed}")

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    print(f"✓ Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print("")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print("")

    # Create model
    model, lesion_head = create_model(config, device)
    print("")

    # Create trainer
    print("Creating trainer...")
    trainer = FusionTrainer(
        model=model,
        lesion_head=lesion_head,
        config=config,
        device=device
    )
    print("")

    # Start training
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print("")

    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=args.resume
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted by user")
        print("=" * 70)
        # Save checkpoint on interrupt
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={},
            is_best=False
        )
        print(f"✓ Saved checkpoint: {trainer.checkpoint_dir}/last.pth")

    print("")
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()