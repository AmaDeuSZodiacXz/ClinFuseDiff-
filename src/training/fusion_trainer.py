"""
Training Framework for Image-Level Diffusion Fusion (CVPR 2026)

Implements end-to-end training loop for ROI-aware guided diffusion fusion.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Optional WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import models and losses
from ..models.unet3d import ImageFusionDiffusion
from ..models.roi_guided_diffusion import ROIGuidedDiffusion
from ..models.lesion_head import create_lesion_head
from .roi_losses import ClinicalROILoss
from ..utils.roi_metrics import ROIMetrics


class FusionTrainer:
    """
    Trainer for ROI-aware diffusion fusion

    Implements training loop with:
    - Diffusion loss (standard denoising)
    - ROI-aware loss (brain, bone, lesion)
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Metrics tracking
    """

    def __init__(
        self,
        model: ROIGuidedDiffusion,
        lesion_head: Optional[nn.Module],
        config: dict,
        device: str = 'cuda',
        experiment_dir: Optional[str] = None
    ):
        """
        Args:
            model: ROI-guided diffusion model
            lesion_head: Lesion segmentation head (frozen)
            config: Training configuration dict
            device: Device to train on
            experiment_dir: Directory to save checkpoints/logs
        """
        self.model = model.to(device)
        self.lesion_head = lesion_head
        if lesion_head is not None:
            self.lesion_head = lesion_head.to(device)
            self.lesion_head.eval()  # Always in eval mode

        self.config = config
        self.device = device

        # Setup experiment directory
        if experiment_dir is None:
            experiment_dir = Path(config['experiment']['save_dir']) / config['experiment']['name']
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup WandB logging
        self.use_wandb = config['experiment'].get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config['experiment'].get('wandb_project', 'clinfusediff'),
                name=config['experiment']['name'],
                config=config,
                dir=str(self.experiment_dir)
            )
            wandb.watch(self.model, log='all', log_freq=100)

        # Setup ROI loss
        roi_config = config['roi_guidance']
        self.roi_loss = ClinicalROILoss(
            alpha=roi_config['alpha'],
            beta=roi_config['beta'],
            gamma=roi_config['gamma'],
            lambda_dice=roi_config['lesion_weights']['dice'],
            lambda_nsd=roi_config['lesion_weights']['nsd'],
            lambda_hd95=roi_config['lesion_weights']['hd95'],
            tolerance_mm=roi_config['nsd_tolerance_mm']
        ).to(device)

        # Setup optimizer
        train_config = config['training']
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=tuple(train_config.get('betas', [0.9, 0.999])),
            eps=train_config.get('eps', 1e-8)
        )

        # Setup scheduler
        self.num_epochs = train_config['num_epochs']
        warmup_epochs = train_config.get('warmup_epochs', 10)

        # Warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - warmup_epochs,
            eta_min=train_config.get('min_lr', 1e-6)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )

        # Mixed precision training
        self.use_amp = train_config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)

        # Loss weights
        self.loss_weights = train_config['loss_weights']

        # Metrics
        self.roi_metrics = ROIMetrics()

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.train_losses = []
        self.val_losses = []

        print(f"✓ FusionTrainer initialized")
        print(f"  Device: {device}")
        print(f"  Experiment dir: {self.experiment_dir}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Gradient accumulation: {self.grad_accum_steps}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'diffusion': 0.0,
            'roi': 0.0,
            'roi_brain': 0.0,
            'roi_bone': 0.0,
            'roi_lesion': 0.0
        }

        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            ct = batch['ct'].to(self.device)
            mri = batch['mri'].to(self.device)
            brain_mask = batch.get('brain_mask', None)
            bone_mask = batch.get('bone_mask', None)
            lesion_mask = batch.get('lesion_mask', None)

            if brain_mask is not None:
                brain_mask = brain_mask.to(self.device)
            if bone_mask is not None:
                bone_mask = bone_mask.to(self.device)
            if lesion_mask is not None:
                lesion_mask = lesion_mask.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Sample random timestep
                batch_size = ct.shape[0]
                t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)

                # Create target fused image (simple average as starting point)
                # In practice, this could be more sophisticated
                target_fused = (mri + ct) / 2.0

                # Add noise to target (forward diffusion)
                noise = torch.randn_like(target_fused)
                noisy_fused = self.model.q_sample(target_fused, t, noise)

                # Predict noise using U-Net conditioned on MRI/CT
                predicted_noise = self.model.model(noisy_fused, t, mri=mri, ct=ct)

                # Diffusion loss (MSE between predicted and actual noise)
                loss_diffusion = F.mse_loss(predicted_noise, noise)

                # Reconstruct denoised image for ROI loss
                # x_0 = (noisy - sqrt(1-alpha) * predicted_noise) / sqrt(alpha)
                sqrt_alphas_cumprod_t = self.model.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alphas_cumprod_t = self.model.sqrt_one_minus_alphas_cumprod[t]

                # Reshape for broadcasting
                while len(sqrt_alphas_cumprod_t.shape) < len(noisy_fused.shape):
                    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
                    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

                pred_x0 = (noisy_fused - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

                # Compute ROI loss
                lesion_pred = None
                if self.lesion_head is not None and lesion_mask is not None:
                    with torch.no_grad():
                        lesion_pred = self.lesion_head(pred_x0)

                loss_roi, roi_loss_dict = self.roi_loss(
                    fused=pred_x0,
                    mri=mri,
                    ct=ct,
                    brain_mask=brain_mask,
                    bone_mask=bone_mask,
                    lesion_pred=lesion_pred,
                    lesion_gt=lesion_mask
                )

                # Total loss
                loss = (
                    self.loss_weights['diffusion'] * loss_diffusion +
                    self.loss_weights['roi'] * loss_roi
                )

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Track losses (unscaled)
            loss_value = loss.item() * self.grad_accum_steps
            epoch_losses['total'] += loss_value
            epoch_losses['diffusion'] += loss_diffusion.item()
            epoch_losses['roi'] += loss_roi.item()
            if 'brain' in roi_loss_dict:
                epoch_losses['roi_brain'] += roi_loss_dict['brain']
            if 'bone' in roi_loss_dict:
                epoch_losses['roi_bone'] += roi_loss_dict['bone']
            if 'lesion' in roi_loss_dict:
                epoch_losses['roi_lesion'] += roi_loss_dict['lesion']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'diff': f"{loss_diffusion.item():.4f}",
                'roi': f"{loss_roi.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to WandB
            if self.use_wandb and self.global_step % self.config['experiment'].get('log_interval', 10) == 0:
                wandb.log({
                    'train/loss_total': loss_value,
                    'train/loss_diffusion': loss_diffusion.item(),
                    'train/loss_roi': loss_roi.item(),
                    'train/loss_roi_brain': roi_loss_dict.get('brain', 0),
                    'train/loss_roi_bone': roi_loss_dict.get('bone', 0),
                    'train/loss_roi_lesion': roi_loss_dict.get('lesion', 0),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate on validation set"""
        self.model.eval()

        epoch_losses = {
            'total': 0.0,
            'diffusion': 0.0,
            'roi': 0.0
        }

        all_metrics = []

        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        for batch in pbar:
            ct = batch['ct'].to(self.device)
            mri = batch['mri'].to(self.device)
            brain_mask = batch.get('brain_mask', None)
            bone_mask = batch.get('bone_mask', None)
            lesion_mask = batch.get('lesion_mask', None)

            if brain_mask is not None:
                brain_mask = brain_mask.to(self.device)
            if bone_mask is not None:
                bone_mask = bone_mask.to(self.device)
            if lesion_mask is not None:
                lesion_mask = lesion_mask.to(self.device)

            # Sample fused image using guided diffusion
            fused = self.model.sample(
                mri=mri,
                ct=ct,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_gt=lesion_mask,
                lesion_head=self.lesion_head
            )

            # Compute metrics
            batch_metrics = self.roi_metrics.compute_all(
                fused=fused,
                mri=mri,
                ct=ct,
                brain_mask=brain_mask,
                bone_mask=bone_mask,
                lesion_pred=None,  # TODO: compute from fused
                lesion_gt=lesion_mask
            )
            all_metrics.append(batch_metrics)

        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m and m[key] is not None]
                if values:
                    avg_metrics[key] = np.mean(values)

        # Average losses (placeholder - would need to compute during validation)
        avg_losses = {k: 0.0 for k in epoch_losses.keys()}

        return avg_losses, avg_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint: {best_path}")

        # Save periodic checkpoint
        save_interval = self.config['experiment'].get('save_interval', 5)
        if epoch % save_interval == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from: Path to checkpoint to resume from
        """
        # Resume if requested
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0

        print("=" * 50)
        print("Starting training")
        print("=" * 50)
        print(f"Epochs: {start_epoch} -> {self.num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        print("=" * 50)

        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_losses)

            # Validate
            if val_loader is not None and (epoch + 1) % self.config['experiment'].get('eval_interval', 1) == 0:
                val_losses, val_metrics = self.validate(val_loader, epoch)
                self.val_losses.append(val_losses)

                # Check if best model
                primary_metric = self.config['checkpoint'].get('metric', 'lesion/nsd@2mm')
                if primary_metric in val_metrics:
                    current_metric = val_metrics[primary_metric]
                    is_best = current_metric > self.best_metric
                    if is_best:
                        self.best_metric = current_metric
                        print(f"✓ New best {primary_metric}: {current_metric:.4f}")
                else:
                    is_best = False

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

                # Log validation to WandB
                if self.use_wandb:
                    wandb_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb_metrics['val/epoch'] = epoch
                    wandb.log(wandb_metrics)

                # Print validation results
                print(f"\nEpoch {epoch} Validation:")
                print(f"  Losses: {val_losses}")
                print(f"  Metrics: {val_metrics}")
            else:
                # Save without validation
                self.save_checkpoint(epoch, {}, is_best=False)

            # Step scheduler
            self.scheduler.step()

            # Print training summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    Diffusion: {train_losses['diffusion']:.4f}")
            print(f"    ROI: {train_losses['roi']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("")

        print("=" * 50)
        print("✓ Training complete!")
        print(f"Best metric: {self.best_metric:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 50)