"""
Train SimpleLesionHead on APIS Dataset

Standalone training script to create pretrained lesion segmentation head
before integrating with fusion model.

Usage:
    python scripts/train_lesion_head.py \
        --data-dir data/apis/preproc \
        --output-dir work/lesion_head \
        --epochs 50 \
        --batch-size 4
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lesion_head import SimpleLesionHead
from src.utils.roi_metrics import ROIMetrics


class LesionDataset(Dataset):
    """
    Dataset for lesion segmentation training

    Input: Fused image (or MRI as proxy during pretraining)
    Target: Lesion mask
    """

    def __init__(self, data_dir, split='train', use_mri=True):
        """
        Args:
            data_dir: Preprocessed APIS directory
            split: 'train', 'val', or 'test'
            use_mri: Use MRI as input (True) or need fused images (False)
        """
        self.data_dir = Path(data_dir)
        self.use_mri = use_mri

        # Load split
        split_file = self.data_dir.parent / 'splits' / f'{split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                self.case_ids = [line.strip() for line in f]
        else:
            # No split file - use all cases
            self.case_ids = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            print(f"Warning: No split file found, using all {len(self.case_ids)} cases")

        print(f"Loaded {split} split: {len(self.case_ids)} cases")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_dir = self.data_dir / case_id

        # Load input (MRI or fused)
        if self.use_mri:
            input_path = case_dir / 'mri_to_ct.nii.gz'
        else:
            input_path = case_dir / 'fused.nii.gz'  # After fusion training

        # Load lesion mask
        lesion_path = case_dir / 'lesion_mask.nii.gz'

        # Check files exist
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not lesion_path.exists():
            # Some cases may not have lesions
            print(f"Warning: No lesion mask for {case_id}, creating empty mask")
            lesion = np.zeros_like(self._load_nifti(input_path))
        else:
            lesion = self._load_nifti(lesion_path)

        input_vol = self._load_nifti(input_path)

        # Normalize input (assume already preprocessed)
        # For MRI: should already be Z-score normalized
        # For fused: should be in same range as training output

        # Convert to tensors
        input_vol = torch.from_numpy(input_vol).float().unsqueeze(0)  # (1, D, H, W)
        lesion = torch.from_numpy(lesion).float().unsqueeze(0)

        return {
            'input': input_vol,
            'lesion': lesion,
            'case_id': case_id
        }

    def _load_nifti(self, path):
        """Load NIfTI file"""
        import nibabel as nib
        nii = nib.load(str(path))
        return nii.get_fdata().astype(np.float32)


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(pred, target, weight_bce=0.5, weight_dice=0.5):
    """BCE + Dice loss"""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return weight_bce * bce + weight_dice * dice


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['lesion'].to(device)

        # Forward
        outputs = model(inputs)

        # Loss
        loss = combined_loss(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            dice = 1 - dice_loss(outputs, targets)

        epoch_loss += loss.item()
        epoch_dice += dice.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice.item():.4f}"
        })

    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate"""
    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    for batch in tqdm(dataloader, desc="Validation"):
        inputs = batch['input'].to(device)
        targets = batch['lesion'].to(device)

        outputs = model(inputs)

        loss = combined_loss(outputs, targets)
        dice = 1 - dice_loss(outputs, targets)

        val_loss += loss.item()
        val_dice += dice.item()

    return val_loss / len(dataloader), val_dice / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Preprocessed APIS directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=2)

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = LesionDataset(args.data_dir, split='train', use_mri=True)
    val_dataset = LesionDataset(args.data_dir, split='val', use_mri=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} cases, Val: {len(val_dataset)} cases")

    # Create model
    model = SimpleLesionHead(
        in_channels=1,
        base_channels=args.base_channels,
        num_classes=1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_dice = validate(model, val_loader, device)
        scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_loss:.4f}, dice={train_dice:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, dice={val_dice:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice
        }

        # Save last
        torch.save(checkpoint, output_dir / 'last.pth')

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(checkpoint, output_dir / 'best.pth')
            print(f"  ✓ New best model! Dice: {val_dice:.4f}")

        # History
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice
        })

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
