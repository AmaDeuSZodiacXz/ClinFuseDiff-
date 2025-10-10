"""Test script for data loading and segmentation pipeline"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data import (
    BrainTumorDataset,
    BraTSDataset,
    create_dataloaders,
    get_train_transforms,
    TotalSegmentatorWrapper,
    BrainSegmentationPreprocessor
)


def test_totalsegmentator():
    """Test TotalSegmentator wrapper"""
    print("=" * 80)
    print("Testing TotalSegmentator Wrapper")
    print("=" * 80)

    try:
        segmentator = TotalSegmentatorWrapper(
            task='brain_structures',
            device='cpu',  # Use CPU for testing
            fast=True,
            ml=True,
            quiet=False
        )
        print("✓ TotalSegmentator initialized successfully")

        # Note: Actual segmentation requires a real NIfTI file
        print("  To segment an image, use:")
        print("  segmentation = segmentator.segment('path/to/image.nii.gz')")

    except Exception as e:
        print(f"✗ Failed to initialize TotalSegmentator: {e}")
        print("  Make sure TotalSegmentator is installed:")
        print("  pip install TotalSegmentator")


def test_brain_tumor_dataset():
    """Test BrainTumorDataset"""
    print("\n" + "=" * 80)
    print("Testing BrainTumorDataset")
    print("=" * 80)

    # Create dummy dataset directory structure for testing
    data_dir = Path("./data/test_brain_tumor")

    print(f"\nExpected data directory structure:")
    print(f"{data_dir}/")
    print("  patient_001/")
    print("    mri_t1.nii.gz")
    print("    mri_t2.nii.gz")
    print("    mri_flair.nii.gz")
    print("    ct.nii.gz")
    print("  patient_002/")
    print("    ...")
    print("  clinical_data.csv")

    if not data_dir.exists():
        print(f"\n⚠ Data directory {data_dir} does not exist")
        print("  Skipping dataset test")
        return

    try:
        # Create dataset
        dataset = BrainTumorDataset(
            data_dir=str(data_dir),
            clinical_data_path=str(data_dir / "clinical_data.csv"),
            modalities=['mri_t1', 'mri_t2', 'clinical'],
            target_column='survival_months',
            image_size=(128, 128, 128),
            use_segmentation=False,  # Skip segmentation for quick test
            use_preprocessing=False,  # Skip preprocessing for quick test
            split='train',
            missing_modality_prob=0.0
        )

        print(f"\n✓ Dataset created successfully")
        print(f"  Number of patients: {len(dataset)}")

        if len(dataset) > 0:
            # Test loading a sample
            sample = dataset[0]
            print(f"\n✓ Successfully loaded sample:")
            print(f"  Patient ID: {sample['patient_id']}")
            print(f"  Available modalities: {sample['available_modalities']}")
            print(f"  Target: {sample['target']}")

            for modality, data in sample['modality_data'].items():
                if data is not None:
                    print(f"  {modality} shape: {data.shape}")

    except Exception as e:
        print(f"\n✗ Failed to create/load dataset: {e}")


def test_brats_dataset():
    """Test BraTSDataset"""
    print("\n" + "=" * 80)
    print("Testing BraTSDataset")
    print("=" * 80)

    data_dir = Path("./data/BraTS2020")

    print(f"\nExpected BraTS directory structure:")
    print(f"{data_dir}/")
    print("  BraTS2020_Training_001/")
    print("    BraTS2020_Training_001_t1.nii.gz")
    print("    BraTS2020_Training_001_t1ce.nii.gz")
    print("    BraTS2020_Training_001_t2.nii.gz")
    print("    BraTS2020_Training_001_flair.nii.gz")
    print("    BraTS2020_Training_001_seg.nii.gz")
    print("  survival_info.csv")

    if not data_dir.exists():
        print(f"\n⚠ Data directory {data_dir} does not exist")
        print("  Skipping BraTS dataset test")
        return

    try:
        dataset = BraTSDataset(
            data_dir=str(data_dir),
            clinical_data_path=str(data_dir / "survival_info.csv"),
            modalities=['mri_t1', 'mri_t1ce', 'mri_t2', 'mri_flair'],
            target_column='survival_days',
            image_size=(128, 128, 128),
            use_segmentation=False,
            use_preprocessing=False,
            split='train'
        )

        print(f"\n✓ BraTS dataset created successfully")
        print(f"  Number of patients: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✓ Successfully loaded BraTS sample:")
            print(f"  Patient ID: {sample['patient_id']}")
            print(f"  Available modalities: {sample['available_modalities']}")

    except Exception as e:
        print(f"\n✗ Failed to create/load BraTS dataset: {e}")


def test_dataloaders():
    """Test dataloader creation"""
    print("\n" + "=" * 80)
    print("Testing DataLoader Creation")
    print("=" * 80)

    data_dir = Path("./data/test_brain_tumor")

    if not data_dir.exists():
        print(f"\n⚠ Data directory {data_dir} does not exist")
        print("  Skipping dataloader test")
        return

    try:
        dataloaders = create_dataloaders(
            data_dir=str(data_dir),
            clinical_data_path=str(data_dir / "clinical_data.csv"),
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            modalities=['mri_t1', 'mri_t2'],
            image_size=(64, 64, 64),  # Smaller for testing
            use_segmentation=False,
            use_preprocessing=False
        )

        print(f"\n✓ DataLoaders created successfully")
        print(f"  Train batches: {len(dataloaders['train'])}")
        print(f"  Val batches: {len(dataloaders['val'])}")
        print(f"  Test batches: {len(dataloaders['test'])}")

        # Test loading a batch
        if len(dataloaders['train']) > 0:
            batch = next(iter(dataloaders['train']))
            print(f"\n✓ Successfully loaded batch:")
            print(f"  Batch size: {len(batch['patient_id'])}")

    except Exception as e:
        print(f"\n✗ Failed to create dataloaders: {e}")


def test_transforms():
    """Test data augmentation transforms"""
    print("\n" + "=" * 80)
    print("Testing Data Augmentation Transforms")
    print("=" * 80)

    try:
        # Create dummy multimodal data
        dummy_data = {
            'mri_t1': torch.randn(1, 64, 64, 64),
            'mri_t2': torch.randn(1, 64, 64, 64),
            'clinical': torch.randn(10)
        }

        # Get train transforms
        transform = get_train_transforms(
            use_flip=True,
            use_rotation=True,
            use_noise=True,
            use_gamma=True,
            use_brightness_contrast=True,
            use_elastic=False  # Skip elastic (slow)
        )

        # Apply transforms
        transformed = transform(dummy_data)

        print(f"\n✓ Transforms applied successfully")
        print(f"  Original mri_t1 shape: {dummy_data['mri_t1'].shape}")
        print(f"  Transformed mri_t1 shape: {transformed['mri_t1'].shape}")
        print(f"  Clinical data unchanged: {torch.equal(dummy_data['clinical'], transformed['clinical'])}")

    except Exception as e:
        print(f"\n✗ Failed to apply transforms: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ClinFuseDiff Data Pipeline Test Suite")
    print("=" * 80)

    # Test TotalSegmentator
    test_totalsegmentator()

    # Test datasets
    test_brain_tumor_dataset()
    test_brats_dataset()

    # Test dataloaders
    test_dataloaders()

    # Test transforms
    test_transforms()

    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Download brain tumor dataset (e.g., BraTS)")
    print("2. Organize data according to expected structure")
    print("3. Install TotalSegmentator: pip install TotalSegmentator")
    print("4. Run this script again to test with real data")


if __name__ == "__main__":
    main()
