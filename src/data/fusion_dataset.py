"""
Image Fusion Dataset for CVPR 2026 CLIN-FuseDiff++
Loads paired CT-MRI images + ROI masks for image-level fusion
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFusionDataset(Dataset):
    """
    Dataset for medical image fusion with ROI masks

    Returns paired images (CT + MRI/ADC) with corresponding ROI masks
    (brain, bone, lesion) for ROI-aware guided diffusion fusion.

    Expected directory structure:
        data_dir/
            case_001/
                ct.nii.gz              # CT image
                mri.nii.gz or adc.nii.gz  # MRI or ADC image
                brain_mask.nii.gz      # Brain ROI (from TotalSegmentator on MRI)
                bone_mask.nii.gz       # Bone/skull ROI (from TotalSegmentator on CT)
                lesion_mask.nii.gz     # Lesion mask (expert annotation)
            case_002/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        cases: Optional[List[str]] = None,
        ct_filename: str = "ct.nii.gz",
        mri_filename: str = "mri.nii.gz",  # or "adc.nii.gz"
        brain_mask_filename: str = "brain_mask.nii.gz",
        bone_mask_filename: str = "bone_mask.nii.gz",
        lesion_mask_filename: str = "lesion_mask.nii.gz",
        transform=None,
        load_lesion_mask: bool = True,
        normalize: bool = True,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing case subdirectories
            cases: List of case IDs to load (if None, auto-discover)
            ct_filename: Name of CT file in each case directory
            mri_filename: Name of MRI/ADC file in each case directory
            brain_mask_filename: Name of brain mask file
            bone_mask_filename: Name of bone mask file
            lesion_mask_filename: Name of lesion mask file
            transform: Transform to apply (should handle multi-modal consistently)
            load_lesion_mask: Whether to load lesion masks
            normalize: Apply intensity normalization
            cache_in_memory: Cache loaded data in RAM (faster but uses more memory)
        """
        self.data_dir = Path(data_dir)
        self.ct_filename = ct_filename
        self.mri_filename = mri_filename
        self.brain_mask_filename = brain_mask_filename
        self.bone_mask_filename = bone_mask_filename
        self.lesion_mask_filename = lesion_mask_filename
        self.transform = transform
        self.load_lesion_mask = load_lesion_mask
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        # Discover cases
        if cases is None:
            self.cases = self._discover_cases()
        else:
            self.cases = cases

        print(f"ImageFusionDataset: Found {len(self.cases)} cases in {data_dir}")

        # Validate cases
        self._validate_cases()

        # Cache
        self.cache = {} if cache_in_memory else None

    def _discover_cases(self) -> List[str]:
        """Auto-discover case directories"""
        cases = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if required files exist
                ct_path = item / self.ct_filename
                mri_path = item / self.mri_filename

                if ct_path.exists() and mri_path.exists():
                    cases.append(item.name)

        return sorted(cases)

    def _validate_cases(self):
        """Validate that required files exist for each case"""
        valid_cases = []
        missing_files = []

        for case_id in self.cases:
            case_dir = self.data_dir / case_id

            # Check required files
            ct_path = case_dir / self.ct_filename
            mri_path = case_dir / self.mri_filename

            if not ct_path.exists():
                missing_files.append(f"{case_id}: missing {self.ct_filename}")
                continue

            if not mri_path.exists():
                missing_files.append(f"{case_id}: missing {self.mri_filename}")
                continue

            valid_cases.append(case_id)

        if missing_files:
            print(f"WARNING: Skipping {len(missing_files)} cases with missing files:")
            for msg in missing_files[:5]:  # Show first 5
                print(f"  - {msg}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")

        self.cases = valid_cases
        print(f"Validated: {len(self.cases)} cases with complete data")

    def _normalize_ct(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        Normalize CT using clinical windowing (proposal requirement)

        Brain + Bone Window (clinical standard for head CT):
        - Soft tissue: 0-80 HU (brain parenchyma)
        - Bone: 200-1000 HU (skull, preserved for ROI guidance)

        Window: Center=40 HU, Width=400 HU → Range: [-160, 240] HU
        Maps to [-2, 2] range to match MRI Z-score scale
        """
        # Clinical brain+bone window
        window_center = 40
        window_width = 400

        window_min = window_center - window_width / 2  # -160 HU
        window_max = window_center + window_width / 2  # 240 HU

        # Clip to window
        ct_windowed = np.clip(ct_volume, window_min, window_max)

        # Normalize to [-2, 2] range (similar to MRI Z-score)
        # Formula: 4 * (x - min) / width - 2
        ct_normalized = 4.0 * (ct_windowed - window_min) / window_width - 2.0

        return ct_normalized.astype(np.float32)

    def _normalize_mri(self, mri_volume: np.ndarray) -> np.ndarray:
        """Normalize MRI/ADC intensity (z-score normalization)"""
        # Compute mean and std from non-zero voxels (exclude background)
        mask = mri_volume > 0
        if mask.sum() > 0:
            mean = mri_volume[mask].mean()
            std = mri_volume[mask].std()
            if std > 0:
                mri_volume = (mri_volume - mean) / std
        return mri_volume.astype(np.float32)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file and return data array"""
        img = nib.load(str(path))
        data = img.get_fdata()
        return data

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a case

        Returns:
            dict with keys:
                - 'ct': CT volume (1, D, H, W)
                - 'mri': MRI/ADC volume (1, D, H, W)
                - 'brain_mask': Brain ROI mask (1, D, H, W)
                - 'bone_mask': Bone/skull ROI mask (1, D, H, W)
                - 'lesion_mask': Lesion mask (1, D, H, W) [if load_lesion_mask=True]
                - 'case_id': str
                - 'available_masks': list of available mask types
        """
        case_id = self.cases[idx]

        # Check cache
        if self.cache_in_memory and case_id in self.cache:
            return self.cache[case_id]

        case_dir = self.data_dir / case_id

        # Load images
        ct_path = case_dir / self.ct_filename
        mri_path = case_dir / self.mri_filename

        ct_volume = self._load_nifti(ct_path)
        mri_volume = self._load_nifti(mri_path)

        # Normalize
        if self.normalize:
            ct_volume = self._normalize_ct(ct_volume)
            mri_volume = self._normalize_mri(mri_volume)

        # Load masks
        brain_mask = None
        bone_mask = None
        lesion_mask = None
        available_masks = []

        brain_mask_path = case_dir / self.brain_mask_filename
        if brain_mask_path.exists():
            brain_mask = self._load_nifti(brain_mask_path)
            brain_mask = (brain_mask > 0).astype(np.float32)
            available_masks.append('brain')

        bone_mask_path = case_dir / self.bone_mask_filename
        if bone_mask_path.exists():
            bone_mask = self._load_nifti(bone_mask_path)
            bone_mask = (bone_mask > 0).astype(np.float32)
            available_masks.append('bone')

        if self.load_lesion_mask:
            lesion_mask_path = case_dir / self.lesion_mask_filename
            if lesion_mask_path.exists():
                lesion_mask = self._load_nifti(lesion_mask_path)
                lesion_mask = (lesion_mask > 0).astype(np.float32)
                available_masks.append('lesion')

        # Add channel dimension and convert to torch
        sample = {
            'ct': torch.from_numpy(ct_volume[None, ...]),  # (1, D, H, W)
            'mri': torch.from_numpy(mri_volume[None, ...]),
            'case_id': case_id,
            'available_masks': available_masks
        }

        if brain_mask is not None:
            sample['brain_mask'] = torch.from_numpy(brain_mask[None, ...])

        if bone_mask is not None:
            sample['bone_mask'] = torch.from_numpy(bone_mask[None, ...])

        if lesion_mask is not None:
            sample['lesion_mask'] = torch.from_numpy(lesion_mask[None, ...])

        # Apply transforms (if any)
        if self.transform:
            sample = self.transform(sample)

        # Cache
        if self.cache_in_memory:
            self.cache[case_id] = sample

        return sample


def create_splits(
    data_dir: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/val/test splits

    Args:
        data_dir: Root data directory
        split_ratios: (train, val, test) ratios (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'test' containing case IDs
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    # Discover all cases
    dataset = ImageFusionDataset(data_dir, normalize=False)
    all_cases = dataset.cases

    # Shuffle with seed
    rng = np.random.RandomState(seed)
    shuffled_cases = rng.permutation(all_cases)

    # Split
    n_total = len(shuffled_cases)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    splits = {
        'train': shuffled_cases[:n_train].tolist(),
        'val': shuffled_cases[n_train:n_train+n_val].tolist(),
        'test': shuffled_cases[n_train+n_val:].tolist()
    }

    print(f"Created splits:")
    print(f"  Train: {len(splits['train'])} cases")
    print(f"  Val:   {len(splits['val'])} cases")
    print(f"  Test:  {len(splits['test'])} cases")

    return splits


def save_splits(splits: Dict[str, List[str]], output_dir: str):
    """Save splits to text files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, cases in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            f.write('\n'.join(cases))
        print(f"Saved {split_name} split: {split_file}")


def load_splits(splits_dir: str) -> Dict[str, List[str]]:
    """Load splits from text files"""
    splits_dir = Path(splits_dir)
    splits = {}

    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split_name}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                cases = [line.strip() for line in f if line.strip()]
            splits[split_name] = cases
            print(f"Loaded {split_name} split: {len(cases)} cases")

    return splits
# Alias for backward compatibility
FusionDataset = ImageFusionDataset

