"""Dataset loaders for multimodal clinical data"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib

from .segmentation import BrainSegmentationPreprocessor, TotalSegmentatorWrapper


class BrainTumorDataset(Dataset):
    """
    Multimodal brain tumor dataset supporting MRI, CT, and clinical data

    Expected directory structure:
    data_dir/
        patient_001/
            mri_t1.nii.gz
            mri_t2.nii.gz
            mri_flair.nii.gz
            ct.nii.gz
            segmentation.nii.gz (optional, ground truth)
        patient_002/
            ...
        clinical_data.csv (with patient_id, age, gender, tumor_grade, etc.)
    """

    def __init__(
        self,
        data_dir: str,
        clinical_data_path: Optional[str] = None,
        modalities: List[str] = ['mri_t1', 'mri_t2', 'ct', 'clinical'],
        target_column: str = 'survival_months',
        image_size: Tuple[int, int, int] = (128, 128, 128),
        use_segmentation: bool = True,
        use_preprocessing: bool = True,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        split: str = 'train',
        missing_modality_prob: float = 0.0,
        seed: int = 42
    ):
        """
        Args:
            data_dir: root directory containing patient folders
            clinical_data_path: path to clinical data CSV file
            modalities: list of modalities to load
            target_column: name of target variable in clinical data
            image_size: target size for 3D images
            use_segmentation: whether to perform segmentation
            use_preprocessing: whether to apply preprocessing
            cache_dir: directory to cache preprocessed data
            transform: additional transforms to apply
            split: 'train', 'val', or 'test'
            missing_modality_prob: probability of simulating missing modalities
            seed: random seed
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.target_column = target_column
        self.image_size = image_size
        self.use_segmentation = use_segmentation
        self.transform = transform
        self.split = split
        self.missing_modality_prob = missing_modality_prob

        np.random.seed(seed)

        # Setup cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / split
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Initialize preprocessor
        if use_preprocessing:
            segmentator = TotalSegmentatorWrapper(
                task='brain_structures',
                fast=False,
                ml=True
            )
            self.preprocessor = BrainSegmentationPreprocessor(
                segmentator=segmentator,
                target_size=image_size,
                normalize=True,
                use_skull_stripping=False
            )
        else:
            self.preprocessor = None

        # Load clinical data
        self.clinical_data = None
        self.clinical_features = None
        if clinical_data_path is not None and os.path.exists(clinical_data_path):
            self.clinical_data = pd.read_csv(clinical_data_path)
            # Extract feature columns (exclude patient_id and target)
            exclude_cols = ['patient_id', 'subject_id', self.target_column]
            self.clinical_features = [
                col for col in self.clinical_data.columns
                if col not in exclude_cols
            ]

        # Get patient IDs
        self.patient_ids = [
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith('patient')
        ]
        self.patient_ids.sort()

        # Map modality names to file names
        self.modality_file_map = {
            'mri_t1': 'mri_t1.nii.gz',
            'mri_t1ce': 'mri_t1ce.nii.gz',
            'mri_t2': 'mri_t2.nii.gz',
            'mri_flair': 'mri_flair.nii.gz',
            'ct': 'ct.nii.gz',
            'pet': 'pet.nii.gz'
        }

        print(f"Loaded {len(self.patient_ids)} patients for {split} split")
        print(f"Modalities: {modalities}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id

        # Determine available modalities
        available_modalities = self._get_available_modalities(patient_dir)

        # Simulate missing modalities during training
        if self.split == 'train' and self.missing_modality_prob > 0:
            available_modalities = self._simulate_missing_modalities(
                available_modalities, self.missing_modality_prob
            )

        # Load modality data
        modality_data = {}

        # Load imaging modalities
        for modality in self.modalities:
            if modality == 'clinical':
                continue

            if modality not in available_modalities:
                # Missing modality: use zeros or skip
                modality_data[modality] = None
            else:
                # Load image
                image_path = patient_dir / self.modality_file_map[modality]

                # Check cache
                if self.cache_dir is not None:
                    cache_file = self.cache_dir / f"{patient_id}_{modality}.npy"
                    if cache_file.exists():
                        image_data = np.load(cache_file)
                    else:
                        image_data = self._load_and_preprocess_image(image_path)
                        np.save(cache_file, image_data)
                else:
                    image_data = self._load_and_preprocess_image(image_path)

                # Convert to tensor
                modality_data[modality] = torch.from_numpy(image_data).unsqueeze(0).float()

        # Load clinical data
        if 'clinical' in self.modalities and self.clinical_data is not None:
            clinical_features = self._get_clinical_features(patient_id)
            modality_data['clinical'] = torch.from_numpy(clinical_features).float()

        # Load target
        target = self._get_target(patient_id)

        # Apply transforms
        if self.transform is not None:
            modality_data = self.transform(modality_data)

        return {
            'patient_id': patient_id,
            'modality_data': modality_data,
            'available_modalities': available_modalities,
            'target': target
        }

    def _get_available_modalities(self, patient_dir: Path) -> List[str]:
        """Check which modalities are available for a patient"""
        available = []

        for modality in self.modalities:
            if modality == 'clinical':
                if self.clinical_data is not None:
                    available.append(modality)
            else:
                file_path = patient_dir / self.modality_file_map.get(modality, f"{modality}.nii.gz")
                if file_path.exists():
                    available.append(modality)

        return available

    def _simulate_missing_modalities(
        self,
        available_modalities: List[str],
        missing_prob: float
    ) -> List[str]:
        """Randomly remove modalities to simulate missing data"""
        if len(available_modalities) <= 1:
            return available_modalities

        # Keep at least one modality
        keep_modalities = []
        for modality in available_modalities:
            if np.random.rand() > missing_prob:
                keep_modalities.append(modality)

        # Ensure at least one modality remains
        if len(keep_modalities) == 0:
            keep_modalities = [np.random.choice(available_modalities)]

        return keep_modalities

    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a medical image"""
        if self.preprocessor is not None:
            result = self.preprocessor.preprocess(
                image_path,
                return_segmentation=self.use_segmentation,
                cache_dir=self.cache_dir
            )
            return result['image']
        else:
            # Simple loading without preprocessing
            nii = nib.load(str(image_path))
            image_data = nii.get_fdata()

            # Simple resize
            from scipy.ndimage import zoom
            zoom_factors = [
                self.image_size[i] / image_data.shape[i]
                for i in range(3)
            ]
            image_data = zoom(image_data, zoom_factors, order=1)

            # Simple normalization
            if image_data.max() > 0:
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

            return image_data.astype(np.float32)

    def _get_clinical_features(self, patient_id: str) -> np.ndarray:
        """Extract clinical features for a patient"""
        if self.clinical_data is None:
            return np.array([])

        # Find patient row
        patient_row = self.clinical_data[
            self.clinical_data['patient_id'] == patient_id
        ]

        if len(patient_row) == 0:
            # Return zeros if patient not found
            return np.zeros(len(self.clinical_features))

        # Extract feature values
        features = patient_row[self.clinical_features].values[0]

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        return features.astype(np.float32)

    def _get_target(self, patient_id: str) -> torch.Tensor:
        """Get target value for a patient"""
        if self.clinical_data is None:
            return torch.tensor(0.0)

        patient_row = self.clinical_data[
            self.clinical_data['patient_id'] == patient_id
        ]

        if len(patient_row) == 0:
            return torch.tensor(0.0)

        target_value = patient_row[self.target_column].values[0]

        # Handle missing target
        if pd.isna(target_value):
            target_value = 0.0

        return torch.tensor(target_value, dtype=torch.float32)


class BraTSDataset(BrainTumorDataset):
    """
    Specialized dataset for BraTS (Brain Tumor Segmentation) challenge data

    BraTS structure:
    data_dir/
        BraTS2020_Training_001/
            BraTS2020_Training_001_t1.nii.gz
            BraTS2020_Training_001_t1ce.nii.gz
            BraTS2020_Training_001_t2.nii.gz
            BraTS2020_Training_001_flair.nii.gz
            BraTS2020_Training_001_seg.nii.gz
        survival_info.csv
    """

    def __init__(self, **kwargs):
        # Update modality file map for BraTS naming convention
        super().__init__(**kwargs)

    def _get_available_modalities(self, patient_dir: Path) -> List[str]:
        """BraTS-specific modality detection"""
        available = []
        patient_name = patient_dir.name

        modality_suffixes = {
            'mri_t1': '_t1.nii.gz',
            'mri_t1ce': '_t1ce.nii.gz',
            'mri_t2': '_t2.nii.gz',
            'mri_flair': '_flair.nii.gz'
        }

        for modality in self.modalities:
            if modality == 'clinical':
                if self.clinical_data is not None:
                    available.append(modality)
            elif modality in modality_suffixes:
                file_path = patient_dir / f"{patient_name}{modality_suffixes[modality]}"
                if file_path.exists():
                    available.append(modality)

        return available

    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """BraTS images are already skull-stripped and registered"""
        # For BraTS, we can skip skull stripping
        if self.preprocessor is not None:
            self.preprocessor.use_skull_stripping = False

        return super()._load_and_preprocess_image(image_path)


def create_dataloaders(
    data_dir: str,
    clinical_data_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    missing_modality_prob: float = 0.0,
    seed: int = 42,
    **dataset_kwargs
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders

    Args:
        data_dir: root data directory
        clinical_data_path: path to clinical data CSV
        batch_size: batch size
        num_workers: number of workers for data loading
        train_split: fraction for training
        val_split: fraction for validation
        test_split: fraction for testing
        missing_modality_prob: probability of missing modalities in training
        seed: random seed
        **dataset_kwargs: additional arguments for dataset

    Returns:
        dict of dataloaders for train/val/test
    """
    # Get all patient IDs
    data_path = Path(data_dir)
    all_patients = [d.name for d in data_path.iterdir() if d.is_dir()]
    all_patients.sort()

    # Split data
    np.random.seed(seed)
    np.random.shuffle(all_patients)

    n_train = int(len(all_patients) * train_split)
    n_val = int(len(all_patients) * val_split)

    train_patients = all_patients[:n_train]
    val_patients = all_patients[n_train:n_train + n_val]
    test_patients = all_patients[n_train + n_val:]

    # Create temporary split directories (or use indices)
    # For simplicity, we'll pass all data and filter in dataset

    # Create datasets
    train_dataset = BrainTumorDataset(
        data_dir=data_dir,
        clinical_data_path=clinical_data_path,
        split='train',
        missing_modality_prob=missing_modality_prob,
        seed=seed,
        **dataset_kwargs
    )

    val_dataset = BrainTumorDataset(
        data_dir=data_dir,
        clinical_data_path=clinical_data_path,
        split='val',
        missing_modality_prob=0.0,  # No missing modalities in validation
        seed=seed,
        **dataset_kwargs
    )

    test_dataset = BrainTumorDataset(
        data_dir=data_dir,
        clinical_data_path=clinical_data_path,
        split='test',
        missing_modality_prob=0.0,  # No missing modalities in test
        seed=seed,
        **dataset_kwargs
    )

    # Filter patients by split
    train_dataset.patient_ids = [p for p in train_dataset.patient_ids if p in train_patients]
    val_dataset.patient_ids = [p for p in val_dataset.patient_ids if p in val_patients]
    test_dataset.patient_ids = [p for p in test_dataset.patient_ids if p in test_patients]

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
