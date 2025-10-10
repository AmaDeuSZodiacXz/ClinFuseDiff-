# Dataset Setup Guide for ClinFuseDiff

This guide explains how to set up multimodal brain tumor datasets for ClinFuseDiff.

## Supported Datasets

### 1. BraTS (Brain Tumor Segmentation Challenge)

**Download**: [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/)

**Expected Structure**:
```
data/BraTS2020/
├── BraTS2020_Training_001/
│   ├── BraTS2020_Training_001_t1.nii.gz
│   ├── BraTS2020_Training_001_t1ce.nii.gz
│   ├── BraTS2020_Training_001_t2.nii.gz
│   ├── BraTS2020_Training_001_flair.nii.gz
│   └── BraTS2020_Training_001_seg.nii.gz (ground truth)
├── BraTS2020_Training_002/
│   └── ...
└── survival_info.csv
```

**Clinical Data CSV Format**:
```csv
patient_id,age,gender,survival_days,extent_of_resection
BraTS2020_Training_001,65,M,365,GTR
BraTS2020_Training_002,52,F,180,STR
```

**Usage**:
```python
from src.data import BraTSDataset

dataset = BraTSDataset(
    data_dir='data/BraTS2020',
    clinical_data_path='data/BraTS2020/survival_info.csv',
    modalities=['mri_t1', 'mri_t1ce', 'mri_t2', 'mri_flair', 'clinical'],
    target_column='survival_days'
)
```

### 2. Custom Brain Tumor Dataset

**Expected Structure**:
```
data/custom_brain_tumor/
├── patient_001/
│   ├── mri_t1.nii.gz
│   ├── mri_t2.nii.gz
│   ├── mri_flair.nii.gz
│   ├── ct.nii.gz (optional)
│   └── pet.nii.gz (optional)
├── patient_002/
│   └── ...
└── clinical_data.csv
```

**Clinical Data CSV Format**:
```csv
patient_id,age,gender,tumor_grade,survival_months,kps_score
patient_001,58,M,IV,12,70
patient_002,45,F,III,24,80
```

**Usage**:
```python
from src.data import BrainTumorDataset

dataset = BrainTumorDataset(
    data_dir='data/custom_brain_tumor',
    clinical_data_path='data/custom_brain_tumor/clinical_data.csv',
    modalities=['mri_t1', 'mri_t2', 'ct', 'clinical'],
    target_column='survival_months'
)
```

## Installing TotalSegmentator

TotalSegmentator is used for automatic brain structure segmentation.

```bash
pip install TotalSegmentator
```

**Test installation**:
```bash
TotalSegmentator --help
```

**Segment a brain scan**:
```bash
# For CT
TotalSegmentator -i brain_ct.nii.gz -o output_dir/ --task brain_structures

# For MRI
TotalSegmentator -i brain_mri.nii.gz -o output_dir/ --task total_mr
```

**Python API**:
```python
from src.data import TotalSegmentatorWrapper

segmentator = TotalSegmentatorWrapper(
    task='brain_structures',
    device='gpu',
    fast=False
)

seg_path = segmentator.segment(
    'brain_scan.nii.gz',
    'output_segmentation.nii.gz'
)
```

## Data Preprocessing

ClinFuseDiff includes automatic preprocessing:

1. **Resampling**: Resample to consistent voxel spacing (1mm³)
2. **Resizing**: Resize to target dimensions (128×128×128)
3. **Normalization**: Percentile-based intensity normalization
4. **Segmentation**: Optional automatic brain structure segmentation
5. **Skull stripping**: Optional using TotalSegmentator masks

**Example with preprocessing**:
```python
dataset = BrainTumorDataset(
    data_dir='data/brain_tumor',
    use_preprocessing=True,
    use_segmentation=True,
    cache_dir='data/cache',  # Cache preprocessed data
    image_size=(128, 128, 128)
)
```

## Data Augmentation

Training augmentations are applied automatically:

```python
from src.data import get_train_transforms, get_val_transforms

train_transform = get_train_transforms(
    use_flip=True,
    use_rotation=True,
    use_noise=True,
    use_gamma=True,
    use_brightness_contrast=True,
    use_elastic=False  # Elastic deformation is slow
)

dataset = BrainTumorDataset(
    data_dir='data/brain_tumor',
    transform=train_transform,
    split='train'
)
```

**Available augmentations**:
- Random flipping (axes: H, W)
- Random rotation (±15°)
- Gaussian noise
- Gamma correction
- Brightness/contrast adjustment
- Elastic deformation (optional)

## Creating DataLoaders

Use the convenience function to create train/val/test dataloaders:

```python
from src.data import create_dataloaders

dataloaders = create_dataloaders(
    data_dir='data/brain_tumor',
    clinical_data_path='data/brain_tumor/clinical_data.csv',
    batch_size=4,
    num_workers=4,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    modalities=['mri_t1', 'mri_t2', 'ct', 'clinical'],
    image_size=(128, 128, 128),
    use_segmentation=True,
    missing_modality_prob=0.3  # Simulate 30% missing modalities
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

## Handling Missing Modalities

ClinFuseDiff is designed to handle missing modalities:

**Simulate during training**:
```python
dataset = BrainTumorDataset(
    data_dir='data/brain_tumor',
    split='train',
    missing_modality_prob=0.3  # 30% chance each modality is missing
)
```

**Natural missing modalities**:
- Dataset automatically detects which modalities exist for each patient
- Missing modalities are indicated in `available_modalities` field
- Model adapts to available modalities at test time

## Testing the Data Pipeline

Run the test script to verify your setup:

```bash
python examples/test_data_pipeline.py
```

This will test:
1. TotalSegmentator installation
2. Dataset loading
3. DataLoader creation
4. Augmentation transforms

## Recommended Datasets for Research

1. **BraTS (Brain Tumor Segmentation)**
   - Multimodal MRI (T1, T1ce, T2, FLAIR)
   - Large dataset (~300-500 patients)
   - Survival prediction task
   - Download: https://www.med.upenn.edu/cbica/brats/

2. **TCIA Brain Collections**
   - Various brain tumor datasets
   - Some include CT and PET
   - Download: https://www.cancerimagingarchive.net/

3. **UK Biobank Brain Imaging**
   - Large-scale population study
   - MRI data
   - Clinical metadata
   - Application required: https://www.ukbiobank.ac.uk/

## Data Format Requirements

### Image Files
- Format: NIfTI (.nii or .nii.gz)
- 3D volumes (D×H×W)
- Any voxel spacing (will be resampled)
- Any dimensions (will be resized)

### Clinical Data
- Format: CSV
- Required column: `patient_id` (matching folder names)
- Target column: specified by `target_column` parameter
- Feature columns: automatically detected (numeric or categorical)
- Missing values: automatically filled with 0

### Example Clinical Features

**Continuous**:
- Age
- Survival time
- Karnofsky Performance Score (KPS)
- Tumor volume

**Categorical** (one-hot encode before training):
- Gender (M/F)
- Tumor grade (I/II/III/IV)
- Extent of resection (GTR/STR/Biopsy)
- Molecular markers (IDH, MGMT, etc.)

## Troubleshooting

### TotalSegmentator not found
```bash
pip install TotalSegmentator
# Or with specific version
pip install TotalSegmentator==2.0.0
```

### GPU out of memory during segmentation
```python
# Use CPU
segmentator = TotalSegmentatorWrapper(device='cpu')

# Or use fast mode
segmentator = TotalSegmentatorWrapper(fast=True)
```

### Slow data loading
```python
# Enable caching
dataset = BrainTumorDataset(
    cache_dir='data/cache',  # Saves preprocessed data
    use_preprocessing=True
)

# Increase workers
dataloaders = create_dataloaders(num_workers=8)
```

### Dataset not found
- Check directory structure matches expected format
- Verify patient folder names (should start with "patient" or match BraTS format)
- Check file extensions (.nii.gz)
- Verify clinical_data.csv has correct column names

## Next Steps

After setting up your data:

1. Test data loading: `python examples/test_data_pipeline.py`
2. Train baseline model: `python src/training/train.py`
3. Implement test-time training: See `docs/TTT_TRAINING.md`
4. Evaluate on missing modalities: See `docs/EVALUATION.md`
