# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClinFuseDiff is a research project implementing **Diffusion-Based Test-Time Training for Robust Multimodal Clinical Data Fusion**. The project combines conditional diffusion models with test-time training (TTT) to enable robust fusion of multimodal clinical data (MRI, CT, PET, clinical features) that adapts to missing modalities at test time.

**Key Innovation**: Unlike traditional fusion methods that fail when trained modality combinations don't match test combinations, ClinFuseDiff uses diffusion models to generate robust fused representations and adapts at test-time through self-supervised learning.

**Target**: ICLR 2026 submission

## Architecture Overview

### Three-Stage Pipeline

1. **Encoding Stage** (`src/models/encoders.py`)
   - `ImageEncoder`: ResNet50-based encoder for 3D medical images (CT/MRI/PET)
   - `ClinicalEncoder`: MLP-based encoder for tabular clinical data
   - `MultiModalEncoder`: Wrapper managing multiple modality-specific encoders
   - Each modality is encoded independently to fixed-dimension feature vectors

2. **Fusion Stage** (`src/models/fusion.py`)
   - **Two fusion approaches**:
     - `AttentionFusion`: Baseline using multi-head attention across modalities
     - `DiffusionFusion`: Main contribution using conditional diffusion to generate fused representations
   - Fusion is conditioned on available modalities (handles missing data)
   - `DiffusionFusion` learns to denoise from random noise to fused representation, conditioned on available modality features

3. **Prediction Stage** (`src/models/fusion.py`)
   - MLP prediction head for downstream task (survival, diagnosis, etc.)
   - Takes fused representation as input

### Diffusion Model Architecture (`src/models/diffusion.py`)

- `GaussianDiffusion`: Implements forward diffusion (adding noise) and reverse diffusion (denoising)
- `DiffusionTransformer`: Transformer-based denoising network
  - Input: noisy fused feature + timestep + conditioning (from available modalities)
  - Output: predicted noise
- Supports configurable noise schedules: linear, cosine
- Training: Learn to predict noise at random timesteps
- Inference: Sample by iteratively denoising from pure noise

### Test-Time Training (TTT) - TO BE IMPLEMENTED

The TTT module will perform per-sample adaptation at test time using three self-supervised objectives:
1. **Reconstruction loss**: Reconstruct available modality features from fused representation
2. **Contrastive loss**: Align features across available modalities
3. **Diffusion denoising loss**: Improve diffusion model on test sample

This allows the model to adapt to novel modality combinations and individual patient characteristics.

## Data Pipeline Architecture

### Dataset Structure (`src/data/datasets.py`)

Two main dataset classes:
- `BrainTumorDataset`: Generic loader for custom brain tumor datasets
- `BraTSDataset`: Specialized loader for BraTS challenge format

**Key features**:
- Automatic modality detection (checks which files exist per patient)
- Missing modality simulation during training (`missing_modality_prob`)
- Returns dict with `modality_data`, `available_modalities`, `target`
- Supports caching of preprocessed data for efficiency

**Expected directory structure**:
```
data/
  patient_001/
    mri_t1.nii.gz, mri_t2.nii.gz, ct.nii.gz, etc.
  patient_002/
    ...
  clinical_data.csv
```

### Preprocessing Pipeline (`src/data/segmentation.py`)

`BrainSegmentationPreprocessor` provides:
1. **TotalSegmentator integration**: Automatic brain structure segmentation
2. **Resampling**: To consistent voxel spacing (default 1mm³)
3. **Resizing**: To target dimensions (default 128×128×128)
4. **Normalization**: Percentile-based intensity normalization
5. **Optional skull stripping**: Using segmentation masks
6. **Caching**: Saves preprocessed data to avoid recomputation

### Data Augmentation (`src/data/transforms.py`)

`MultiModalCompose` ensures **consistent transforms across modalities** (important for maintaining spatial correspondence):
- All imaging modalities receive same random transform (same flip, rotation, etc.)
- Clinical data is not transformed
- Uses fixed random seed per sample to ensure consistency

## Configuration System

All hyperparameters are in `configs/default_config.yaml`:
- Model architecture params (encoder dims, diffusion timesteps, fusion dims)
- Training params (batch size, learning rate, loss weights)
- TTT params (num steps, learning rate, objectives)
- Data params (splits, missing modality probabilities)

When implementing training scripts, use this config system rather than hardcoding values.

## Development Commands

### Testing Data Pipeline
```bash
# Test dataset loading and TotalSegmentator integration
python examples/test_data_pipeline.py
```

### Environment Setup
```bash
# Create environment
conda create -n clinfusediff python=3.9
conda activate clinfusediff

# Install dependencies
pip install -r requirements.txt

# Install TotalSegmentator (for brain segmentation)
pip install TotalSegmentator
```

### Running TotalSegmentator (standalone)
```bash
# Segment brain CT
TotalSegmentator -i brain_ct.nii.gz -o output/ --task brain_structures

# Segment brain MRI
TotalSegmentator -i brain_mri.nii.gz -o output/ --task total_mr

# Fast mode (less accurate but faster)
TotalSegmentator -i image.nii.gz -o output/ --fast
```

## Key Implementation Patterns

### 1. Handling Missing Modalities

When loading data:
```python
# Dataset automatically detects available modalities
sample = dataset[idx]
available_mods = sample['available_modalities']  # e.g., ['mri_t1', 'clinical']
modality_data = sample['modality_data']  # dict with available modalities only
```

When passing to model:
```python
# Model adapts fusion based on available modalities
outputs = model(modality_inputs, available_modalities=['mri_t1', 'ct'])
```

### 2. Diffusion Training vs. Sampling

**Training mode** (`mode='train'`):
- Computes diffusion loss (predicting noise)
- Uses average of projected features as "ground truth" fused representation

**Sampling mode** (`mode='sample'`):
- Generates fused representation through reverse diffusion
- Starts from random noise, iteratively denoises

### 3. Encoder Configuration

When creating encoders, use modality configs dict:
```python
modality_configs = {
    'mri_t1': {'type': 'image', 'in_channels': 1, 'feat_dim': 2048},
    'ct': {'type': 'image', 'in_channels': 1, 'feat_dim': 2048},
    'clinical': {'type': 'clinical', 'input_dim': 50, 'feat_dim': 1024}
}
encoder = MultiModalEncoder(modality_configs)
```

### 4. Data Caching Strategy

Preprocessing 3D medical images is expensive. Always use caching:
```python
dataset = BrainTumorDataset(
    cache_dir='data/cache',  # Saves preprocessed .npy files
    use_preprocessing=True
)
```

## Current Implementation Status

**✅ Complete**:
- Model architecture (encoders, diffusion, fusion)
- Data pipeline (datasets, preprocessing, augmentation)
- TotalSegmentator integration
- Configuration system

**🚧 TODO** (next priorities):
1. Training framework (`src/training/trainer.py`)
   - Loss computation (reconstruction, contrastive, diffusion, prediction)
   - Training loop with validation
   - Checkpoint management

2. Test-Time Training (`src/training/ttt.py`)
   - Per-sample adaptation
   - Self-supervised objectives
   - Efficient gradient updates

3. Training script (`train.py`)
   - Argument parsing
   - Config loading
   - Logging setup

See `IMPLEMENTATION_STATUS.md` for detailed task breakdown.

## Important Notes for Development

### Medical Image Conventions
- **Format**: NIfTI (.nii or .nii.gz) - standard for medical imaging
- **Orientation**: Images may have different orientations (RAS, LPS, etc.)
- **Spacing**: Voxel spacing varies across scans (needs resampling)
- **Intensity**: No standard range (CT: Hounsfield units, MRI: arbitrary)

### Modality Naming Conventions
- `mri_t1`, `mri_t1ce`, `mri_t2`, `mri_flair`: MRI sequences
- `ct`: Computed Tomography
- `pet`: Positron Emission Tomography
- `clinical`: Tabular clinical features

### Loss Function Design (for implementation)
Total loss should be weighted combination:
```
L_total = λ_pred * L_pred + λ_diff * L_diff + λ_recon * L_recon + λ_contra * L_contra
```
Where:
- `L_pred`: Prediction loss (cross-entropy or MSE for survival)
- `L_diff`: Diffusion denoising loss
- `L_recon`: Reconstruction loss (encoder outputs → decoder → reconstruct inputs)
- `L_contra`: Contrastive loss (align features across modalities)

Loss weights are in `configs/default_config.yaml`.

### GPU Memory Management
3D medical images are memory-intensive:
- Use smaller batch sizes (4-8 typical)
- Use gradient accumulation if needed
- Enable mixed precision training (AMP)
- Use gradient checkpointing for diffusion model

### Reproducibility
- All random seeds should respect `config.experiment.seed`
- Set seeds for: numpy, torch, random, dataloader workers
- Deterministic mode for PyTorch operations

## Dataset Resources

**Recommended datasets**:
- BraTS (Brain Tumor Segmentation): https://www.med.upenn.edu/cbica/brats/
- TCIA Brain Collections: https://www.cancerimagingarchive.net/

**Setup guide**: See `docs/DATASET_SETUP.md` for detailed instructions on:
- Directory structure requirements
- Clinical data CSV format
- TotalSegmentator usage
- Data preprocessing pipeline

## Research Context

This work builds on **TTTFusion** (see `TTTFusion.pdf`), extending it with:
1. Diffusion-based fusion (vs. deterministic fusion)
2. Feature-level modality imputation capability
3. Enhanced test-time adaptation with diffusion objective

The goal is to show that diffusion improves robustness to missing modalities compared to deterministic fusion methods.

## Git Workflow Notes

- `.gitignore` excludes `/data/` and `/datasets/` (actual data files)
- `.gitignore` excludes model checkpoints (`.pth`, `.pt`, `.ckpt`)
- `.gitignore` excludes medical images (`.nii`, `.nii.gz`)
- Source code in `src/` is tracked
- Use meaningful commit messages with context

## When Implementing Training

Key files to create:
1. `src/training/trainer.py`: Main training logic
2. `src/training/ttt.py`: Test-time adaptation
3. `src/training/evaluator.py`: Evaluation metrics
4. `src/utils/metrics.py`: Metric calculations
5. `train.py`: Entry point script

Follow patterns from existing model code (e.g., using configs, returning dicts with named outputs).
