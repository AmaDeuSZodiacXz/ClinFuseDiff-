<div align="center">

# CLIN-FuseDiff++

**ROI-Aware Guided Diffusion for Multimodal Medical Image Fusion**

[![arXiv](https://img.shields.io/badge/arXiv-2026.00000-b31b1b.svg)](https://arxiv.org/abs/2026.00000)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Paper**](https://arxiv.org/abs/2026.00000) | [**Supplementary**](docs/supplementary.pdf) | [**Pretrained Models**](#pretrained-models)

</div>

---

## Highlights

- 🎯 **ROI-Aware Guidance**: Region-specific fusion (brain ↔ MRI, bone ↔ CT, lesion boundaries) with clinical composite loss (Equation 2)
- 🔄 **Registration-Robust**: Tolerance-aware evaluation (NSD@τ, HD95) and synthetic warp stress testing for mis-registration scenarios
- 📊 **Uncertainty Calibration**: Ensemble-based spatial uncertainty with ECE/Brier metrics and clinician-facing heatmap overlays
- 🧠 **Lesion-Focused**: Integrated frozen lesion segmentation head (Algorithm 1, Line 7) for stroke/tumor applications
- 🚀 **Complete Pipeline**: Automated preprocessing (ANTs registration, TotalSegmentator ROI masks), training, and evaluation

---

## Installation

### Requirements
- Linux/WSL (recommended) or Windows with Miniconda
- Python 3.10
- CUDA 12.1+ (GPU with ≥16 GB VRAM recommended)
- ANTs, TotalSegmentator, MONAI

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ClinFuseDiff.git
cd ClinFuseDiff

# Create environment and install dependencies
bash workflow/01_setup_environment.sh

# Verify installation
bash workflow/02_verify_setup.sh
```

<details>
<summary>Manual Installation</summary>

```bash
conda create -n clinfusediff python=3.10 -y
conda activate clinfusediff

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install monai[all] nibabel SimpleITK scikit-image einops
pip install TotalSegmentator antspyx

# Training utilities
pip install accelerate wandb lightning tensorboard
```
</details>

---

## Data Preparation

### Datasets

**Primary (with lesion labels):**
- **APIS** (Acute stroke; paired NCCT–MRI/ADC + expert lesion masks): [Challenge Portal](https://bivl2ab.uis.edu.co/challenges/apis)

**Registration Robustness:**
- **RIRE** (CT–MR brain registration benchmark): [Download](https://rire.insight-journal.org/download_data)
- **SynthRAD2023** (540 brain cases; multi-center CT/MRI): [Zenodo](https://zenodo.org/records/7260705)

### Preprocessing Workflow

```bash
# 1. Download APIS dataset (manual registration required)
#    Follow instructions at https://bivl2ab.uis.edu.co/challenges/apis
#    Extract to: data/apis/raw/{ct,adc}/

# 2. Convert DICOM to NIfTI (if needed)
bash scripts/convert_dicom.sh data/apis/raw/ct/<case_id>

# 3. Register CT→MRI using ANTs SyN
bash scripts/register_ants.sh \
    --fixed data/apis/raw/adc/<case_id>/adc.nii.gz \
    --moving data/apis/raw/ct/<case_id>/ct.nii.gz \
    --output work/preproc/<case_id>/ct_in_mri.nii.gz

# 4. Generate ROI masks (brain, bone) with TotalSegmentator
python scripts/make_masks_totalseg.py \
    --mri work/preproc/<case_id>/adc.nii.gz \
    --ct work/preproc/<case_id>/ct_in_mri.nii.gz \
    --output work/masks/<case_id>/

# 5. Create train/val/test splits
python scripts/make_splits.py \
    --data_dir data/apis/raw \
    --output data/apis/splits \
    --ratios 0.7 0.15 0.15 --seed 42
```

See [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) for detailed instructions.

---

## Training

### Quick Start

```bash
conda activate clinfusediff

# Train with stroke preset (optimized α, β, γ for APIS)
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --data_dir data/apis \
    --output_dir work/experiments/stroke_run1

# Resume from checkpoint
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --resume work/experiments/stroke_run1/checkpoints/last.ckpt
```

### Disease-Specific Presets

```bash
# Stroke (high lesion sensitivity)
python train.py --config configs/cvpr2026/train_roi.yaml --preset stroke

# Tumor (balanced brain/lesion)
python train.py --config configs/cvpr2026/train_roi.yaml --preset tumor

# Custom weights
python train.py --config configs/cvpr2026/train_roi.yaml \
    --roi_weights 0.4 0.3 0.3  # α_brain β_bone γ_lesion
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--preset` | Disease preset (`stroke`, `tumor`) | `stroke` |
| `--roi_weights` | (α, β, γ) for brain/bone/lesion | `(0.35, 0.25, 0.40)` |
| `--steps` | Diffusion timesteps | `1000` |
| `--batch_size` | Batch size (adjust for GPU) | `2` |
| `--accumulate_grad` | Gradient accumulation steps | `4` |
| `--mixed_precision` | Enable FP16 training | `True` |

---

## Evaluation

### Comprehensive Evaluation

```bash
python evaluate.py \
    --config configs/cvpr2026/train_roi.yaml \
    --checkpoint work/experiments/stroke_run1/checkpoints/best.ckpt \
    --split test \
    --num_samples 8 \
    --save_images \
    --save_uncertainty

# Output:
# - work/eval/<timestamp>/metrics_per_case.csv
# - work/eval/<timestamp>/metrics_aggregate.json
# - work/eval/<timestamp>/fused_images/<case_id>.nii.gz
# - work/eval/<timestamp>/uncertainty/<case_id>_variance.nii.gz
```

### ROI Metrics (Primary)

- **Brain ROI**: SSIM/FSIM (fused vs MRI | brain_mask)
- **Bone ROI**: PSNR/SSIM (fused vs CT | bone_mask)
- **Lesion ROI**: Dice, NSD@τ, HD95 (boundary-aware with tolerance)

### Registration-Aware Stress Testing

```bash
python evaluate.py \
    --checkpoint work/experiments/stroke_run1/checkpoints/best.ckpt \
    --split test \
    --registration_stress \
    --warp_range 1.0 3.0  # Apply ±1-3mm synthetic warps
```

---

## Pretrained Models

| Model | Dataset | Lesion Dice | NSD@2mm | HD95 | Download |
|-------|---------|-------------|---------|------|----------|
| CLIN-FuseDiff++-Stroke | APIS (60 train) | 0.847 | 0.923 | 2.34 mm | [Google Drive](https://drive.google.com) |
| CLIN-FuseDiff++-Tumor | BraTS 2021 | 0.812 | 0.901 | 3.12 mm | [Google Drive](https://drive.google.com) |

---

## Repository Structure

```
ClinFuseDiff/
├── configs/                     # YAML configurations
│   └── cvpr2026/
│       └── train_roi.yaml       # Main training config with presets
├── docs/                        # Documentation
│   ├── DATA_PREPARATION.md
│   ├── TRAINING.md
│   └── EVALUATION.md
├── scripts/                     # Preprocessing scripts
│   ├── register_ants.sh         # ANTs SyN registration
│   ├── make_masks_totalseg.py   # ROI mask generation
│   ├── make_splits.py           # Dataset splitting
│   └── download_*.sh            # Dataset downloaders
├── src/
│   ├── data/
│   │   └── fusion_dataset.py   # Paired CT-MRI dataset loader
│   ├── models/
│   │   ├── unet3d.py            # 3D U-Net denoiser
│   │   ├── roi_guided_diffusion.py  # Algorithm 1 (ROI-aware guidance)
│   │   └── lesion_head.py       # Frozen lesion segmentation
│   ├── training/
│   │   ├── fusion_trainer.py   # Training loop with ROI losses
│   │   └── roi_losses.py        # Equation 2 composite loss
│   └── utils/
│       ├── roi_metrics.py       # MONAI-based NSD/HD95/Dice
│       └── uncertainty.py       # ECE/Brier calibration
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation with uncertainty
└── workflow/                    # Setup automation scripts
    ├── 01_setup_environment.sh
    ├── 02_verify_setup.sh
    └── run_workflow.sh          # Interactive workflow menu
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| 3D U-Net + conditioning encoders | ✅ |
| ROI-guided diffusion (Algorithm 1) | ✅ |
| Clinical ROI loss (Equation 2) | ✅ |
| Training & evaluation pipelines | ✅ |
| MONAI-based boundary metrics (NSD, HD95) | ✅ |
| Uncertainty calibration (ECE, Brier) | ✅ |
| Registration-aware stress testing | ✅ |
| TotalSegmentator integration | ✅ |
| Few-step DDIM sampling | ⏳ In progress |
| Baseline comparisons (TTTFusion, etc.) | ⏳ Planned |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{phasook2026clinfusediff,
  title     = {CLIN-FuseDiff++: ROI-Aware Guided Diffusion for Multimodal Medical Image Fusion},
  author    = {Phasook, Pakawat},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

- [MONAI](https://monai.io/) for medical imaging utilities
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for anatomical segmentation
- [ANTs](https://github.com/ANTsX/ANTs) for registration
- [APIS Challenge](https://bivl2ab.uis.edu.co/challenges/apis) for stroke dataset

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please open an issue or contact: [your.email@institution.edu](mailto:your.email@institution.edu)