<div align="center">

# CLIN-FuseDiff++

**ROI-Aware Guided Diffusion for Multimodal Medical Image Fusion**

[![arXiv](https://img.shields.io/badge/arXiv-2026.00000-b31b1b.svg)](https://arxiv.org/abs/2026.00000)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Paper**](https://arxiv.org/abs/2026.00000) | [**Dataset**](https://huggingface.co/datasets/Pakawat-Phasook/ClinFuseDiff-APIS-Data) | [**Pretrained Models**](#pretrained-models)

</div>

---

## 🔥 Highlights

- 🎯 **ROI-Aware Guidance**: Region-specific fusion with clinical composite loss (brain ↔ MRI, bone ↔ CT, lesion boundaries)
- 🔄 **Registration-Robust**: Tolerance-aware evaluation (NSD@τmm, HD95) with synthetic warp stress testing
- 📊 **Uncertainty Calibration**: Ensemble-based spatial uncertainty with ECE/Brier metrics
- 🧠 **Lesion-Focused**: Integrated frozen segmentation head for stroke/tumor applications
- 🚀 **Complete Pipeline**: End-to-end preprocessing, training, and evaluation

**Target Conference**: CVPR 2026 (primary) | ICLR 2026 (fallback)

---

## 📋 Table of Contents

- [Installation](#️-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Pretrained Models](#-pretrained-models)
- [Citation](#-citation)

---

## 🛠️ Installation

### Requirements

- Linux/WSL (recommended) or macOS  
- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 16GB+ GPU VRAM (A100/V100 recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/AmaDeuSZodiacXz/ClinFuseDiff.git
cd ClinFuseDiff
```

### Step 2: Create Environment

```bash
# Create conda environment
conda create -n clinfusediff python=3.10 -y
conda activate clinfusediff
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📦 Dataset Preparation

### Download Preprocessed APIS Dataset

We provide preprocessed APIS dataset (60 paired CT-MRI scans with lesion masks) on Hugging Face:

```bash
# Install Hugging Face CLI
pip install -U huggingface_hub

# Download dataset (~226MB)
huggingface-cli download \
    Pakawat-Phasook/ClinFuseDiff-APIS-Data \
    --repo-type dataset \
    --local-dir data/apis
```

**Verify download:**

```bash
ls data/apis/
# Expected: preproc/  raw/  splits/  splits.json

ls data/apis/preproc/ | wc -l
# Expected: 60

cat data/apis/splits/train.txt | wc -l
# Expected: 42
```

### Dataset Structure

```
data/apis/
├── preproc/                    # Preprocessed volumes
│   ├── train_000/
│   │   ├── ct.nii.gz          # CT scan
│   │   ├── mri.nii.gz         # MRI/ADC scan
│   │   ├── brain_mask.nii.gz  # Brain ROI
│   │   ├── bone_mask.nii.gz   # Bone ROI
│   │   └── lesion_mask.nii.gz # Expert annotation
│   └── ...
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── splits.json
```

---

## 🚀 Training

### Quick Start

```bash
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke
```

### Disease-Specific Presets

```bash
# Stroke (default for APIS)
python train.py --config configs/cvpr2026/train_roi.yaml --preset stroke

# Brain tumor
python train.py --config configs/cvpr2026/train_roi.yaml --preset brain_tumor

# Bone tumor
python train.py --config configs/cvpr2026/train_roi.yaml --preset bone_tumor

# Metastasis
python train.py --config configs/cvpr2026/train_roi.yaml --preset metastasis
```

### Custom ROI Weights

```bash
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --alpha 1.5 \    # Brain region weight
    --beta 0.5 \     # Bone region weight
    --gamma 2.0      # Lesion region weight
```

### With WandB Logging

```bash
wandb login YOUR_API_KEY

python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --wandb
```

### Expected Training Time

| GPU | Time/Epoch | Full Training (200 epochs) |
|-----|------------|----------------------------|
| A100 (40GB) | ~10 min | ~33 hours |
| V100 (32GB) | ~15 min | ~50 hours |
| RTX 4090 (24GB) | ~20 min | ~67 hours |

---

## 📊 Evaluation

### Inference

```bash
python evaluate.py \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --config configs/cvpr2026/train_roi.yaml \
    --split test \
    --output-dir work/experiments/clinfusediff_cvpr2026/results
```

### Evaluation Metrics

**ROI Metrics** (Primary):
- Lesion: Dice, NSD@2mm, HD95
- Brain: SSIM, FSIM
- Bone: PSNR, SSIM

**Global Metrics** (Secondary):
- PSNR, SSIM, FSIM, FMI

**Uncertainty Metrics**:
- ECE, Brier Score

### Ensemble Sampling

```bash
python evaluate.py \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --config configs/cvpr2026/train_roi.yaml \
    --num-samples 5 \
    --save-uncertainty-maps
```

---

## 🎯 Pretrained Models

Coming soon. Checkpoints will be released on Hugging Face Model Hub.

---

## 📖 Citation

```bibtex
@article{clinfusediff2026,
  title={CLIN-FuseDiff++: ROI-Aware Guided Diffusion for Multimodal Medical Image Fusion},
  author={Your Name},
  journal={arXiv preprint arXiv:2026.00000},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- [LeFusion](https://github.com/HINTLab/LeFusion) - Lesion-focused diffusion
- [MONAI](https://monai.io/) - Medical imaging framework
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - ROI segmentation
- [APIS Dataset](https://github.com/Tabrisrei/ISLES2022-SPES) - Acute stroke imaging

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**[⬆ Back to Top](#clin-fusediff)**

Made with ❤️ for advancing medical image fusion

</div>
