# Google Colab Quick Start Guide

This guide helps you train CLIN-FuseDiff++ on Google Colab with A100 GPU.

## Prerequisites
- Google Colab Enterprise account (for A100 GPU)
- GitHub repository: https://github.com/AmaDeuSZodiacXz/ClinFuseDiff
- Hugging Face dataset: https://huggingface.co/datasets/Pakawat-Phasook/ClinFuseDiff-APIS-Data

---

## Step 1: Clone Repository

```bash
!git clone https://github.com/AmaDeuSZodiacXz/ClinFuseDiff.git
%cd ClinFuseDiff
```

---

## Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support (Colab usually has this)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Step 3: Download APIS Dataset from Hugging Face

```python
from huggingface_hub import snapshot_download

# Download preprocessed APIS data (226MB)
print("Downloading APIS dataset from Hugging Face...")
snapshot_download(
    repo_id="Pakawat-Phasook/ClinFuseDiff-APIS-Data",
    repo_type="dataset",
    local_dir="data/apis",
    token=None  # Public dataset, no token needed
)

# Verify data structure
!ls data/apis/
!ls data/apis/preproc/ | head -5
!cat data/apis/splits/train.txt | wc -l  # Should show 42 training cases
```

---

## Step 4: Start Training

### Option A: Quick Test (1 epoch)

```bash
!python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --num-epochs 1 \
    --device cuda
```

### Option B: Full Training (200 epochs, ~hours)

```bash
!python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --device cuda
```

### Option C: With WandB Logging

```bash
# Login to WandB
!wandb login YOUR_WANDB_API_KEY

# Train with WandB
!python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --wandb \
    --device cuda
```

---

## Step 5: Monitor Training

```python
# View training logs
!tail -f work/experiments/clinfusediff_cvpr2026/logs/training.log
```

Or open TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir work/experiments/clinfusediff_cvpr2026/logs
```

---

## Step 6: Evaluate Model

```bash
!python evaluate.py \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --config configs/cvpr2026/train_roi.yaml \
    --output-dir work/experiments/clinfusediff_cvpr2026/eval
```

---

## Expected Training Time (A100 GPU)

- **1 epoch**: ~10-15 minutes (42 training cases, batch_size=1, grad_accum=8)
- **Full training (200 epochs)**: ~30-50 hours
- **With mixed precision (FP16)**: ~20-30% faster

---

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

```yaml
# Edit configs/cvpr2026/train_roi.yaml
training:
  batch_size: 1  # Already at minimum
  gradient_accumulation_steps: 4  # Reduce from 8
  mixed_precision: true  # Enable FP16

model:
  unet3d:
    base_channels: 32  # Reduce from 64
```

### Slow Training

```bash
# Check GPU utilization
!nvidia-smi

# Monitor GPU usage during training
!watch -n 1 nvidia-smi
```

---

## Download Trained Model

After training completes, download checkpoints:

```python
from google.colab import files

# Download best checkpoint
files.download('work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth')

# Download metrics
files.download('work/experiments/clinfusediff_cvpr2026/metrics.csv')
```

---

## Disease-Specific Presets

Choose appropriate preset for your use case:

```bash
# Stroke (default for APIS dataset)
--preset stroke

# Brain tumor
--preset brain_tumor

# Bone tumor
--preset bone_tumor

# Metastasis
--preset metastasis

# Balanced
--preset default
```

---

## Custom ROI Weights

Override ROI weights directly:

```bash
!python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --alpha 1.5 \  # Brain region weight
    --beta 0.5 \   # Bone region weight
    --gamma 2.0 \  # Lesion region weight
    --device cuda
```

---

## Complete Workflow Example

```python
# 1. Setup
!git clone https://github.com/AmaDeuSZodiacXz/ClinFuseDiff.git
%cd ClinFuseDiff
!pip install -q -r requirements.txt

# 2. Download data
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Pakawat-Phasook/ClinFuseDiff-APIS-Data",
    repo_type="dataset",
    local_dir="data/apis"
)

# 3. Verify setup
!ls data/apis/preproc/ | wc -l  # Should show 60
!ls data/apis/splits/

# 4. Train
!python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --device cuda

# 5. Evaluate
!python evaluate.py \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --config configs/cvpr2026/train_roi.yaml

# 6. Download results
from google.colab import files
files.download('work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth')
```

---

## Support

- **GitHub Issues**: https://github.com/AmaDeuSZodiacXz/ClinFuseDiff/issues
- **Documentation**: See `README.md` and `CLAUDE.md`
- **Dataset**: https://huggingface.co/datasets/Pakawat-Phasook/ClinFuseDiff-APIS-Data