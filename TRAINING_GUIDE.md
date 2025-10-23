# CLIN-FuseDiff++ Training & Evaluation Guide

Complete step-by-step guide from training to evaluation.

---

## 📋 Prerequisites

- Google Colab Enterprise with A100 GPU (40GB VRAM)
- GitHub account
- Hugging Face account
- WandB account (optional, for logging)

---

## 🚀 Step-by-Step Workflow

---

### **STEP 1: Setup Environment (10 minutes)**

```bash
# 1.1 Clone repository
git clone https://github.com/AmaDeuSZodiacXz/ClinFuseDiff-.git
cd ClinFuseDiff-

# 1.2 Create conda environment
conda create -n clinfusediff python=3.10 -y
conda activate clinfusediff

# 1.3 Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 1.4 Install dependencies
pip install -r requirements.txt

# 1.5 Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
MONAI: 1.3.x
```

---

### **STEP 2: Download APIS Dataset (5 minutes)**

```bash
# 2.1 Install Hugging Face CLI
pip install -U huggingface_hub

# 2.2 Download preprocessed APIS dataset (~226MB)
huggingface-cli download \
    Pakawat-Phasook/ClinFuseDiff-APIS-Data \
    --repo-type dataset \
    --local-dir data/apis

# 2.3 Verify dataset structure
ls data/apis/
# Expected: preproc/  raw/  splits/  splits.json

ls data/apis/preproc/ | wc -l
# Expected: 60 (60 cases)

cat data/apis/splits/train.txt | wc -l
# Expected: 42 (training cases)

cat data/apis/splits/val.txt | wc -l
# Expected: 9 (validation cases)

cat data/apis/splits/test.txt | wc -l
# Expected: 9 (test cases)
```

---

### **STEP 3: (Optional) Setup WandB Logging**

```bash
# 3.1 Login to WandB
wandb login

# 3.2 Enter your API key when prompted
# Get key from: https://wandb.ai/authorize
```

**If you skip this step:** Use `--no-wandb` flag in training command

---

### **STEP 4: Start Training (33 hours on A100)**

#### **Option A: With WandB Logging (Recommended)**

```bash
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --wandb
```

#### **Option B: Without WandB**

```bash
python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --no-wandb
```

#### **Training Progress:**

You will see output like:
```
======================================================================
CLIN-FuseDiff++ Training (CVPR 2026)
ROI-Aware Guided Diffusion for Medical Image Fusion
======================================================================

Loading config: configs/cvpr2026/train_roi.yaml
✓ Applied preset: stroke
  Description: Ischemic lesion and tissue viability
  α (brain): 1.0
  β (bone): 0.8
  γ (lesion): 2.5
✓ Random seed: 42
✓ Device: cuda
  GPU: NVIDIA A100-SXM4-40GB
  CUDA version: 12.1

Creating dataloaders...
✓ Created dataloaders:
  Train: 42 cases, 42 batches
  Val: 9 cases, 9 batches

Creating model...
✓ Model created:
  Total parameters: 20,341,824
  Trainable parameters: 20,341,824

Creating trainer...
✓ FusionTrainer initialized
  Device: cuda
  Experiment dir: work/experiments/clinfusediff_cvpr2026
  Mixed precision: True
  Gradient accumulation: 8

======================================================================
Starting training...
======================================================================

Epoch 1/200: 100%|████| 42/42 [01:24<00:00, loss=0.3421, diff=0.2104, roi=0.1317, lr=1.00e-05]
Validation 1: 100%|████████| 9/9 [00:15<00:00]
✓ New best lesion/nsd@2mm: 0.4521

Epoch 1 Summary:
  Train Loss: 0.3421
    Diffusion: 0.2104
    ROI: 0.1317
  LR: 1.00e-05

...
```

#### **Training Checkpoints:**

Checkpoints are saved to: `work/experiments/clinfusediff_cvpr2026/checkpoints/`

```
work/experiments/clinfusediff_cvpr2026/
├── checkpoints/
│   ├── best.pth          # Best model (max lesion/nsd@2mm)
│   ├── last.pth          # Last epoch
│   ├── epoch_005.pth     # Periodic (every 5 epochs)
│   ├── epoch_010.pth
│   └── ...
└── wandb/                # WandB logs (if enabled)
```

#### **Monitoring Training:**

If using WandB, view at: https://wandb.ai/your-username/clinfusediff-cvpr2026

Metrics logged:
- `train/loss_total`, `train/loss_diffusion`, `train/loss_roi`
- `train/loss_roi_brain`, `train/loss_roi_bone`, `train/loss_roi_lesion`
- `train/learning_rate`
- `val/lesion/dice`, `val/lesion/nsd`, `val/brain/ssim`, etc.

---

### **STEP 5: Evaluate Trained Model (1 hour)**

```bash
# 5.1 Evaluate on test set
python evaluate.py \
    --config configs/cvpr2026/train_roi.yaml \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --split test \
    --num-samples 5 \
    --save-images \
    --output work/results/clinfusediff_test
```

**Arguments explained:**
- `--config`: Same config used for training
- `--checkpoint`: Path to trained model (use `best.pth`)
- `--split`: Which split to evaluate (`test`, `val`, or `train`)
- `--num-samples`: Number of diffusion samples for uncertainty (default: 5)
- `--save-images`: Save fused images and uncertainty maps
- `--output`: Output directory for results

#### **Evaluation Output:**

```
======================================================================
CLIN-FuseDiff++ Evaluation (CVPR 2026)
======================================================================

✓ Loaded config: configs/cvpr2026/train_roi.yaml
✓ Model loaded successfully
✓ Lesion head loaded
✓ Created dataloader for 'test' split:
  Cases: 9
  Batches: 9

======================================================================
Evaluating on 'test' split...
======================================================================

Evaluating: 100%|████████| 9/9 [00:52<00:00, SSIM_brain=0.921, Dice=0.782]

✓ Saved per-case metrics: work/results/clinfusediff_test/per_case_metrics.csv
✓ Saved aggregate metrics: work/results/clinfusediff_test/aggregate_metrics.json

======================================================================
Evaluation Results Summary
======================================================================
Number of cases: 9

Key Metrics (mean ± std):
  lesion/dice         : 0.7821 ± 0.0854
  lesion/nsd          : 0.8431 ± 0.0721
  lesion/hd95         : 3.2145 ± 0.9821
  brain/ssim          : 0.9215 ± 0.0321
  brain/fsim          : 0.9012 ± 0.0287
  bone/psnr           : 29.1234 ± 1.2341
  bone/ssim           : 0.8821 ± 0.0412
  calibration/ece     : 0.0521 ± 0.0123
  calibration/brier   : 0.0821 ± 0.0198
  uncertainty/mean    : 0.0234 ± 0.0067
======================================================================
Evaluation complete!
Results saved to: work/results/clinfusediff_test
======================================================================
```

#### **Result Files:**

```
work/results/clinfusediff_test/
├── per_case_metrics.csv          # Metrics for each case
├── aggregate_metrics.json        # Mean/std/median/min/max
└── images/                       # Fused images (if --save-images)
    ├── train_000_fused.nii.gz           # Fused volume
    ├── train_000_uncertainty.nii.gz     # Uncertainty map (std)
    ├── train_000_lesion_pred.nii.gz     # Lesion prediction
    └── ...
```

---

### **STEP 6: Run Baseline Comparisons (30 minutes)**

```bash
# 6.1 Simple Average Baseline
python baselines/simple_methods/average.py \
    --data-dir data/apis/preproc \
    --split test \
    --save-images \
    --output work/results/baselines/average

# 6.2 Weighted Average Baseline (ROI-optimized)
python baselines/simple_methods/weighted_average.py \
    --data-dir data/apis/preproc \
    --split test \
    --optimize-weights \
    --save-images \
    --output work/results/baselines/weighted_opt

# 6.3 (Optional) Fixed Weighted Average
python baselines/simple_methods/weighted_average.py \
    --data-dir data/apis/preproc \
    --split test \
    --alpha 0.8 --beta 0.2 \
    --output work/results/baselines/weighted_8020
```

---

### **STEP 7: Generate Comparison Table**

```bash
# Create comparison summary
python -c "
import pandas as pd
import json
from pathlib import Path

methods = {
    'Simple Average': 'work/results/baselines/average/aggregate_metrics.json',
    'Weighted (Opt)': 'work/results/baselines/weighted_opt/aggregate_metrics.json',
    'CLIN-FuseDiff++': 'work/results/clinfusediff_test/aggregate_metrics.json'
}

metrics = ['lesion/dice', 'lesion/nsd', 'brain/ssim', 'bone/psnr', 'calibration/ece', 'calibration/brier']

results = []
for method_name, json_path in methods.items():
    with open(json_path) as f:
        data = json.load(f)

    row = {'Method': method_name}
    for metric in metrics:
        mean_key = metric + '/mean'
        std_key = metric + '/std'
        if mean_key in data:
            row[metric] = f\"{data[mean_key]:.4f} ± {data.get(std_key, 0):.4f}\"
        else:
            row[metric] = '-'
    results.append(row)

df = pd.DataFrame(results)
df.to_csv('work/results/comparison_table.csv', index=False)
print(df.to_string(index=False))
"
```

**Example Output:**
```
           Method  lesion/dice     lesion/nsd     brain/ssim       bone/psnr calibration/ece calibration/brier
   Simple Average  0.4521 ± 0.1521 0.5234 ± 0.1821 0.7234 ± 0.0821 22.5134 ± 2.1234               -                 -
  Weighted (Opt)  0.5234 ± 0.1421 0.6021 ± 0.1621 0.7821 ± 0.0721 24.1234 ± 2.0234               -                 -
CLIN-FuseDiff++  0.7821 ± 0.0854 0.8431 ± 0.0721 0.9215 ± 0.0321 29.1234 ± 1.2341 0.0521 ± 0.0123   0.0821 ± 0.0198
```

---

## 📊 Expected Results (Target Performance)

| Metric | Simple Average | Weighted (Opt) | **CLIN-FuseDiff++** | Target |
|--------|----------------|----------------|---------------------|--------|
| Lesion Dice ↑ | 0.45±0.15 | 0.52±0.14 | **0.78±0.09** | >0.75 |
| Lesion NSD@2mm ↑ | 0.52±0.18 | 0.60±0.16 | **0.84±0.10** | >0.80 |
| Brain SSIM ↑ | 0.72±0.08 | 0.78±0.07 | **0.92±0.03** | >0.90 |
| Bone PSNR ↑ | 22.5±2.1 | 24.1±2.0 | **29.1±1.3** | >28.0 |
| ECE ↓ | - | - | **0.05±0.01** | <0.10 |
| Brier ↓ | - | - | **0.08±0.02** | <0.10 |

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

```bash
# Reduce batch size (already 1 in config)
# Disable mixed precision
# Edit configs/cvpr2026/train_roi.yaml:
#   mixed_precision: false
```

### Issue 2: Training Too Slow

```bash
# Verify GPU utilization
nvidia-smi

# Should show ~90-100% GPU usage
# If low, check:
# - num_workers in config (should be 4)
# - gradient_accumulation_steps (should be 8)
```

### Issue 3: Validation Metrics Not Improving

```bash
# Check learning rate schedule
# View WandB: train/learning_rate should increase (warmup) then decrease (cosine)

# Check ROI guidance weights
# For stroke: α=1.0, β=0.8, γ=2.5
```

### Issue 4: Data Not Found

```bash
# Verify data structure
ls data/apis/preproc/
# Should have 60 directories: train_000, train_001, ..., train_059

# Check each case has required files
ls data/apis/preproc/train_000/
# Expected: ct.nii.gz  mri.nii.gz  brain_mask.nii.gz  bone_mask.nii.gz  lesion_mask.nii.gz
```

---

## 📝 Notes

- **Training time**: ~33 hours on A100 40GB (200 epochs)
- **Evaluation time**: ~1 hour for test set (9 cases)
- **Total disk space**: ~5GB (code + data + checkpoints)
- **Best checkpoint**: Selected by max `lesion/nsd@2mm` on validation set
- **Uncertainty**: Computed from 5 ensemble samples (configurable via `--num-samples`)

---

## 🎯 For CVPR 2026 Paper

Your paper comparison table should include:
1. **Simple Average** (trivial baseline)
2. **Weighted Average** (optimized baseline)
3. **CLIN-FuseDiff++** (your method)
4. (Optional) Diff-IF, SwinFusion (if time permits)

Emphasis:
- Show improvement in **accuracy** (Dice, SSIM, PSNR)
- Highlight **calibration** (ECE, Brier) as novelty
- Discuss **clinical utility** of well-calibrated uncertainty

---

## 📧 Support

See main [README.md](README.md) for more information.