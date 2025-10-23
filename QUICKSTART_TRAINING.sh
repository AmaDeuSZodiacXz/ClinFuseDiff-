#!/bin/bash
# CLIN-FuseDiff++ Quick Start Training Script
# Run this in Google Colab Enterprise with A100 GPU

set -e  # Exit on error

echo "======================================================================="
echo "CLIN-FuseDiff++ CVPR 2026 - Quick Start"
echo "======================================================================="
echo ""

# ============================================================================
# STEP 1: Setup Environment
# ============================================================================
echo "[STEP 1/7] Setting up environment..."

# Clone repo (if not already)
if [ ! -d "ClinFuseDiff-" ]; then
    git clone https://github.com/AmaDeuSZodiacXz/ClinFuseDiff-.git
fi
cd ClinFuseDiff-

# Create environment
conda create -n clinfusediff python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clinfusediff

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "✓ Environment setup complete"
echo ""

# ============================================================================
# STEP 2: Download APIS Dataset
# ============================================================================
echo "[STEP 2/7] Downloading APIS dataset..."

pip install -U huggingface_hub

huggingface-cli download \
    Pakawat-Phasook/ClinFuseDiff-APIS-Data \
    --repo-type dataset \
    --local-dir data/apis

echo "✓ Dataset downloaded (60 cases)"
echo "  Train: 42 cases"
echo "  Val:    9 cases"
echo "  Test:   9 cases"
echo ""

# ============================================================================
# STEP 3: Verify Setup
# ============================================================================
echo "[STEP 3/7] Verifying installation..."

python -c "
import torch
import monai
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'✓ MONAI: {monai.__version__}')
"

# Verify data
echo ""
echo "Verifying dataset..."
python -c "
from pathlib import Path
preproc_dir = Path('data/apis/preproc')
cases = list(preproc_dir.glob('train_*'))
print(f'✓ Found {len(cases)} cases in data/apis/preproc/')
"

echo ""

# ============================================================================
# STEP 4: (Optional) Setup WandB
# ============================================================================
echo "[STEP 4/7] WandB setup (optional)..."
echo "Do you want to enable WandB logging? (y/n)"
read -r use_wandb

if [ "$use_wandb" = "y" ]; then
    echo "Please enter your WandB API key:"
    echo "(Get it from: https://wandb.ai/authorize)"
    wandb login
    wandb_flag="--wandb"
else
    echo "✓ Skipping WandB (will use --no-wandb)"
    wandb_flag="--no-wandb"
fi

echo ""

# ============================================================================
# STEP 5: Start Training
# ============================================================================
echo "[STEP 5/7] Starting training..."
echo ""
echo "Configuration:"
echo "  - Model: CLIN-FuseDiff++ (20M parameters)"
echo "  - Preset: stroke (α=1.0, β=0.8, γ=2.5)"
echo "  - Epochs: 200"
echo "  - Expected time: ~33 hours on A100"
echo "  - Checkpoints: work/experiments/clinfusediff_cvpr2026/checkpoints/"
echo ""
echo "Training will start in 5 seconds..."
sleep 5

python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    $wandb_flag

echo ""
echo "✓ Training complete!"
echo ""

# ============================================================================
# STEP 6: Evaluate on Test Set
# ============================================================================
echo "[STEP 6/7] Evaluating on test set..."

python evaluate.py \
    --config configs/cvpr2026/train_roi.yaml \
    --checkpoint work/experiments/clinfusediff_cvpr2026/checkpoints/best.pth \
    --split test \
    --num-samples 5 \
    --save-images \
    --output work/results/clinfusediff_test

echo ""
echo "✓ Evaluation complete!"
echo "  Results: work/results/clinfusediff_test/"
echo ""

# ============================================================================
# STEP 7: Run Baselines
# ============================================================================
echo "[STEP 7/7] Running baseline comparisons..."

# Simple average
python baselines/simple_methods/average.py \
    --data-dir data/apis/preproc \
    --split test \
    --save-images \
    --output work/results/baselines/average

# Weighted average
python baselines/simple_methods/weighted_average.py \
    --data-dir data/apis/preproc \
    --split test \
    --optimize-weights \
    --save-images \
    --output work/results/baselines/weighted_opt

echo ""
echo "✓ Baselines complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================="
echo "All Done! 🎉"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  1. CLIN-FuseDiff++: work/results/clinfusediff_test/"
echo "  2. Simple Average:  work/results/baselines/average/"
echo "  3. Weighted Avg:    work/results/baselines/weighted_opt/"
echo ""
echo "View metrics:"
echo "  cat work/results/clinfusediff_test/aggregate_metrics.json"
echo ""
echo "Generate comparison table:"
echo "  python scripts/compare_results.py"
echo ""
echo "======================================================================="
