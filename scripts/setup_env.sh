#!/bin/bash
# Setup development environment for ClinFuseDiff CVPR 2026

set -e

echo "==========================================="
echo "ClinFuseDiff Environment Setup"
echo "==========================================="
echo ""

ENV_NAME="clinfusediff"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "  Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing MONAI and medical imaging tools..."
pip install monai[all] nibabel SimpleITK scikit-image scikit-learn einops

echo ""
echo "Installing diffusion models and training tools..."
pip install accelerate diffusers transformers lightning wandb tensorboard

echo ""
echo "Installing TotalSegmentator for ROI segmentation..."
pip install TotalSegmentator

echo ""
echo "Installing registration and preprocessing tools..."
# ANTsPy (Python wrapper for ANTs)
pip install antspyx

# DICOM conversion
echo ""
echo "Installing dcm2niix for DICOM conversion..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected - install dcm2niix via apt:"
    echo "  sudo apt-get update && sudo apt-get install -y dcm2niix"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected - install dcm2niix via homebrew:"
    echo "  brew install dcm2niix"
else
    echo "Windows detected - download dcm2niix from:"
    echo "  https://github.com/rordenlab/dcm2niix/releases"
fi

echo ""
echo "Installing additional utilities..."
pip install pyyaml tqdm pandas matplotlib seaborn plotly

echo ""
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo ""
echo "==========================================="
echo "✓ Environment setup complete!"
echo "==========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import monai; print(f\"MONAI: {monai.__version__}\")'"
echo "  TotalSegmentator --help"
echo "  dcm2niix -version"
echo ""
echo "Next steps:"
echo "  1. Download datasets: bash scripts/download_apis.sh"
echo "  2. Preprocess data: bash scripts/register_ants.sh"
echo "  3. Train model: python train.py --config configs/cvpr2026/train_roi.yaml"
echo ""