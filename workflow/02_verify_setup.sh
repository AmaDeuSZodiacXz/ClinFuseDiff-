#!/usr/bin/env bash
# Step 2: verify ClinFuseDiff environment setup

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==========================================="
echo "ClinFuseDiff • Step 2: Verify Environment"
echo "==========================================="
echo

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda command not found. Install Miniconda/Anaconda before running this step." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda activate clinfusediff 2>/dev/null; then
  echo "ERROR: 'clinfusediff' environment not found. Run workflow/01_setup_environment.sh first." >&2
  exit 1
fi

echo "[1/5] Python version"
python --version
echo

echo "[2/5] PyTorch + CUDA availability"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo

echo "[3/5] MONAI"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
echo

echo "[4/5] TotalSegmentator CLI"
if command -v TotalSegmentator &>/dev/null; then
  echo "TotalSegmentator: $(command -v TotalSegmentator)"
else
  echo "WARNING: TotalSegmentator executable not found on PATH. Ensure the environment is activated before running CLI commands."
fi
echo

echo "[5/5] Project entrypoints"
if python train.py --help >/dev/null 2>&1; then
  echo "train.py: OK"
else
  echo "train.py: ERROR"
fi

if python evaluate.py --help >/dev/null 2>&1; then
  echo "evaluate.py: OK"
else
  echo "evaluate.py: ERROR"
fi

echo
echo "==========================================="
echo "Verification complete."
echo "==========================================="
