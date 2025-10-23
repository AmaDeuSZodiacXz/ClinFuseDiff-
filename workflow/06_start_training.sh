#!/usr/bin/env bash
# Step 6: launch ClinFuseDiff training with preset selection

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda activate clinfusediff 2>/dev/null; then
  echo "ERROR: 'clinfusediff' environment not found. Run workflow/01_setup_environment.sh first." >&2
  exit 1
fi

declare -A PRESETS=(
  [1]="stroke"
  [2]="brain_tumor"
  [3]="bone_tumor"
  [4]="metastasis"
  [5]="default"
)

echo "==========================================="
echo "ClinFuseDiff • Step 6: Start Training"
echo "==========================================="
echo "Select disease preset:"
echo "  1) Stroke            (APIS default)"
echo "  2) Brain Tumor"
echo "  3) Bone Tumor"
echo "  4) Metastasis"
echo "  5) Default (balanced weights)"
echo
read -rp "Enter choice [1-5]: " choice

PRESET="${PRESETS[$choice]:-stroke}"

echo
echo "Starting training with preset: ${PRESET}"
echo "Logs and checkpoints will be saved under work/experiments/"
echo

python train.py --config configs/cvpr2026/train_roi.yaml --preset "${PRESET}"

echo
echo "==========================================="
echo "Training run finished."
echo "Check results in work/experiments/"
echo "==========================================="
