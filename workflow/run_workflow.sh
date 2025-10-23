#!/usr/bin/env bash
# Interactive workflow launcher for ClinFuseDiff (bash version of legacy RUN_WORKFLOW.bat)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pause() {
  read -rp $'\nPress <enter> to continue...'
}

ensure_conda() {
  if ! command -v conda &>/dev/null; then
    echo "ERROR: conda command not found. Install Miniconda/Anaconda first." >&2
    return 1
  fi
  return 0
}

activate_env() {
  ensure_conda || return 1
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate clinfusediff 2>/dev/null || {
    echo "ERROR: 'clinfusediff' environment not found. Run workflow/01_setup_environment.sh first." >&2
    return 1
  }
}

show_dataset_help() {
  cat <<'EOF'
============================================================
Step 3: Download Dataset (manual)
============================================================
APIS (required):
  1. Register at https://bivl2ab.uis.edu.co/challenges/apis
  2. Accept the data usage agreement
  3. Download the CT / ADC (MRI) / lesion mask archives
  4. Extract into data/apis/raw/{ct,adc,lesion_masks}

Optional (for registration robustness):
  • RIRE: https://rire.insight-journal.org/download_data
  • SynthRAD2023: follow links in docs/DATASET_SETUP.md

After downloading, run preprocessing scripts:
  - bash scripts/register_ants.sh --fixed <mri.nii.gz> --moving <ct.nii.gz> --out work/reg/<case>_
  - python scripts/make_masks_totalseg.py --ct <registered_ct> --mri <mri> --out work/masks/<case>
  - python scripts/make_splits.py --dataset apis --data-root data/apis
============================================================
EOF
  pause
}

show_preprocess_help() {
  cat <<'EOF'
============================================================
Step 4: Preprocess Data
============================================================
Example for one APIS case (run after downloading):

1) Register CT to MRI/ADC (ANTs SyN):
   bash scripts/register_ants.sh \
     --fixed data/apis/raw/adc/case_001.nii.gz \
     --moving data/apis/raw/ct/case_001.nii.gz \
     --out work/reg/case_001_

2) Generate ROI masks (TotalSegmentator):
   python scripts/make_masks_totalseg.py \
     --ct work/reg/case_001_Warped.nii.gz \
     --mri data/apis/raw/adc/case_001.nii.gz \
     --out work/masks/case_001

3) Normalise / cache dataset via ImageFusionDataset (see src/data/fusion_dataset.py)

Inspect outputs in work/reg/ and work/masks/ before scaling up.
============================================================
EOF
  pause
}

show_results_help() {
  cat <<'EOF'
============================================================
Step 8: View Results
============================================================
Key directories:
  • work/experiments/<run_id>/          – training logs/checkpoints
  • work/experiments/<run_id>/metrics   – per-step summaries
  • work/results/                       – evaluate.py outputs
  • work/results/images/                – fused images + uncertainty maps

For TensorBoard:
  tensorboard --logdir work/experiments
============================================================
EOF
  pause
}

show_status() {
  echo "============================================================"
  echo "Current Status"
  echo "============================================================"

  if activate_env 2>/dev/null; then
    echo "[1] Environment: clinfusediff (activated)"
  else
    echo "[1] Environment: NOT AVAILABLE"
  fi

  if [[ -d data/apis/raw/ct ]]; then
    count_ct=$(find data/apis/raw/ct -name '*.nii*' | wc -l)
    echo "[2] APIS raw data: ${count_ct} CT volumes found"
  else
    echo "[2] APIS raw data: not found"
  fi

  if [[ -d data/apis/preproc ]]; then
    echo "[3] Preprocessed data directory: present"
  else
    echo "[3] Preprocessed data directory: not found"
  fi

  if [[ -d work/experiments ]]; then
    echo "[4] Experiments:"
    find work/experiments -maxdepth 1 -mindepth 1 -type d -printf "    • %f\n"
  else
    echo "[4] Experiments: none yet"
  fi

  echo
  echo "Helpful references:"
  echo "  • CURRENT_STATUS.md"
  echo "  • QUICKSTART_CVPR2026.md"
  echo "  • docs/DATASET_SETUP.md"
  echo "============================================================"
  pause
}

start_training() {
  bash workflow/06_start_training.sh
  pause
}

evaluate_model() {
  activate_env || { pause; return; }
  read -rp "Path to checkpoint [default: work/experiments/latest/checkpoints/best.ckpt]: " checkpoint
  checkpoint=${checkpoint:-work/experiments/latest/checkpoints/best.ckpt}
  echo
  echo "Running evaluation on checkpoint: ${checkpoint}"
  python evaluate.py \
    --config configs/cvpr2026/train_roi.yaml \
    --checkpoint "${checkpoint}" \
    --split test \
    --save-images
  pause
}

while true; do
  cat <<'EOF'
============================================================
ClinFuseDiff CVPR 2026 – Workflow Launcher
============================================================
 1) Step 1: Setup Environment
 2) Step 2: Verify Environment
 3) Step 3: Download Dataset (info)
 4) Step 4: Preprocess Data (guidance)
 5) Step 5: Create Data Splits
 6) Step 6: Train Model
 7) Step 7: Evaluate Model
 8) Step 8: View Results Guide
 9) Quick Status Overview
 0) Exit
============================================================
EOF
  read -rp "Select an option [0-9]: " choice
  echo
  case "$choice" in
    1)
      bash workflow/01_setup_environment.sh
      pause
      ;;
    2)
      bash workflow/02_verify_setup.sh
      pause
      ;;
    3)
      show_dataset_help
      ;;
    4)
      show_preprocess_help
      ;;
    5)
      read -rp "Path to dataset directory for splitting [default: data/apis/preproc]: " data_dir
      data_dir=${data_dir:-data/apis/preproc}
      read -rp "Output directory for split files [default: $(dirname "${data_dir}")/splits]: " output_dir
      if [[ -n "${output_dir}" ]]; then
        python scripts/make_splits.py --data-dir "${data_dir}" --output-dir "${output_dir}"
      else
        python scripts/make_splits.py --data-dir "${data_dir}"
      fi
      pause
      ;;
    6)
      start_training
      ;;
    7)
      evaluate_model
      ;;
    8)
      show_results_help
      ;;
    9)
      show_status
      ;;
    0)
      echo "Good luck with ClinFuseDiff!"
      exit 0
      ;;
    *)
      echo "Invalid choice."
      pause
      ;;
  esac
  clear
done
