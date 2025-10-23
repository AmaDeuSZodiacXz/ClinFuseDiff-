#!/bin/bash
# Complete ClinFuseDiff++ Workflow with Comprehensive Logging
# Automates: Data Download → Preprocessing → Training → Evaluation
# Saves all outputs, visualizations, and logs systematically

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Configuration
DATASET="${1:-apis}"
RUN_NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"

# Directory structure
WORK_DIR="work/${RUN_NAME}"
LOG_DIR="${WORK_DIR}/logs"
VIZ_DIR="${WORK_DIR}/visualizations"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
RESULTS_DIR="${WORK_DIR}/results"

# Create directories
mkdir -p "$LOG_DIR" "$VIZ_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" | tee -a "${LOG_DIR}/workflow.log"
}

log_section() {
    local section="$1"
    log ""
    log "=========================================="
    log "$section"
    log "=========================================="
}

# Error handler
handle_error() {
    log "ERROR: Workflow failed at line $1"
    log "Check ${LOG_DIR}/workflow.log for details"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Start workflow
log_section "ClinFuseDiff++ Complete Workflow"
log "Run name: ${RUN_NAME}"
log "Dataset: ${DATASET}"
log "Working directory: ${WORK_DIR}"
log ""

# ============================================================================
# STEP 1: Environment Verification
# ============================================================================
log_section "STEP 1: Environment Verification"

log "Checking conda environment..."
if conda env list | grep -q "clinfusediff"; then
    log "✓ clinfusediff environment found"
else
    log "ERROR: clinfusediff environment not found"
    log "Please run: bash workflow/01_setup_environment.sh"
    exit 1
fi

log "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clinfusediff

log "Verifying dependencies..."
python -c "import torch, monai, nibabel, ants; print('✓ All imports successful')" 2>&1 | tee -a "${LOG_DIR}/workflow.log"

log "Python version: $(python --version)"
log "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log "MONAI version: $(python -c 'import monai; print(monai.__version__)')"

# ============================================================================
# STEP 2: Data Download Instructions
# ============================================================================
log_section "STEP 2: Data Download Status"

DATA_DIR="data/${DATASET}"

if [ ! -d "${DATA_DIR}/raw/ct" ] || [ ! -d "${DATA_DIR}/raw/adc" ]; then
    log "WARNING: APIS dataset not found at ${DATA_DIR}/raw"
    log ""
    log "Please download APIS dataset manually:"
    log "  1. Visit: https://bivl2ab.uis.edu.co/challenges/apis"
    log "  2. Register and accept data usage agreement"
    log "  3. Download CT, ADC/MRI, and lesion masks"
    log "  4. Extract to: ${DATA_DIR}/raw/"
    log ""
    log "Expected structure:"
    log "  ${DATA_DIR}/raw/"
    log "    ├── ct/case_*/ct.nii.gz"
    log "    ├── adc/case_*/adc.nii.gz"
    log "    └── lesion_masks/case_*.nii.gz"
    log ""

    read -p "Press Enter after downloading data, or Ctrl+C to exit..."
fi

# Count available cases
CT_COUNT=$(find "${DATA_DIR}/raw/ct" -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)
ADC_COUNT=$(find "${DATA_DIR}/raw/adc" -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)

log "Found ${CT_COUNT} CT volumes"
log "Found ${ADC_COUNT} ADC/MRI volumes"

if [ "$CT_COUNT" -eq 0 ] || [ "$ADC_COUNT" -eq 0 ]; then
    log "ERROR: No data found. Please download APIS dataset first."
    exit 1
fi

# ============================================================================
# STEP 3: Preprocessing with Comprehensive Logging
# ============================================================================
log_section "STEP 3: Preprocessing Pipeline"

PREPROC_LOG="${LOG_DIR}/preprocessing.log"
PREPROC_DIR="${DATA_DIR}/preproc"

log "Preprocessing output: ${PREPROC_DIR}"
log "Preprocessing log: ${PREPROC_LOG}"

# Find all cases
CASES=()
for case_dir in "${DATA_DIR}/raw/ct"/*; do
    if [ -d "$case_dir" ]; then
        case_id=$(basename "$case_dir")
        CASES+=("$case_id")
    fi
done

log "Processing ${#CASES[@]} cases..."

for case_id in "${CASES[@]}"; do
    log ""
    log "Processing: ${case_id}"

    CT_PATH="${DATA_DIR}/raw/ct/${case_id}"
    ADC_PATH="${DATA_DIR}/raw/adc/${case_id}"
    OUT_DIR="${PREPROC_DIR}/${case_id}"
    MASK_DIR="${WORK_DIR}/masks/${case_id}"
    REG_DIR="${WORK_DIR}/reg/${case_id}"

    mkdir -p "$OUT_DIR" "$MASK_DIR" "$REG_DIR"

    # Find CT and ADC files
    CT_FILE=$(find "$CT_PATH" -name "*.nii.gz" -o -name "*.nii" | head -n 1)
    ADC_FILE=$(find "$ADC_PATH" -name "*.nii.gz" -o -name "*.nii" | head -n 1)

    if [ -z "$CT_FILE" ] || [ -z "$ADC_FILE" ]; then
        log "  WARNING: Missing files for ${case_id}, skipping..."
        continue
    fi

    log "  CT: $CT_FILE"
    log "  ADC: $ADC_FILE"

    # Registration
    if [ ! -f "${REG_DIR}/ct_in_mri.nii.gz" ]; then
        log "  [1/3] Registration (CT → MRI)..."
        bash scripts/register_ants.sh \
            --fixed "$ADC_FILE" \
            --moving "$CT_FILE" \
            --out "${REG_DIR}/ct2mri_" \
            --type s \
            --threads 8 \
            2>&1 | tee -a "$PREPROC_LOG"

        mv "${REG_DIR}/ct2mri_Warped.nii.gz" "${REG_DIR}/ct_in_mri.nii.gz"
        log "  ✓ Registration complete"
    else
        log "  ✓ Registration already done"
    fi

    # ROI mask generation with visualization
    if [ ! -f "${MASK_DIR}/brain_mask.nii.gz" ] || [ ! -f "${MASK_DIR}/bone_mask.nii.gz" ]; then
        log "  [2/3] Generating ROI masks with TotalSegmentator..."
        python scripts/make_masks_totalseg_with_viz.py \
            --mri "$ADC_FILE" \
            --ct "${REG_DIR}/ct_in_mri.nii.gz" \
            --out "$MASK_DIR" \
            --fast \
            --save-viz \
            2>&1 | tee -a "$PREPROC_LOG"
        log "  ✓ ROI masks generated"
    else
        log "  ✓ ROI masks already exist"
    fi

    # Copy to final preprocessing directory
    log "  [3/3] Finalizing preprocessed data..."
    cp "$ADC_FILE" "${OUT_DIR}/mri.nii.gz"
    cp "${REG_DIR}/ct_in_mri.nii.gz" "${OUT_DIR}/ct.nii.gz"
    cp "${MASK_DIR}/brain_mask.nii.gz" "${OUT_DIR}/brain_mask.nii.gz"
    cp "${MASK_DIR}/bone_mask.nii.gz" "${OUT_DIR}/bone_mask.nii.gz"

    # Copy lesion mask if available
    if [ -f "${DATA_DIR}/raw/lesion_masks/${case_id}.nii.gz" ]; then
        cp "${DATA_DIR}/raw/lesion_masks/${case_id}.nii.gz" "${OUT_DIR}/lesion_mask.nii.gz"
        log "  ✓ Lesion mask copied"
    fi

    # Copy visualizations
    if [ -d "${MASK_DIR}/visualizations" ]; then
        cp -r "${MASK_DIR}/visualizations" "${VIZ_DIR}/${case_id}_masks"
        log "  ✓ Visualizations saved to ${VIZ_DIR}/${case_id}_masks"
    fi

    log "  ✓ ${case_id} preprocessing complete"
done

log ""
log "✓ Preprocessing completed for ${#CASES[@]} cases"

# ============================================================================
# STEP 4: Create Data Splits
# ============================================================================
log_section "STEP 4: Creating Data Splits"

SPLITS_DIR="${DATA_DIR}/splits"

python scripts/make_splits.py \
    --data_dir "$PREPROC_DIR" \
    --output "$SPLITS_DIR" \
    --ratios 0.7 0.15 0.15 \
    --seed 42 \
    2>&1 | tee -a "${LOG_DIR}/splits.log"

log "✓ Data splits created:"
log "  Train: $(cat ${SPLITS_DIR}/train.txt | wc -l) cases"
log "  Val:   $(cat ${SPLITS_DIR}/val.txt | wc -l) cases"
log "  Test:  $(cat ${SPLITS_DIR}/test.txt | wc -l) cases"

# ============================================================================
# STEP 5: Training with Detailed Logging
# ============================================================================
log_section "STEP 5: Training"

TRAIN_LOG="${LOG_DIR}/training.log"

log "Starting training..."
log "Config: configs/cvpr2026/train_roi.yaml"
log "Preset: stroke"
log "Output: ${CHECKPOINT_DIR}"
log "Training log: ${TRAIN_LOG}"

python train.py \
    --config configs/cvpr2026/train_roi.yaml \
    --preset stroke \
    --data_dir "$DATA_DIR" \
    --output_dir "$CHECKPOINT_DIR" \
    --save_viz \
    2>&1 | tee "$TRAIN_LOG"

log "✓ Training complete"
log "Best checkpoint: ${CHECKPOINT_DIR}/best.ckpt"

# ============================================================================
# STEP 6: Evaluation with Comprehensive Output
# ============================================================================
log_section "STEP 6: Evaluation"

EVAL_LOG="${LOG_DIR}/evaluation.log"

log "Running comprehensive evaluation..."
log "Checkpoint: ${CHECKPOINT_DIR}/best.ckpt"
log "Evaluation log: ${EVAL_LOG}"

python evaluate.py \
    --config configs/cvpr2026/train_roi.yaml \
    --checkpoint "${CHECKPOINT_DIR}/best.ckpt" \
    --split test \
    --num_samples 8 \
    --save_images \
    --save_uncertainty \
    --registration_stress \
    --warp_range 1.0 3.0 \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$EVAL_LOG"

log "✓ Evaluation complete"
log "Results saved to: ${RESULTS_DIR}"

# ============================================================================
# STEP 7: Generate Summary Report
# ============================================================================
log_section "STEP 7: Summary Report"

SUMMARY_FILE="${WORK_DIR}/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# ClinFuseDiff++ Workflow Summary

**Run Name**: ${RUN_NAME}
**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Dataset**: ${DATASET}

## Pipeline Steps

1. ✅ Environment Verification
2. ✅ Data Download (${CT_COUNT} cases)
3. ✅ Preprocessing (registration + ROI masks)
4. ✅ Data Splits (train/val/test)
5. ✅ Training
6. ✅ Evaluation

## Output Structure

\`\`\`
${WORK_DIR}/
├── logs/
│   ├── workflow.log              # Main workflow log
│   ├── preprocessing.log         # Preprocessing details
│   ├── training.log              # Training output
│   └── evaluation.log            # Evaluation output
├── visualizations/
│   └── case_*/                   # ROI mask visualizations
├── checkpoints/
│   ├── best.ckpt                 # Best model checkpoint
│   └── last.ckpt                 # Latest checkpoint
└── results/
    ├── metrics_per_case.csv      # Per-case metrics
    ├── metrics_aggregate.json    # Summary statistics
    ├── fused_images/             # Fused NIfTI volumes
    └── uncertainty/              # Uncertainty maps
\`\`\`

## Key Metrics

EOF

# Extract metrics from evaluation results
if [ -f "${RESULTS_DIR}/metrics_aggregate.json" ]; then
    log "Extracting metrics..."
    python -c "
import json
with open('${RESULTS_DIR}/metrics_aggregate.json', 'r') as f:
    metrics = json.load(f)
    print('\n## Aggregate Metrics\n')
    for key, value in metrics.items():
        print(f'- **{key}**: {value:.4f}')
" >> "$SUMMARY_FILE"
fi

log "✓ Summary report generated: ${SUMMARY_FILE}"

# ============================================================================
# WORKFLOW COMPLETE
# ============================================================================
log_section "WORKFLOW COMPLETE"

log "All steps completed successfully!"
log ""
log "Output directory: ${WORK_DIR}"
log "Summary report: ${SUMMARY_FILE}"
log ""
log "Quick access:"
log "  Logs:            ${LOG_DIR}/"
log "  Visualizations:  ${VIZ_DIR}/"
log "  Checkpoints:     ${CHECKPOINT_DIR}/"
log "  Results:         ${RESULTS_DIR}/"
log ""
log "Next steps:"
log "  1. Review summary: cat ${SUMMARY_FILE}"
log "  2. Check metrics: cat ${RESULTS_DIR}/metrics_aggregate.json"
log "  3. View logs: tail ${LOG_DIR}/workflow.log"
log ""

echo ""
echo "✅ ClinFuseDiff++ workflow completed successfully!"
echo "📊 Check ${SUMMARY_FILE} for full report"