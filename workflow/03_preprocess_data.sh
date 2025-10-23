#!/bin/bash
# Preprocess data: Registration + ROI mask generation
# Part of CLIN-FuseDiff++ workflow

set -e

DATASET="${1:-apis}"
DATA_DIR="${2:-data/$DATASET}"

echo "========================================="
echo "Data Preprocessing Pipeline"
echo "========================================="
echo "Dataset: $DATASET"
echo "Data directory: $DATA_DIR"
echo ""

# Check if data exists
if [ ! -d "$DATA_DIR/raw" ]; then
    echo "Error: Raw data not found at $DATA_DIR/raw"
    echo ""
    echo "Please download data first:"
    echo "  bash scripts/download_${DATASET}.sh"
    exit 1
fi

# Create output directories
mkdir -p "$DATA_DIR/preproc"
mkdir -p work/masks
mkdir -p work/reg
mkdir -p work/qc

echo "Step 1/4: Finding cases..."
echo ""

# Find all cases (assuming directory structure: raw/{ct,adc}/<case_id>/)
CASES=()
if [ -d "$DATA_DIR/raw/ct" ]; then
    for case_dir in "$DATA_DIR/raw/ct"/*; do
        if [ -d "$case_dir" ]; then
            case_id=$(basename "$case_dir")
            CASES+=("$case_id")
        fi
    done
fi

if [ ${#CASES[@]} -eq 0 ]; then
    echo "Error: No cases found in $DATA_DIR/raw/ct/"
    exit 1
fi

echo "Found ${#CASES[@]} cases to process"
echo ""

# Process each case
for case_id in "${CASES[@]}"; do
    echo "========================================="
    echo "Processing: $case_id"
    echo "========================================="

    CT_PATH="$DATA_DIR/raw/ct/$case_id"
    ADC_PATH="$DATA_DIR/raw/adc/$case_id"
    OUT_DIR="$DATA_DIR/preproc/$case_id"
    MASK_DIR="work/masks/$case_id"
    REG_DIR="work/reg/$case_id"

    mkdir -p "$OUT_DIR"
    mkdir -p "$MASK_DIR"
    mkdir -p "$REG_DIR"

    # Find CT and ADC files
    CT_FILE=$(find "$CT_PATH" -name "*.nii.gz" -o -name "*.nii" | head -n 1)
    ADC_FILE=$(find "$ADC_PATH" -name "*.nii.gz" -o -name "*.nii" | head -n 1)

    if [ -z "$CT_FILE" ] || [ -z "$ADC_FILE" ]; then
        echo "  Warning: Missing CT or ADC file for $case_id, skipping..."
        continue
    fi

    echo "  CT:  $CT_FILE"
    echo "  ADC: $ADC_FILE"
    echo ""

    # Step 2: Register CT to ADC/MRI
    echo "Step 2/4: Registration (CT → MRI)..."

    if [ ! -f "$REG_DIR/ct_in_mri.nii.gz" ]; then
        bash scripts/register_ants.sh \
            --fixed "$ADC_FILE" \
            --moving "$CT_FILE" \
            --out "$REG_DIR/ct2mri_" \
            --type s \
            --threads 8

        # Rename warped image
        mv "$REG_DIR/ct2mri_Warped.nii.gz" "$REG_DIR/ct_in_mri.nii.gz"
    else
        echo "  ✓ Registration already done"
    fi
    echo ""

    # Step 3: Generate ROI masks with TotalSegmentator
    echo "Step 3/4: Generating ROI masks..."

    if [ ! -f "$MASK_DIR/brain_mask.nii.gz" ] || [ ! -f "$MASK_DIR/bone_mask.nii.gz" ]; then
        python scripts/make_masks_totalseg.py \
            --mri "$ADC_FILE" \
            --ct "$REG_DIR/ct_in_mri.nii.gz" \
            --out "$MASK_DIR" \
            --fast
    else
        echo "  ✓ ROI masks already generated"
    fi
    echo ""

    # Step 4: Copy/resample to final preprocessing directory
    echo "Step 4/4: Finalizing preprocessed data..."

    # Copy ADC (fixed image)
    cp "$ADC_FILE" "$OUT_DIR/mri.nii.gz"

    # Copy registered CT
    cp "$REG_DIR/ct_in_mri.nii.gz" "$OUT_DIR/ct.nii.gz"

    # Copy ROI masks
    cp "$MASK_DIR/brain_mask.nii.gz" "$OUT_DIR/brain_mask.nii.gz"
    cp "$MASK_DIR/bone_mask.nii.gz" "$OUT_DIR/bone_mask.nii.gz"

    # Copy lesion mask if available
    if [ -f "$DATA_DIR/raw/lesion_masks/$case_id.nii.gz" ]; then
        cp "$DATA_DIR/raw/lesion_masks/$case_id.nii.gz" "$OUT_DIR/lesion_mask.nii.gz"
    fi

    echo "  ✓ Preprocessed data saved to $OUT_DIR"
    echo ""

    echo "========================================="
    echo "✓ $case_id complete!"
    echo "========================================="
    echo ""
done

echo "========================================="
echo "✓ All preprocessing complete!"
echo "========================================="
echo ""
echo "Processed ${#CASES[@]} cases"
echo "Output: $DATA_DIR/preproc/"
echo ""
echo "Next steps:"
echo "  1. Verify preprocessed data:"
echo "     ls -lh $DATA_DIR/preproc/*/"
echo ""
echo "  2. Create data splits:"
echo "     python scripts/make_splits.py --data_dir $DATA_DIR/preproc"
echo ""
echo "  3. Start training:"
echo "     bash workflow/06_start_training.sh"
echo ""