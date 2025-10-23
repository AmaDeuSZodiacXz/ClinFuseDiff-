#!/bin/bash
# Copy lesion masks from raw to preprocessed directory

echo "Copying lesion masks from raw to preprocessed..."

count=0
for lesion_file in data/apis/raw/lesion_masks/*.nii.gz; do
    # Extract case ID (train_000.nii.gz -> train_000)
    case_id=$(basename "$lesion_file" .nii.gz)

    # Target directory
    case_dir="data/apis/preproc/$case_id"

    if [ ! -d "$case_dir" ]; then
        echo "⚠️  Case directory not found: $case_dir"
        continue
    fi

    # Copy file
    target="$case_dir/lesion_mask.nii.gz"
    cp "$lesion_file" "$target"

    if [ $? -eq 0 ]; then
        count=$((count + 1))
        echo "✓ Copied $case_id"
    else
        echo "❌ Failed to copy $case_id"
    fi
done

echo ""
echo "✅ Successfully copied $count lesion masks"

# Verify first 5
echo ""
echo "Verification (first 5 cases):"
ls -lh data/apis/preproc/train_00{0,1,2,3,4}/lesion_mask.nii.gz 2>&1 | head -5