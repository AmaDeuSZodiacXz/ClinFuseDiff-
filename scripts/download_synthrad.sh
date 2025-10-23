#!/bin/bash
# Download SynthRAD2023 dataset (brain subset)
# 540 brain cases with paired CT-MRI from multiple centers
# Source: https://zenodo.org/records/7260705
# Paper: https://doi.org/10.1002/mp.16529

set -e

SYNTHRAD_DIR="${1:-data/synthrad2023/raw}"
mkdir -p "$SYNTHRAD_DIR"

echo "====================================="
echo "SynthRAD2023 Dataset Downloader"
echo "====================================="
echo ""
echo "Multi-center CT-MRI paired dataset for radiation therapy"
echo "Brain subset: 540 paired cases"
echo ""
echo "Output directory: $SYNTHRAD_DIR"
echo ""

# Zenodo record files
# NOTE: Update these URLs based on the actual Zenodo record structure
# Visit https://zenodo.org/records/7260705 for the latest file list

ZENODO_RECORD="7260705"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

# Expected files (adjust based on actual dataset structure)
FILES=(
    "brain-train.tar.gz"
    "brain-val.tar.gz"
    "brain-test.tar.gz"
)

echo "Downloading from Zenodo record: $ZENODO_RECORD"
echo ""

# Function to download file with progress
download_file() {
    local filename=$1
    local url="${BASE_URL}/${filename}"
    local output="$SYNTHRAD_DIR/$filename"

    echo "Downloading: $filename"

    if wget --show-progress -O "$output" "$url"; then
        echo "✓ Downloaded $filename"

        # Extract if it's a tar.gz
        if [[ "$filename" == *.tar.gz ]]; then
            echo "  Extracting..."
            tar -xzf "$output" -C "$SYNTHRAD_DIR/" || {
                echo "  Warning: Failed to extract $filename"
                return 1
            }
            # Optionally remove tar file to save space
            # rm -f "$output"
            echo "  ✓ Extracted $filename"
        fi

        return 0
    else
        echo "✗ Failed to download $filename"
        echo ""
        echo "NOTE: You may need to manually download from:"
        echo "  https://zenodo.org/records/$ZENODO_RECORD"
        echo ""
        echo "Or check if the file structure has changed."
        return 1
    fi
}

# Alternative: Manual download instructions
show_manual_instructions() {
    echo "====================================="
    echo "Manual Download Required"
    echo "====================================="
    echo ""
    echo "Please follow these steps:"
    echo ""
    echo "1. Visit: https://zenodo.org/records/7260705"
    echo "2. Download the brain subset files (train/val/test)"
    echo "3. Extract to: $SYNTHRAD_DIR"
    echo ""
    echo "Expected directory structure:"
    echo "  $SYNTHRAD_DIR/"
    echo "    ├── train/"
    echo "    │   ├── <center_id>/"
    echo "    │   │   ├── <patient_id>/"
    echo "    │   │   │   ├── ct.nii.gz"
    echo "    │   │   │   └── mri.nii.gz"
    echo "    ├── val/"
    echo "    └── test/"
    echo ""
}

# Try automated download
echo "Attempting automated download..."
echo ""

success=0
for file in "${FILES[@]}"; do
    if download_file "$file"; then
        ((success++))
    fi
done

if [ $success -eq 0 ]; then
    echo ""
    show_manual_instructions
    exit 1
fi

echo ""
echo "====================================="
echo "Download Complete!"
echo "====================================="
echo ""
echo "Downloaded $success/${#FILES[@]} files"
echo "Dataset location: $SYNTHRAD_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify data integrity"
echo "  2. Check file structure matches expected format"
echo "  3. Run preprocessing pipeline"
echo ""
echo "For registration-aware robustness testing:"
echo "  bash scripts/register_ants.sh --fixed <mri> --moving <ct>"
echo ""