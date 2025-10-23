#!/bin/bash
# Download RIRE dataset using wget
# For registration robustness testing in ClinFuseDiff++

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RIRE_DIR="data/rire/raw"
mkdir -p "$RIRE_DIR"

echo "======================================================================"
echo "RIRE Dataset Download (training_001)"
echo "======================================================================"
echo ""
echo "Output directory: $RIRE_DIR"
echo ""

# IPFS gateway base URL
IPFS_BASE="https://dweb.link/ipfs/bafybeih23xv6uamx7k27wk4uvzkxdtdryqeok22hpl3ybideggcjhipwme/rire"

# Files to download
declare -A FILES=(
    ["ct.tar.gz"]="CT image"
    ["mr_T1_rectified.tar.gz"]="MR T1 rectified"
    ["mr_T2_rectified.tar.gz"]="MR T2 rectified"
)

echo "Downloading RIRE training_001 dataset..."
echo ""

# Download each file
for file in "${!FILES[@]}"; do
    desc="${FILES[$file]}"
    url="${IPFS_BASE}/${file}"
    output="${RIRE_DIR}/${file}"

    echo "----------------------------------------------------------------------"
    echo "Downloading: $desc"
    echo "URL: $url"
    echo "Output: $output"
    echo ""

    if [ -f "$output" ]; then
        echo "  ✓ Already exists, skipping..."
    else
        wget -c -O "$output" "$url"
        if [ $? -eq 0 ]; then
            echo "  ✓ Downloaded: $(du -h "$output" | cut -f1)"
        else
            echo "  ✗ Download failed!"
            exit 1
        fi
    fi
    echo ""
done

echo "======================================================================"
echo "Extracting archives..."
echo "======================================================================"
echo ""

cd "$RIRE_DIR"

# Extract each tar.gz
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting: $file"
        tar -xzf "$file"
        echo "  ✓ Extracted"

        # Remove tar.gz after extraction
        # rm "$file"
        # echo "  ✓ Removed archive"
    fi
done

echo ""
echo "======================================================================"
echo "Organizing files..."
echo "======================================================================"
echo ""

# Create patient_001 directory
PATIENT_DIR="patient_001"
mkdir -p "$PATIENT_DIR"

# Move extracted files to patient directory
# (RIRE structure may vary, adjust as needed)
if [ -d "ct" ]; then
    mv ct/* "$PATIENT_DIR/" 2>/dev/null || true
    rmdir ct 2>/dev/null || true
fi

if [ -d "mr_T1_rectified" ]; then
    mv mr_T1_rectified/* "$PATIENT_DIR/" 2>/dev/null || true
    rmdir mr_T1_rectified 2>/dev/null || true
fi

if [ -d "mr_T2_rectified" ]; then
    mv mr_T2_rectified/* "$PATIENT_DIR/" 2>/dev/null || true
    rmdir mr_T2_rectified 2>/dev/null || true
fi

# Rename files for consistency
cd "$PATIENT_DIR"

# Find and rename CT files
if ls ct_* 1> /dev/null 2>&1; then
    for f in ct_*; do
        ext="${f##*.}"
        if [ "$ext" != "$f" ]; then
            mv "$f" "ct.$ext"
        fi
    done
fi

# Find and rename MR files
if ls mr_T1_* 1> /dev/null 2>&1; then
    for f in mr_T1_*; do
        ext="${f##*.}"
        if [ "$ext" != "$f" ]; then
            mv "$f" "mr_T1.$ext"
        fi
    done
fi

if ls mr_T2_* 1> /dev/null 2>&1; then
    for f in mr_T2_*; do
        ext="${f##*.}"
        if [ "$ext" != "$f" ]; then
            mv "$f" "mr_T2.$ext"
        fi
    done
fi

cd "$REPO_ROOT"

echo ""
echo "======================================================================"
echo "RIRE Dataset Download Complete!"
echo "======================================================================"
echo ""
echo "Downloaded files in: $RIRE_DIR/$PATIENT_DIR/"
ls -lh "$RIRE_DIR/$PATIENT_DIR/"
echo ""
echo "File formats: ANALYZE (.hdr + .img)"
echo ""
echo "Next step: Convert to NIfTI format"
echo "  python scripts/convert_rire_to_nifti.py"
echo ""
echo "======================================================================"