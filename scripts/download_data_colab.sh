#!/bin/bash
# Download APIS dataset from Hugging Face Hub to Colab

set -e

# Configuration
REPO_ID="Pakawat-Phasook/ClinFuseDiff-APIS-Data"
TARGET_DIR="/content/ClinFuseDiff-/data/apis"

echo "================================================================================"
echo "DOWNLOAD APIS DATASET FROM HUGGING FACE HUB"
echo "================================================================================"
echo ""
echo "Repository: ${REPO_ID}"
echo "Target directory: ${TARGET_DIR}"
echo ""

# Create target directory
mkdir -p "${TARGET_DIR}"

# Download using huggingface-cli
echo "Downloading dataset..."
huggingface-cli download \
    "${REPO_ID}" \
    --repo-type dataset \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False

echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE!"
echo "================================================================================"
echo ""
echo "Dataset location: ${TARGET_DIR}"
echo ""
echo "Structure:"
ls -lh "${TARGET_DIR}"
echo ""
echo "Verifying preproc cases:"
ls "${TARGET_DIR}/preproc" | wc -l
echo "cases found"
echo ""
echo "Splits:"
wc -l "${TARGET_DIR}/splits"/*.txt