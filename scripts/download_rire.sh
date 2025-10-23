#!/bin/bash
# Download RIRE (Retrospective Image Registration Evaluation) dataset
# License: CC BY 3.0
# Source: https://rire.insight-journal.org/

set -e

RIRE_DIR="data/rire/raw"
mkdir -p "$RIRE_DIR"

echo "==========================================="
echo "RIRE Dataset Downloader"
echo "==========================================="
echo ""
echo "RIRE provides multi-modal brain imaging for registration benchmarking"
echo "Modalities: CT, MR-T1, MR-T2, MR-MPRAGE, PET"
echo "License: CC BY 3.0"
echo ""

# List of available patients (examples - check RIRE website for complete list)
PATIENTS=(
    "patient_001"
    "patient_002"
    "patient_003"
    "patient_004"
    "patient_005"
    "patient_101"
    "patient_102"
    "patient_103"
    "patient_104"
    "patient_105"
    "patient_109"
)

MODALITIES=(
    "ct"
    "mr_T1"
    "mr_T2"
    "mr_PD"
)

echo "Available patients: ${PATIENTS[@]}"
echo "Available modalities: ${MODALITIES[@]}"
echo ""

# Select subset to download (modify as needed)
DOWNLOAD_PATIENTS=("patient_101" "patient_102" "patient_103" "patient_104" "patient_105")
DOWNLOAD_MODALITIES=("ct" "mr_T1" "mr_T2")

echo "Downloading patients: ${DOWNLOAD_PATIENTS[@]}"
echo "Downloading modalities: ${DOWNLOAD_MODALITIES[@]}"
echo ""

BASE_URL="https://rire.insight-journal.org/download_data"

for patient in "${DOWNLOAD_PATIENTS[@]}"; do
    echo "Downloading $patient..."
    mkdir -p "$RIRE_DIR/$patient"

    for modality in "${DOWNLOAD_MODALITIES[@]}"; do
        url="${BASE_URL}/${patient}/${modality}.tar.gz"
        output="${RIRE_DIR}/${patient}/${modality}.tar.gz"

        echo "  - Fetching $modality from $url"

        # Try to download (may require authentication/updated URL)
        if wget -q --spider "$url" 2>/dev/null; then
            wget -c -O "$output" "$url"

            # Extract
            echo "    Extracting..."
            tar -xzf "$output" -C "$RIRE_DIR/$patient/"
            rm "$output"
        else
            echo "    WARNING: URL not accessible. You may need to:"
            echo "      1. Visit https://rire.insight-journal.org/download_data"
            echo "      2. Manually download $patient/$modality.tar.gz"
            echo "      3. Place in $RIRE_DIR/$patient/"
        fi
    done
    echo ""
done

echo "==========================================="
echo "RIRE download complete!"
echo "Data location: $RIRE_DIR"
echo ""
echo "IMPORTANT NOTES:"
echo "1. If download failed, manually download from:"
echo "   https://rire.insight-journal.org/download_data"
echo "2. RIRE data is in analyze/interfile format - convert to NIfTI"
echo "3. Registration transforms are provided separately"
echo ""
echo "Next steps:"
echo "  1. Convert to NIfTI: bash scripts/convert_dicom.sh"
echo "  2. Preprocess: bash scripts/register_ants.sh"
echo "==========================================="
