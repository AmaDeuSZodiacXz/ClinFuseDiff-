#!/bin/bash
# Register CT to MRI/ADC using ANTs SyN
# Uses mutual information for multi-modal registration

set -e

# Default parameters
FIXED=""
MOVING=""
OUTPUT_PREFIX="work/reg/ct2mri_"
TRANSFORM_TYPE="s"  # s=SyN (deformable), a=affine only, r=rigid only
NUM_THREADS=8
QUICK_MODE=0

usage() {
    echo "Usage: $0 --fixed <mri.nii.gz> --moving <ct.nii.gz> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --fixed PATH      Fixed image (reference, e.g., MRI/ADC)"
    echo "  --moving PATH     Moving image (to be registered, e.g., CT)"
    echo ""
    echo "Optional:"
    echo "  --out PREFIX      Output prefix (default: work/reg/ct2mri_)"
    echo "  --type TYPE       Transform type: s=SyN, a=affine, r=rigid (default: s)"
    echo "  --threads N       Number of threads (default: 8)"
    echo "  --quick           Use quick mode (faster, less accurate)"
    echo "  --help            Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --fixed mri_adc.nii.gz --moving ct.nii.gz --out reg/case001_"
    echo ""
    echo "Output files:"
    echo "  {PREFIX}Warped.nii.gz           - Registered moving image"
    echo "  {PREFIX}0GenericAffine.mat      - Affine transform"
    echo "  {PREFIX}1Warp.nii.gz           - Deformation field (if SyN)"
    echo "  {PREFIX}1InverseWarp.nii.gz    - Inverse deformation (if SyN)"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fixed)
            FIXED="$2"
            shift 2
            ;;
        --moving)
            MOVING="$2"
            shift 2
            ;;
        --out)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --type)
            TRANSFORM_TYPE="$2"
            shift 2
            ;;
        --threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$FIXED" ] || [ -z "$MOVING" ]; then
    echo "ERROR: --fixed and --moving are required"
    usage
fi

if [ ! -f "$FIXED" ]; then
    echo "ERROR: Fixed image not found: $FIXED"
    exit 1
fi

if [ ! -f "$MOVING" ]; then
    echo "ERROR: Moving image not found: $MOVING"
    exit 1
fi

# Check if ANTs is installed
if ! command -v antsRegistration &> /dev/null; then
    echo "ERROR: ANTs not found. Please install ANTs:"
    echo "  Linux: Download from https://github.com/ANTsX/ANTs/releases"
    echo "  or use: pip install antspyx"
    exit 1
fi

# Create output directory
OUT_DIR=$(dirname "$OUTPUT_PREFIX")
mkdir -p "$OUT_DIR"

echo "==========================================="
echo "ANTs Registration"
echo "==========================================="
echo "Fixed (reference): $FIXED"
echo "Moving (to register): $MOVING"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Transform type: $TRANSFORM_TYPE"
echo "Threads: $NUM_THREADS"
echo "Quick mode: $QUICK_MODE"
echo ""

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NUM_THREADS

if [ $QUICK_MODE -eq 1 ]; then
    echo "Running quick registration (antsRegistrationSyNQuick.sh)..."
    antsRegistrationSyNQuick.sh \
        -d 3 \
        -f "$FIXED" \
        -m "$MOVING" \
        -o "$OUTPUT_PREFIX" \
        -t "$TRANSFORM_TYPE" \
        -n "$NUM_THREADS"
else
    echo "Running full registration (antsRegistrationSyN.sh)..."
    antsRegistrationSyN.sh \
        -d 3 \
        -f "$FIXED" \
        -m "$MOVING" \
        -o "$OUTPUT_PREFIX" \
        -t "$TRANSFORM_TYPE" \
        -n "$NUM_THREADS"
fi

echo ""
echo "==========================================="
echo "✓ Registration complete!"
echo "==========================================="
echo ""
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"*
echo ""
echo "Registered image: ${OUTPUT_PREFIX}Warped.nii.gz"
echo ""
echo "To apply transform to other images (e.g., segmentation masks):"
echo "  antsApplyTransforms -d 3 \\"
echo "    -i input.nii.gz \\"
echo "    -r $FIXED \\"
echo "    -o output.nii.gz \\"

if [ "$TRANSFORM_TYPE" == "s" ]; then
    echo "    -t ${OUTPUT_PREFIX}1Warp.nii.gz \\"
fi

echo "    -t ${OUTPUT_PREFIX}0GenericAffine.mat"
echo ""