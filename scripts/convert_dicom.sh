#!/bin/bash
# Convert DICOM to NIfTI using dcm2niix
# Supports batch conversion with proper orientation handling

set -e

show_usage() {
    echo "Usage: $0 [OPTIONS] <input_dir> [output_dir]"
    echo ""
    echo "Convert DICOM series to NIfTI format"
    echo ""
    echo "Arguments:"
    echo "  input_dir     Directory containing DICOM files"
    echo "  output_dir    Output directory (default: same as input)"
    echo ""
    echo "Options:"
    echo "  -c            Compress output (gzip)"
    echo "  -s            Single file mode (don't merge series)"
    echo "  -b            BIDS sidecar (generate JSON metadata)"
    echo "  -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Convert single case"
    echo "  $0 data/apis/raw/ct/case_001"
    echo ""
    echo "  # Convert with compression and BIDS metadata"
    echo "  $0 -c -b data/apis/raw/ct/case_001 data/apis/preproc/case_001"
    echo ""
    echo "  # Batch convert all cases"
    echo "  for d in data/apis/raw/ct/*/; do $0 \$d; done"
}

# Parse options
COMPRESS="-z y"
BIDS=""
SINGLE=""

while getopts "csbh" opt; do
    case $opt in
        c) COMPRESS="-z y" ;;
        s) SINGLE="-s y" ;;
        b) BIDS="-b y" ;;
        h) show_usage; exit 0 ;;
        *) show_usage; exit 1 ;;
    esac
done

shift $((OPTIND-1))

# Check arguments
if [ $# -lt 1 ]; then
    echo "Error: Missing input directory"
    show_usage
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-$INPUT_DIR}"

# Verify dcm2niix is installed
if ! command -v dcm2niix &> /dev/null; then
    echo "Error: dcm2niix not found"
    echo ""
    echo "Install instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install dcm2niix"
    echo "  macOS: brew install dcm2niix"
    echo "  conda: conda install -c conda-forge dcm2niix"
    exit 1
fi

# Verify input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "====================================="
echo "DICOM to NIfTI Conversion"
echo "====================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run dcm2niix
# -f: filename format (%p = protocol, %s = series, %i = ID)
# -o: output directory
# -z: compress (y/n)
# -b: BIDS sidecar (y/n)
# -s: single file per series (y/n)

dcm2niix $COMPRESS $BIDS $SINGLE \
    -f "%p_%s" \
    -o "$OUTPUT_DIR" \
    "$INPUT_DIR"

echo ""
echo "====================================="
echo "Conversion Complete!"
echo "====================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.nii* 2>/dev/null || echo "  (check $OUTPUT_DIR for output)"
echo ""
echo "Next steps:"
echo "  1. Verify orientation (qform/sform) with FSLeyes or ITK-SNAP"
echo "  2. Run registration: bash scripts/register_ants.sh"
echo ""