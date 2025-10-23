#!/bin/bash
# Pre-flight Check for ClinFuseDiff++ Workflow
# Verifies all dependencies, scripts, and configurations are ready

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==========================================="
echo "ClinFuseDiff++ Pre-Flight Check"
echo "==========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

PASSED=0
FAILED=0
WARNINGS=0

# ============================================================================
# 1. Environment Check
# ============================================================================
echo "1. Checking conda environment..."

if conda env list | grep -q "clinfusediff"; then
    check_pass "clinfusediff environment exists"
    ((PASSED++))
else
    check_fail "clinfusediff environment not found"
    ((FAILED++))
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clinfusediff 2>/dev/null || {
    check_fail "Cannot activate clinfusediff environment"
    ((FAILED++))
    exit 1
}

# ============================================================================
# 2. Python Dependencies Check
# ============================================================================
echo ""
echo "2. Checking Python dependencies..."

# Check PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    check_pass "PyTorch ($TORCH_VERSION)"
    ((PASSED++))
else
    check_fail "PyTorch not found"
    ((FAILED++))
fi

# Check MONAI
if python -c "import monai" 2>/dev/null; then
    MONAI_VERSION=$(python -c "import monai; print(monai.__version__)")
    check_pass "MONAI ($MONAI_VERSION)"
    ((PASSED++))
else
    check_fail "MONAI not found"
    ((FAILED++))
fi

# Check ANTsPy
if python -c "import ants" 2>/dev/null; then
    check_pass "ANTsPy"
    ((PASSED++))
else
    check_fail "ANTsPy not found"
    ((FAILED++))
fi

# Check NiBabel
if python -c "import nibabel" 2>/dev/null; then
    check_pass "NiBabel"
    ((PASSED++))
else
    check_fail "NiBabel not found"
    ((FAILED++))
fi

# Check TotalSegmentator
if command -v TotalSegmentator &>/dev/null; then
    check_pass "TotalSegmentator CLI"
    ((PASSED++))
else
    check_fail "TotalSegmentator CLI not found"
    ((FAILED++))
fi

# ============================================================================
# 3. Directory Structure Check
# ============================================================================
echo ""
echo "3. Checking directory structure..."

required_dirs=(
    "data/apis/raw/ct"
    "data/apis/raw/adc"
    "data/apis/raw/lesion_masks"
    "data/apis/preproc"
    "data/apis/splits"
    "work/experiments"
    "configs/cvpr2026"
    "scripts"
    "workflow"
    "src/models"
    "src/training"
    "src/data"
    "src/utils"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory: $dir"
        ((PASSED++))
    else
        check_warn "Directory missing: $dir (will be created)"
        mkdir -p "$dir"
        ((WARNINGS++))
    fi
done

# ============================================================================
# 4. Essential Scripts Check
# ============================================================================
echo ""
echo "4. Checking essential scripts..."

essential_scripts=(
    "scripts/make_masks_totalseg_with_viz.py"
    "scripts/make_splits.py"
    "scripts/register_ants.sh"
    "workflow/complete_workflow_with_logging.sh"
    "train.py"
    "evaluate.py"
)

for script in "${essential_scripts[@]}"; do
    if [ -f "$script" ]; then
        check_pass "Script: $script"
        ((PASSED++))
    else
        check_fail "Script missing: $script"
        ((FAILED++))
    fi
done

# ============================================================================
# 5. Configuration Files Check
# ============================================================================
echo ""
echo "5. Checking configuration files..."

if [ -f "configs/cvpr2026/train_roi.yaml" ]; then
    check_pass "Config: configs/cvpr2026/train_roi.yaml"
    ((PASSED++))
else
    check_fail "Config missing: configs/cvpr2026/train_roi.yaml"
    ((FAILED++))
fi

# ============================================================================
# 6. Dataset Check
# ============================================================================
echo ""
echo "6. Checking APIS dataset..."

CT_COUNT=$(find data/apis/raw/ct -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)
ADC_COUNT=$(find data/apis/raw/adc -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)
MASK_COUNT=$(find data/apis/raw/lesion_masks -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)

if [ "$CT_COUNT" -gt 0 ]; then
    check_pass "CT volumes: $CT_COUNT found"
    ((PASSED++))
else
    check_warn "CT volumes: 0 found (dataset not downloaded yet)"
    ((WARNINGS++))
fi

if [ "$ADC_COUNT" -gt 0 ]; then
    check_pass "ADC/MRI volumes: $ADC_COUNT found"
    ((PASSED++))
else
    check_warn "ADC/MRI volumes: 0 found (dataset not downloaded yet)"
    ((WARNINGS++))
fi

if [ "$MASK_COUNT" -gt 0 ]; then
    check_pass "Lesion masks: $MASK_COUNT found"
    ((PASSED++))
else
    check_warn "Lesion masks: 0 found (dataset not downloaded yet)"
    ((WARNINGS++))
fi

# ============================================================================
# 7. GPU Check (optional)
# ============================================================================
echo ""
echo "7. Checking GPU availability (optional)..."

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    check_pass "GPU available: $GPU_NAME"
    ((PASSED++))
else
    check_warn "No GPU detected (training will use CPU - slower)"
    ((WARNINGS++))
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "==========================================="
echo "Pre-Flight Check Summary"
echo "==========================================="
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC}   $FAILED"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""

    if [ "$CT_COUNT" -eq 0 ] || [ "$ADC_COUNT" -eq 0 ]; then
        echo "⚠ APIS dataset not found."
        echo ""
        echo "Next step: Download APIS dataset"
        echo "  1. Visit: https://bivl2ab.uis.edu.co/challenges/apis"
        echo "  2. Register and accept data usage agreement"
        echo "  3. Download CT, ADC/MRI, and lesion mask archives"
        echo "  4. Extract to: data/apis/raw/{ct,adc,lesion_masks}"
        echo ""
        echo "See DATASET_DOWNLOAD_GUIDE.md for detailed instructions."
    else
        echo "✓ Dataset found: $CT_COUNT cases"
        echo ""
        echo "🚀 Ready to start workflow!"
        echo ""
        echo "Run:"
        echo "  bash workflow/complete_workflow_with_logging.sh apis run_\$(date +%Y%m%d_%H%M%S)"
    fi
else
    echo -e "${RED}✗ Some critical checks failed.${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    exit 1
fi

echo ""
echo "==========================================="