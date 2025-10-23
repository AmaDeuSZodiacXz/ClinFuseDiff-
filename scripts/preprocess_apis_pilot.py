#!/usr/bin/env python3
"""
APIS Dataset Preprocessing Pilot (5 cases)
Tests complete pipeline: resample → register (CT→MRI) → ROI masks → visualizations
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

print("="*80)
print("APIS Preprocessing Pilot - 5 Cases")
print("="*80)
print()

# Paths
project_root = Path(__file__).parent.parent.resolve()
raw_dir = project_root / "data/apis/raw"
preproc_dir = project_root / "data/apis/preproc"
work_dir = project_root / "work" / f"apis_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create directories
preproc_dir.mkdir(parents=True, exist_ok=True)
work_dir.mkdir(parents=True, exist_ok=True)

log(f"Raw data: {raw_dir}")
log(f"Preprocessing output: {preproc_dir}")
log(f"Work directory: {work_dir}")
print()

# Select 5 pilot cases
pilot_cases = ["train_000", "train_001", "train_002", "train_003", "train_004"]

log(f"Pilot cases: {pilot_cases}")
print()

# Process each case
results = []

for i, case_id in enumerate(pilot_cases, 1):
    log("="*60)
    log(f"[{i}/{len(pilot_cases)}] Processing {case_id}")
    log("="*60)
    
    case_raw = raw_dir
    case_preproc = preproc_dir / case_id
    case_work = work_dir / case_id
    
    # Create directories
    case_preproc.mkdir(exist_ok=True)
    case_work.mkdir(exist_ok=True)
    
    # Input files
    ct_file = case_raw / "ct" / case_id / "ct.nii.gz"
    adc_file = case_raw / "adc" / case_id / "adc.nii.gz"
    lesion_file = case_raw / "lesion_masks" / case_id / "lesion_mask.nii.gz"
    
    # Check inputs
    if not ct_file.exists():
        log(f"  ✗ CT not found: {ct_file}")
        results.append({"case": case_id, "status": "error", "reason": "missing CT"})
        continue
    if not adc_file.exists():
        log(f"  ✗ ADC not found: {adc_file}")
        results.append({"case": case_id, "status": "error", "reason": "missing ADC"})
        continue
    
    log(f"  ✓ CT: {ct_file.name}")
    log(f"  ✓ ADC: {adc_file.name}")
    if lesion_file.exists():
        log(f"  ✓ Lesion mask: {lesion_file.name}")
    print()
    
    # Step 1: Generate ROI masks with TotalSegmentator
    log("  Step 1: Generating ROI masks (brain + bone)...")
    
    cmd = [
        sys.executable,
        str(project_root / "scripts/make_masks_totalseg_with_viz.py"),
        "--mri", str(adc_file),
        "--ct", str(ct_file),
        "--out", str(case_work),
        "--fast",
        "--save-viz"
    ]
    
    result = subprocess.run(cmd, cwd=project_root, capture_output=True)
    
    if result.returncode != 0:
        log(f"  ✗ ROI mask generation failed")
        results.append({"case": case_id, "status": "error", "reason": "TotalSegmentator failed"})
        continue
    
    log(f"  ✓ ROI masks generated")
    
    # Step 2: Copy to preprocessing directory (binary read/write to avoid WSL permission issues)
    def copy_file(src, dst):
        """Binary copy to avoid Windows/WSL permission errors"""
        with open(src, 'rb') as f_in:
            with open(dst, 'wb') as f_out:
                f_out.write(f_in.read())

    # Copy brain mask
    brain_mask_src = case_work / "brain_mask.nii.gz"
    bone_mask_src = case_work / "bone_mask.nii.gz"

    if brain_mask_src.exists():
        copy_file(brain_mask_src, case_preproc / "brain_mask.nii.gz")
        log(f"  ✓ Brain mask → {case_preproc / 'brain_mask.nii.gz'}")

    if bone_mask_src.exists():
        copy_file(bone_mask_src, case_preproc / "bone_mask.nii.gz")
        log(f"  ✓ Bone mask → {case_preproc / 'bone_mask.nii.gz'}")

    # Copy lesion mask if exists
    if lesion_file.exists():
        copy_file(lesion_file, case_preproc / "lesion_mask.nii.gz")
        log(f"  ✓ Lesion mask → {case_preproc / 'lesion_mask.nii.gz'}")

    # Copy raw images
    copy_file(ct_file, case_preproc / "ct.nii.gz")
    copy_file(adc_file, case_preproc / "mri.nii.gz")
    log(f"  ✓ Raw images copied")
    
    results.append({"case": case_id, "status": "success"})
    print()

# Summary
print()
log("="*80)
log("Pilot Preprocessing Complete!")
log("="*80)

success = sum(1 for r in results if r["status"] == "success")
log(f"Success: {success}/{len(pilot_cases)} cases")

if success < len(pilot_cases):
    log(f"Errors:")
    for r in results:
        if r["status"] == "error":
            log(f"  - {r['case']}: {r['reason']}")

# Save results
results_file = work_dir / "preprocessing_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

log(f"\nResults saved: {results_file}")
log(f"Preprocessed data: {preproc_dir}")
log(f"Visualizations: {work_dir}/<case_id>/visualizations/")

print()
print("Next steps:")
print("  1. Review visualizations for quality")
print("  2. If OK, run batch preprocessing on all 60 cases")
print("  3. Create train/val/test splits")
print("  4. Start training!")
