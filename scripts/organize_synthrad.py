#!/usr/bin/env python3
"""
Organize SynthRAD2023 dataset into standardized ClinFuseDiff structure
"""
import shutil
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

print("="*80)
print("SynthRAD2023 Dataset Organization")
print("="*80)
print()

# Paths
project_root = Path('/mnt/c/Users/User/Documents/ClinFuseDiff')
extracted = project_root / 'SynthRAD_data/training/extracted/Task1/brain'
organized = project_root / 'data/synthrad/raw'

# Create organized directory
organized.mkdir(parents=True, exist_ok=True)

log(f"Source: {extracted}")
log(f"Target: {organized}")
print()

# Get all case directories
case_dirs = sorted([d for d in extracted.iterdir() if d.is_dir()])
total_cases = len(case_dirs)

log(f"Found {total_cases} cases")
print()

# Organize each case
success_count = 0
error_cases = []

for i, case_dir in enumerate(case_dirs, 1):
    case_id = case_dir.name
    
    # Check required files
    ct_file = case_dir / 'ct.nii.gz'
    mr_file = case_dir / 'mr.nii.gz'
    mask_file = case_dir / 'mask.nii.gz'
    
    if not ct_file.exists() or not mr_file.exists():
        log(f"  [{i}/{total_cases}] ✗ {case_id} - Missing files")
        error_cases.append(case_id)
        continue
    
    # Create target directory
    target_dir = organized / case_id
    target_dir.mkdir(exist_ok=True)
    
    # Copy files with binary read/write
    try:
        # CT
        with open(ct_file, 'rb') as f_in:
            with open(target_dir / 'ct.nii.gz', 'wb') as f_out:
                f_out.write(f_in.read())
        
        # MRI
        with open(mr_file, 'rb') as f_in:
            with open(target_dir / 'mri.nii.gz', 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Body mask (optional)
        if mask_file.exists():
            with open(mask_file, 'rb') as f_in:
                with open(target_dir / 'body_mask.nii.gz', 'wb') as f_out:
                    f_out.write(f_in.read())
        
        success_count += 1
        if i % 20 == 0 or i == total_cases:
            log(f"  Progress: {i}/{total_cases} ({100*i//total_cases}%)")
    
    except Exception as e:
        log(f"  [{i}/{total_cases}] ✗ {case_id} - Error: {e}")
        error_cases.append(case_id)

print()
log("="*80)
log(f"Organization complete!")
log(f"  Success: {success_count}/{total_cases} cases")
if error_cases:
    log(f"  Errors: {len(error_cases)} cases")
    for case_id in error_cases[:5]:
        log(f"    - {case_id}")
    if len(error_cases) > 5:
        log(f"    ... and {len(error_cases)-5} more")
log("="*80)
print()
log(f"Organized dataset: {organized}")
print()
print("Next steps:")
print("  1. Preprocess cases (resample, register if needed)")
print("  2. Generate ROI masks with TotalSegmentator")
print("  3. Create registration perturbation test set")
