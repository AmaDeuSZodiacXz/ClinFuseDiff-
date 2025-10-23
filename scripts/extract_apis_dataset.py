#!/usr/bin/env python3
"""
Extract and organize APIS dataset from zip file
Organizes into the expected directory structure:
  data/apis/raw/ct/
  data/apis/raw/adc/
  data/apis/raw/lesion_masks/
"""

import zipfile
import shutil
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("APIS Dataset Extraction and Organization")
    print("=" * 60)
    print()

    # Paths
    repo_root = Path(__file__).parent.parent
    zip_path = repo_root / "APIS_dataset.zip"
    extract_temp = repo_root / "data" / "apis" / "raw_temp"

    ct_dir = repo_root / "data" / "apis" / "raw" / "ct"
    adc_dir = repo_root / "data" / "apis" / "raw" / "adc"
    mask_dir = repo_root / "data" / "apis" / "raw" / "lesion_masks"

    # Check zip exists
    if not zip_path.exists():
        print(f"✗ APIS_dataset.zip not found at {zip_path}")
        return 1

    print(f"✓ Found: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # Create directories
    extract_temp.mkdir(parents=True, exist_ok=True)
    ct_dir.mkdir(parents=True, exist_ok=True)
    adc_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip
    print("1. Extracting APIS_dataset.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        total_files = len(zf.namelist())
        print(f"   Total files in archive: {total_files}")
        zf.extractall(extract_temp)
    print("   ✓ Extraction complete")
    print()

    # Organize files
    print("2. Organizing files into proper structure...")

    # Find all case directories
    case_dirs = sorted([d for d in extract_temp.iterdir() if d.is_dir() and d.name.startswith("train_")])

    ct_count = 0
    adc_count = 0
    mask_count = 0

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Find files
        ncct_file = case_dir / f"{case_id}_ncct.nii.gz"
        adc_file = case_dir / f"{case_id}_adc.nii.gz"
        mask_file = case_dir / "masks" / f"{case_id}_r1_mask.nii.gz"

        # Copy CT volume (use read/write to avoid permission issues on Windows/WSL)
        if ncct_file.exists():
            dest_dir = ct_dir / case_id
            dest_dir.mkdir(exist_ok=True)
            dest_file = dest_dir / "ct.nii.gz"
            with open(ncct_file, 'rb') as f_in:
                with open(dest_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            ct_count += 1

        # Copy ADC volume
        if adc_file.exists():
            dest_dir = adc_dir / case_id
            dest_dir.mkdir(exist_ok=True)
            dest_file = dest_dir / "adc.nii.gz"
            with open(adc_file, 'rb') as f_in:
                with open(dest_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            adc_count += 1

        # Copy lesion mask
        if mask_file.exists():
            dest_file = mask_dir / f"{case_id}.nii.gz"
            with open(mask_file, 'rb') as f_in:
                with open(dest_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            mask_count += 1

    print(f"   ✓ Organized {len(case_dirs)} cases:")
    print(f"     - CT volumes: {ct_count}")
    print(f"     - ADC volumes: {adc_count}")
    print(f"     - Lesion masks: {mask_count}")
    print()

    # Clean up temp directory
    print("3. Cleaning up temporary files...")
    shutil.rmtree(extract_temp)
    print("   ✓ Cleanup complete")
    print()

    # Summary
    print("=" * 60)
    print("APIS Dataset Ready!")
    print("=" * 60)
    print()
    print(f"Dataset location: data/apis/raw/")
    print(f"  - CT volumes:     {ct_count} cases")
    print(f"  - ADC/MRI volumes: {adc_count} cases")
    print(f"  - Lesion masks:    {mask_count} cases")
    print()
    print("Next step: Run verification")
    print("  python verify_ready.py")
    print()
    print("Then: Run complete workflow")
    print("  bash workflow/complete_workflow_with_logging.sh apis run_$(date +%Y%m%d_%H%M%S)")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())