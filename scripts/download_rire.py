#!/usr/bin/env python3
"""
Download RIRE (Retrospective Image Registration Evaluation) dataset
for registration robustness testing in ClinFuseDiff++

RIRE provides CT-MR pairs with known ground-truth transformations,
perfect for validating registration accuracy and stress testing.

Reference: https://rire.insight-journal.org/
"""

import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
import sys

def download_file(url, output_path, description=""):
    """Download file with progress bar"""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded: {output_path.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def extract_gz(gz_path, output_path):
    """Extract .gz file"""
    print(f"  Extracting: {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()  # Remove .gz file
    print(f"  ✓ Extracted: {output_path}")

def main():
    print("=" * 70)
    print("RIRE Dataset Download for ClinFuseDiff++")
    print("Registration Robustness Testing")
    print("=" * 70)
    print()

    repo_root = Path(__file__).parent.parent
    rire_dir = repo_root / "data" / "rire" / "raw"
    rire_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {rire_dir}")
    print()

    # RIRE base URL
    base_url = "https://rire.insight-journal.org/download/data"

    # RIRE dataset structure:
    # We'll download a few representative patients for testing
    # Each patient has: CT, MR-T1, MR-T2, MR-PD, and ground-truth transformations

    patients = [
        "patient_001",  # Representative case
        "patient_002",  # Another case for validation
    ]

    modalities = {
        "ct": "CT rectified image",
        "mr_T1": "MR T1-weighted image",
        "mr_T2": "MR T2-weighted image",
    }

    print("=" * 70)
    print("Note: RIRE Dataset Download")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: RIRE dataset requires manual download")
    print()
    print("The RIRE dataset is hosted at:")
    print("  https://rire.insight-journal.org/")
    print()
    print("Download instructions:")
    print("  1. Visit: https://rire.insight-journal.org/download")
    print("  2. Download 'Training Images' section")
    print("  3. Select patients (e.g., patient_001, patient_002)")
    print("  4. Download modalities:")
    print("     - CT rectified images")
    print("     - MR T1-weighted images")
    print("     - MR T2-weighted images")
    print("     - Ground-truth transformations (if available)")
    print()
    print("Expected structure:")
    print(f"  {rire_dir}/")
    print("    ├── patient_001/")
    print("        ├── ct.hdr")
    print("        ├── ct.img")
    print("        ├── mr_T1.hdr")
    print("        ├── mr_T1.img")
    print("        └── transform_ct2mr.txt  (ground-truth)")
    print("    ├── patient_002/")
    print("        └── ... (same structure)")
    print()
    print("=" * 70)
    print()

    # Check if dataset already exists
    existing_patients = [d for d in rire_dir.iterdir() if d.is_dir() and d.name.startswith("patient_")]

    if existing_patients:
        print(f"✓ Found {len(existing_patients)} existing patients:")
        for p in existing_patients:
            ct_files = list(p.glob("ct.*"))
            mr_files = list(p.glob("mr_*.*"))
            print(f"  - {p.name}: {len(ct_files)} CT files, {len(mr_files)} MR files")
        print()
        print("Dataset appears to be already downloaded.")
    else:
        print("⚠️  No RIRE data found. Please download manually.")
        print()
        print("Alternative: Use sample synthetic data for testing")
        print()

        # Create placeholder structure for documentation
        for patient in patients:
            patient_dir = rire_dir / patient
            patient_dir.mkdir(exist_ok=True)

            readme = patient_dir / "README.txt"
            with open(readme, 'w') as f:
                f.write(f"RIRE {patient}\n")
                f.write("=" * 40 + "\n\n")
                f.write("Expected files:\n")
                f.write("  - ct.hdr, ct.img     (CT rectified)\n")
                f.write("  - mr_T1.hdr, mr_T1.img (MR T1-weighted)\n")
                f.write("  - mr_T2.hdr, mr_T2.img (MR T2-weighted)\n")
                f.write("  - transform_ct2mr.txt  (ground-truth)\n")
                f.write("\n")
                f.write("Download from:\n")
                f.write("  https://rire.insight-journal.org/download\n")

    print()
    print("=" * 70)
    print("RIRE Download Summary")
    print("=" * 70)
    print()
    print(f"Output directory: {rire_dir}")
    print()

    if existing_patients:
        print(f"✓ Status: {len(existing_patients)} patients available")
        print()
        print("Next step: Convert ANALYZE format to NIfTI")
        print("  python scripts/convert_rire_to_nifti.py")
    else:
        print("⚠️  Status: Manual download required")
        print()
        print("Visit: https://rire.insight-journal.org/download")

    print()
    print("=" * 70)
    print()
    print("Purpose: RIRE dataset for registration robustness testing")
    print("  - Known ground-truth transformations")
    print("  - Multi-modal CT-MR pairs")
    print("  - Stress testing with synthetic misalignment")
    print("  - Validate registration accuracy (TRE < 2mm)")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())