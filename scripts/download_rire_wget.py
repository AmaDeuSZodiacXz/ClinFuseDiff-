#!/usr/bin/env python3
"""
Download RIRE dataset using wget
For registration robustness testing in ClinFuseDiff++
"""

import subprocess
import sys
from pathlib import Path
import tarfile

def download_file(url, output_path, description=""):
    """Download file using wget"""
    print(f"----------------------------------------------------------------------")
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print()

    if output_path.exists():
        print(f"  ✓ Already exists ({output_path.stat().st_size / 1e6:.1f} MB), skipping...")
        return True

    try:
        # Run wget with continue option
        result = subprocess.run(
            ["wget", "-c", "-O", str(output_path), url],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            size_mb = output_path.stat().st_size / 1e6
            print(f"  ✓ Downloaded: {size_mb:.1f} MB")
            return True
        else:
            print(f"  ✗ Download failed!")
            print(result.stderr[:500])
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timeout!")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def extract_tar_gz(tar_path, extract_to):
    """Extract tar.gz file"""
    print(f"Extracting: {tar_path.name}")

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False

def main():
    print("=" * 70)
    print("RIRE Dataset Download (training_001)")
    print("=" * 70)
    print()

    repo_root = Path(__file__).parent.parent
    rire_dir = repo_root / "data" / "rire" / "raw"
    rire_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {rire_dir}")
    print()

    # IPFS gateway base URL
    ipfs_base = "https://dweb.link/ipfs/bafybeih23xv6uamx7k27wk4uvzkxdtdryqeok22hpl3ybideggcjhipwme/rire"

    # Files to download
    files = {
        "ct.tar.gz": "CT image",
        "mr_T1_rectified.tar.gz": "MR T1 rectified",
        "mr_T2_rectified.tar.gz": "MR T2 rectified",
    }

    print("Downloading RIRE training_001 dataset...")
    print()

    # Download each file
    for filename, description in files.items():
        url = f"{ipfs_base}/{filename}"
        output_path = rire_dir / filename

        success = download_file(url, output_path, description)
        if not success:
            print(f"✗ Failed to download {filename}")
            return 1

        print()

    print("=" * 70)
    print("Extracting archives...")
    print("=" * 70)
    print()

    # Extract each tar.gz
    for filename in files.keys():
        tar_path = rire_dir / filename
        if tar_path.exists():
            extract_tar_gz(tar_path, rire_dir)
            print()

    print("=" * 70)
    print("Organizing files...")
    print("=" * 70)
    print()

    # Create patient_001 directory
    patient_dir = rire_dir / "patient_001"
    patient_dir.mkdir(exist_ok=True)

    # Move extracted directories to patient folder
    for subdir in ["ct", "mr_T1_rectified", "mr_T2_rectified"]:
        subdir_path = rire_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            # Move all files from subdir to patient_dir
            for file in subdir_path.iterdir():
                dest = patient_dir / file.name
                file.rename(dest)
                print(f"  Moved: {file.name} → patient_001/")

            # Remove empty directory
            subdir_path.rmdir()

    print()

    # Rename files for consistency
    print("Renaming files for consistency...")

    # CT files: ct_* → ct.*
    for f in patient_dir.glob("ct_*"):
        ext = f.suffix
        if ext:
            new_name = f"ct{ext}"
            f.rename(patient_dir / new_name)
            print(f"  {f.name} → {new_name}")

    # MR T1 files: mr_T1_* → mr_T1.*
    for f in patient_dir.glob("mr_T1_*"):
        ext = f.suffix
        if ext:
            new_name = f"mr_T1{ext}"
            f.rename(patient_dir / new_name)
            print(f"  {f.name} → {new_name}")

    # MR T2 files: mr_T2_* → mr_T2.*
    for f in patient_dir.glob("mr_T2_*"):
        ext = f.suffix
        if ext:
            new_name = f"mr_T2{ext}"
            f.rename(patient_dir / new_name)
            print(f"  {f.name} → {new_name}")

    print()
    print("=" * 70)
    print("RIRE Dataset Download Complete!")
    print("=" * 70)
    print()

    # List downloaded files
    print(f"Downloaded files in: {patient_dir}/")
    files_list = sorted(patient_dir.iterdir())
    for f in files_list:
        size_mb = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"  - {f.name:<30} {size_mb:>8.1f} MB")

    print()
    print("File formats: ANALYZE (.hdr + .img)")
    print()
    print("Next step: Convert to NIfTI format")
    print("  python scripts/convert_rire_to_nifti.py")
    print()
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())