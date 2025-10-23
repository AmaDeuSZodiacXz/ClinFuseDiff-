#!/usr/bin/env python3
"""
Download APIS dataset from Hugging Face Hub to Colab.

Usage in Colab:
    !python scripts/download_data_colab.py
"""

import os
import subprocess
from pathlib import Path

def main():
    # Configuration
    repo_id = "Pakawat-Phasook/ClinFuseDiff-APIS-Data"
    target_dir = Path("/content/ClinFuseDiff-/data/apis")

    print("=" * 80)
    print("DOWNLOAD APIS DATASET FROM HUGGING FACE HUB")
    print("=" * 80)
    print()
    print(f"Repository: {repo_id}")
    print(f"Target directory: {target_dir}")
    print()

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {target_dir}")
    print()

    # Download using huggingface-cli
    print("Downloading dataset...")
    print("This may take 5-10 minutes depending on your connection speed...")
    print()

    cmd = [
        "huggingface-cli", "download",
        repo_id,
        "--repo-type", "dataset",
        "--local-dir", str(target_dir),
        "--local-dir-use-symlinks", "False"
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print()
        print("✗ Download failed!")
        return 1

    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print()

    # Verify structure
    print(f"Dataset location: {target_dir}")
    print()

    preproc_dir = target_dir / "preproc"
    splits_dir = target_dir / "splits"

    if preproc_dir.exists():
        cases = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir()])
        print(f"✓ Found {len(cases)} cases in preproc/")
        print(f"  First 5: {cases[:5]}")
        print(f"  Last 5: {cases[-5:]}")
    else:
        print("✗ preproc/ directory not found!")

    print()

    if splits_dir.exists():
        split_files = list(splits_dir.glob("*.txt"))
        print(f"✓ Found {len(split_files)} split files:")
        for f in sorted(split_files):
            with open(f) as fp:
                n_lines = len(fp.readlines())
            print(f"  - {f.name}: {n_lines} cases")
    else:
        print("✗ splits/ directory not found!")

    print()
    print("Ready to train! Run:")
    print("  cd /content/ClinFuseDiff-")
    print("  python train.py --config configs/cvpr2026/train_roi.yaml --preset stroke")
    print()

    return 0

if __name__ == '__main__':
    exit(main())