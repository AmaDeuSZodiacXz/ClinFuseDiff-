#!/usr/bin/env python3
"""
Copy lesion masks from raw to preprocessed directory

APIS dataset structure:
- raw/lesion_masks/train_000.nii.gz
- preproc/train_000/lesion_mask.nii.gz  ← target
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # Paths
    raw_lesion_dir = Path("data/apis/raw/lesion_masks")
    preproc_dir = Path("data/apis/preproc")

    if not raw_lesion_dir.exists():
        print(f"❌ Raw lesion masks directory not found: {raw_lesion_dir}")
        return

    if not preproc_dir.exists():
        print(f"❌ Preprocessed directory not found: {preproc_dir}")
        return

    # Find all lesion mask files
    lesion_files = sorted(raw_lesion_dir.glob("*.nii.gz"))

    print(f"Found {len(lesion_files)} lesion mask files")
    print(f"Copying to preprocessed directories...\n")

    copied = 0
    skipped = 0

    for lesion_file in tqdm(lesion_files, desc="Copying lesion masks"):
        # Extract case ID (train_000.nii.gz → train_000)
        case_id = lesion_file.stem.replace('.nii', '')

        # Target directory
        case_dir = preproc_dir / case_id

        if not case_dir.exists():
            print(f"⚠️  Case directory not found: {case_dir}, skipping {case_id}")
            skipped += 1
            continue

        # Target file path
        target_file = case_dir / "lesion_mask.nii.gz"

        # Copy file by reading and writing (avoid permission issues)
        try:
            with open(lesion_file, 'rb') as src:
                with open(target_file, 'wb') as dst:
                    dst.write(src.read())
            copied += 1
        except Exception as e:
            print(f"❌ Failed to copy {case_id}: {e}")
            skipped += 1
            continue

    print(f"\n✅ Successfully copied {copied} lesion masks")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} files (case directory not found)")

    # Verify
    print(f"\nVerification:")
    for case_dir in sorted(preproc_dir.iterdir())[:5]:
        if case_dir.is_dir():
            lesion_mask = case_dir / "lesion_mask.nii.gz"
            status = "✅" if lesion_mask.exists() else "❌"
            print(f"  {status} {case_dir.name}/lesion_mask.nii.gz")

if __name__ == "__main__":
    main()