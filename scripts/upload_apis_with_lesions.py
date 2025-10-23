#!/usr/bin/env python3
"""
Upload APIS Dataset (with Lesion Masks) to Hugging Face

This version includes:
- CT images
- MRI/ADC images
- Brain masks (TotalSegmentator)
- Bone masks (TotalSegmentator)
- Lesion masks (expert annotations) ← NEW!
- Train/val/test splits

Repository: Pakawat-Phasook/ClinFuseDiff-APIS-Data
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

def get_directory_size(path):
    """Calculate total size of directory"""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def main():
    print("=" * 70)
    print("Uploading APIS Dataset (WITH LESION MASKS) to Hugging Face")
    print("=" * 70)

    repo_id = "Pakawat-Phasook/ClinFuseDiff-APIS-Data"
    data_dir = "data/apis"

    # Verify lesion masks exist
    preproc_dir = Path(data_dir) / "preproc"
    lesion_count = len(list(preproc_dir.glob("*/lesion_mask.nii.gz")))

    print(f"Repository: {repo_id}")
    print(f"Data directory: {data_dir}")
    print(f"Lesion masks found: {lesion_count}/60")

    if lesion_count < 60:
        print(f"\n⚠️  WARNING: Only {lesion_count} lesion masks found!")
        print("Run: python scripts/copy_lesion_masks.py")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Calculate size
    size_bytes = get_directory_size(data_dir)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Total size: ~{size_mb:.0f}MB\n")

    # Login
    print("Logging in...")
    login()

    # Create API
    api = HfApi()

    # Upload with new commit message
    print("Uploading complete data/apis folder structure...")
    print("This preserves: preproc/, raw/, splits/, splits.json")
    print("\n✨ NEW: Now includes lesion masks for all 60 cases!\n")

    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add lesion masks to all 60 cases (expert annotations from APIS challenge)",
        ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"]
    )

    print("\n" + "=" * 70)
    print("✅ Upload complete!")
    print("=" * 70)
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"Files uploaded: {lesion_count} lesion masks + CT/MRI/brain/bone masks")
    print("\nDataset now includes:")
    print("  ✅ CT images (60 cases)")
    print("  ✅ MRI/ADC images (60 cases)")
    print("  ✅ Brain masks (60 cases)")
    print("  ✅ Bone masks (60 cases)")
    print("  ✅ Lesion masks (60 cases) ← NEW!")
    print("  ✅ Train/val/test splits")

if __name__ == "__main__":
    main()