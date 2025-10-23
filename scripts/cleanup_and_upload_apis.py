#!/usr/bin/env python3
"""
Clean up APIS dataset folder and upload to Hugging Face Hub.

This script:
1. Removes temporary folders (splits_lesion_only, upload_hf)
2. Removes 6 non-lesion cases from preproc/
3. Updates splits/ with lesion-only metadata
4. Uploads entire data/apis folder to HF Hub
"""

import os
import shutil
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Cases without lesions to remove
NO_LESION_CASES = [
    'train_027', 'train_038', 'train_048',
    'train_051', 'train_058', 'train_059'
]

def main():
    parser = argparse.ArgumentParser(description="Clean and upload APIS dataset")
    parser.add_argument('--apis-dir', type=str, default='data/apis',
                        help='Path to APIS data directory')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='Hugging Face repo ID (e.g., username/dataset-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run without actually deleting or uploading')
    args = parser.parse_args()

    apis_dir = Path(args.apis_dir).resolve()
    preproc_dir = apis_dir / 'preproc'
    splits_dir = apis_dir / 'splits'
    splits_lesion_dir = apis_dir / 'splits_lesion_only'
    upload_hf_dir = apis_dir / 'upload_hf'

    token = args.token or os.environ.get('HF_TOKEN')

    print("=" * 80)
    print("CLEAN AND UPLOAD APIS DATASET TO HUGGING FACE")
    print("=" * 80)
    print()

    # Step 1: Remove temporary folders
    print("Step 1: Removing temporary folders...")
    temp_folders = [splits_lesion_dir, upload_hf_dir]
    for folder in temp_folders:
        if folder.exists():
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {folder}")
            else:
                print(f"  Removing: {folder}")
                shutil.rmtree(folder)
        else:
            print(f"  Not found (skip): {folder}")
    print()

    # Step 2: Remove non-lesion cases from preproc/
    print("Step 2: Removing 6 non-lesion cases from preproc/...")
    removed_count = 0
    for case_id in NO_LESION_CASES:
        case_dir = preproc_dir / case_id
        if case_dir.exists():
            if args.dry_run:
                print(f"  [DRY RUN] Would remove: {case_id}")
            else:
                print(f"  Removing: {case_id}")
                shutil.rmtree(case_dir)
            removed_count += 1
        else:
            print(f"  Not found (skip): {case_id}")

    remaining_cases = len([d for d in preproc_dir.iterdir() if d.is_dir()])
    print(f"\n  ✓ Removed {removed_count} cases")
    print(f"  ✓ Remaining cases: {remaining_cases} (expected: 54)")
    print()

    # Step 3: Update splits metadata
    print("Step 3: Updating splits/ with lesion-only metadata...")
    if splits_lesion_dir.exists():
        split_files = ['train.txt', 'val.txt', 'test.txt', 'split_metadata.json']
        for filename in split_files:
            src = splits_lesion_dir / filename
            dst = splits_dir / filename
            if src.exists():
                if args.dry_run:
                    print(f"  [DRY RUN] Would copy: {filename}")
                else:
                    print(f"  Copying: {filename}")
                    shutil.copy2(src, dst)
            else:
                print(f"  Not found (skip): {filename}")
    else:
        print("  Warning: splits_lesion_only/ not found. Splits not updated.")
    print()

    # Step 4: Verify structure
    print("Step 4: Verifying final structure...")
    print(f"  apis_dir: {apis_dir}")
    print(f"    preproc/: {len(list(preproc_dir.glob('train_*')))} cases")
    print(f"    splits/: {len(list(splits_dir.glob('*.txt')))} split files")

    if not splits_dir.exists() or not preproc_dir.exists():
        print("\n  ✗ Error: Missing required directories!")
        return

    print("\n  ✓ Structure verified")
    print()

    # Step 5: Upload to Hugging Face
    if args.dry_run:
        print("Step 5: [DRY RUN] Would upload to Hugging Face Hub...")
        print(f"  Repo: {args.repo_id}")
        print(f"  Source: {apis_dir}")
        print(f"  Files: ~{sum(1 for _ in apis_dir.rglob('*') if _.is_file())} files")
        return

    print("Step 5: Uploading to Hugging Face Hub...")
    print(f"  Repo: {args.repo_id}")
    print(f"  Source: {apis_dir}")

    if not token:
        print("\n  ✗ Error: No Hugging Face token provided!")
        print("  Use --token or set HF_TOKEN environment variable")
        return

    try:
        # Create repo if it doesn't exist
        api = HfApi(token=token)
        try:
            create_repo(
                repo_id=args.repo_id,
                repo_type="dataset",
                exist_ok=True,
                token=token
            )
            print(f"\n  ✓ Repository ready: {args.repo_id}")
        except Exception as e:
            print(f"\n  Note: Repo creation check: {e}")

        # Create README
        readme_content = f"""---
license: cc-by-4.0
task_categories:
- image-segmentation
- image-to-image
tags:
- medical
- neuroimaging
- stroke
- image-fusion
pretty_name: APIS Stroke Dataset (Lesion Cases Only)
size_categories:
- n<1K
---

# APIS Stroke Dataset - Preprocessed (Lesion Cases Only)

This dataset contains **54 acute ischemic stroke cases** with expert lesion annotations from the APIS dataset.

## Dataset Structure

```
preproc/
  train_000/
    ct.nii.gz              # CT scan
    mri.nii.gz             # Registered MRI (ADC)
    brain_mask.nii.gz      # Brain ROI mask (TotalSegmentator)
    bone_mask.nii.gz       # Bone/skull ROI mask (TotalSegmentator)
    lesion_mask.nii.gz     # Expert-annotated lesion segmentation
  train_001/
    ...
  (54 cases total)

splits/
  train.txt              # 37 cases (68.5%)
  val.txt                # 8 cases (14.8%)
  test.txt               # 9 cases (16.7%)
  split_metadata.json    # Split statistics
```

## Excluded Cases

6 cases without lesions were excluded:
- train_027, train_038, train_048, train_051, train_058, train_059

## Usage

```python
from pathlib import Path
import nibabel as nib

# Download dataset
from huggingface_hub import snapshot_download
data_dir = snapshot_download(repo_id="{args.repo_id}", repo_type="dataset")

# Load a case
case_dir = Path(data_dir) / "preproc" / "train_000"
ct = nib.load(case_dir / "ct.nii.gz")
mri = nib.load(case_dir / "mri.nii.gz")
lesion_mask = nib.load(case_dir / "lesion_mask.nii.gz")
```

## Citation

```bibtex
@article{{li2023apis,
  title={{APIS: A paired CT-MRI dataset with lesion labels for acute ischemic stroke}},
  author={{Li, Zongwei and others}},
  journal={{Scientific Data}},
  year={{2023}}
}}
```

## Preprocessing

- **Registration**: MRI (ADC) registered to CT using ANTs SyN
- **ROI Masks**: Generated using TotalSegmentator v2
- **Normalization**: CT windowed to brain (C=40, W=400 HU)
- **Format**: NIfTI (.nii.gz), isotropic 1mm spacing

## License

CC-BY-4.0 (original APIS dataset license)
"""

        readme_path = apis_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"  ✓ Created README.md")

        # Upload folder
        print(f"\n  Uploading files...")
        upload_folder(
            folder_path=str(apis_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload APIS dataset (lesion cases only, 54 cases) - clean structure"
        )

        print(f"\n  ✓ Upload complete!")
        print(f"\n  Dataset: https://huggingface.co/datasets/{args.repo_id}")

    except Exception as e:
        print(f"\n  ✗ Upload failed: {e}")
        return

    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()