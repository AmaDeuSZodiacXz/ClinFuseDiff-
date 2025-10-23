#!/usr/bin/env python3
"""
Generate ROI masks (brain, bone/skull) using TotalSegmentator
For CVPR 2026 CLIN-FuseDiff++ proposal
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def run_totalseg(input_path, output_dir, roi_subset=None, fast=True, task=None):
    """
    Run TotalSegmentator CLI

    Args:
        input_path: Input NIfTI file
        output_dir: Output directory for segmentations
        roi_subset: Subset of ROIs to segment (e.g., 'brain', 'vertebrae')
        fast: Use fast mode (less accurate but faster)
        task: Specific task ('total', 'body', 'lung_vessels', 'cerebral_bleed', etc.)
    """
    cmd = ["TotalSegmentator", "-i", str(input_path), "-o", str(output_dir)]

    if roi_subset:
        cmd.extend(["--roi_subset"] + roi_subset.split())

    if fast:
        cmd.append("--fast")

    if task:
        cmd.extend(["--task", task])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: TotalSegmentator failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    print(f"✓ TotalSegmentator completed")
    return output_dir


def combine_brain_mask(seg_dir, output_path):
    """
    Combine brain-related structures into single brain mask

    TotalSegmentator brain structures:
    - brain (whole brain)
    - brainstem
    - cerebellum
    """
    seg_dir = Path(seg_dir)

    # List of brain structures to combine
    brain_structures = [
        "brain.nii.gz",
        "brainstem.nii.gz",
    ]

    # Load first structure as template
    first_struct = None
    for struct_name in brain_structures:
        struct_path = seg_dir / struct_name
        if struct_path.exists():
            first_struct = struct_path
            break

    if first_struct is None:
        print("WARNING: No brain structures found")
        return None

    # Load reference
    ref_img = nib.load(first_struct)
    combined_mask = np.zeros(ref_img.shape, dtype=np.uint8)

    # Combine all brain structures
    count = 0
    for struct_name in brain_structures:
        struct_path = seg_dir / struct_name
        if struct_path.exists():
            struct_img = nib.load(struct_path)
            combined_mask = np.logical_or(combined_mask, struct_img.get_fdata() > 0)
            count += 1
            print(f"  Added: {struct_name}")

    print(f"  Combined {count} brain structures")

    # Save combined mask
    out_img = nib.Nifti1Image(combined_mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, output_path)
    print(f"✓ Saved brain mask: {output_path}")

    return output_path


def combine_bone_mask(seg_dir, output_path):
    """
    Combine bone/skull structures into single bone mask

    TotalSegmentator bone structures:
    - skull (if available)
    - vertebrae_*
    - rib_*
    """
    seg_dir = Path(seg_dir)

    # Find all bone-related structures
    bone_files = []

    # Skull
    if (seg_dir / "skull.nii.gz").exists():
        bone_files.append("skull.nii.gz")

    # Vertebrae
    for struct in seg_dir.glob("vertebrae_*.nii.gz"):
        bone_files.append(struct.name)

    # Ribs (if thorax included)
    for struct in seg_dir.glob("rib_*.nii.gz"):
        bone_files.append(struct.name)

    if len(bone_files) == 0:
        print("WARNING: No bone structures found")
        return None

    # Load first structure as template
    ref_img = nib.load(seg_dir / bone_files[0])
    combined_mask = np.zeros(ref_img.shape, dtype=np.uint8)

    # Combine all bone structures
    count = 0
    for bone_file in bone_files:
        bone_path = seg_dir / bone_file
        if bone_path.exists():
            bone_img = nib.load(bone_path)
            combined_mask = np.logical_or(combined_mask, bone_img.get_fdata() > 0)
            count += 1
            print(f"  Added: {bone_file}")

    print(f"  Combined {count} bone structures")

    # Save combined mask
    out_img = nib.Nifti1Image(combined_mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, output_path)
    print(f"✓ Saved bone mask: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROI masks using TotalSegmentator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate brain mask from MRI
  python scripts/make_masks_totalseg.py --mri mri.nii.gz --out work/masks/case001

  # Generate bone mask from CT
  python scripts/make_masks_totalseg.py --ct ct.nii.gz --out work/masks/case001

  # Generate both (separate runs)
  python scripts/make_masks_totalseg.py --mri mri.nii.gz --ct ct.nii.gz --out work/masks/case001
        """
    )

    parser.add_argument("--mri", type=str, help="MRI image (for brain mask)")
    parser.add_argument("--ct", type=str, help="CT image (for bone mask)")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (default: True)", default=True)
    parser.add_argument("--no-fast", dest="fast", action="store_false", help="Use full mode (slower, more accurate)")

    args = parser.parse_args()

    if not args.mri and not args.ct:
        print("ERROR: At least one of --mri or --ct must be provided")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("TotalSegmentator ROI Mask Generation")
    print("=" * 50)
    print("")

    # Process MRI (brain mask)
    if args.mri:
        print("Processing MRI for brain mask...")
        mri_path = Path(args.mri)

        if not mri_path.exists():
            print(f"ERROR: MRI file not found: {mri_path}")
            sys.exit(1)

        # Run TotalSegmentator on MRI
        mri_seg_dir = out_dir / "totalseg_mri"
        run_totalseg(
            mri_path,
            mri_seg_dir,
            roi_subset="brain",
            fast=args.fast,
            task=None  # Auto-detect modality
        )

        # Combine into brain mask
        brain_mask_path = out_dir / "brain_mask.nii.gz"
        combine_brain_mask(mri_seg_dir, brain_mask_path)
        print("")

    # Process CT (bone mask)
    if args.ct:
        print("Processing CT for bone mask...")
        ct_path = Path(args.ct)

        if not ct_path.exists():
            print(f"ERROR: CT file not found: {ct_path}")
            sys.exit(1)

        # Run TotalSegmentator on CT
        ct_seg_dir = out_dir / "totalseg_ct"
        run_totalseg(
            ct_path,
            ct_seg_dir,
            roi_subset="skull vertebrae",
            fast=args.fast,
            task=None  # Auto-detect modality
        )

        # Combine into bone mask
        bone_mask_path = out_dir / "bone_mask.nii.gz"
        combine_bone_mask(ct_seg_dir, bone_mask_path)
        print("")

    print("=" * 50)
    print("✓ ROI mask generation complete!")
    print("=" * 50)
    print("")
    print(f"Output directory: {out_dir}")

    if args.mri:
        print(f"  - brain_mask.nii.gz")
    if args.ct:
        print(f"  - bone_mask.nii.gz")

    print("")


if __name__ == "__main__":
    main()