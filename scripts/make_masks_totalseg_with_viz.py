#!/usr/bin/env python3
"""
Generate ROI masks (brain, bone/skull) using TotalSegmentator
WITH comprehensive visualization and logging
For CVPR 2026 CLIN-FuseDiff++ proposal
"""

import os
from typing import Optional
# Device selection is configured at runtime in configure_device()

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def configure_device(requested: str, log_file) -> str:
    """Configure GPU/CPU behavior for TotalSegmentator.

    requested: 'auto' | 'cpu' | 'gpu'
    returns: selected device 'cpu' or 'gpu'
    """
    # Clear any previous forced-CPU settings from earlier runs
    for var in ["TOTALSEG_DISABLE_GPU"]:
        if var in os.environ:
            del os.environ[var]

    selected = 'cpu'

    if requested == 'cpu':
        os.environ['TOTALSEG_DISABLE_GPU'] = '1'
        selected = 'cpu'
        log_message(log_file, "Device: CPU (forced)")
        return selected

    # For 'gpu' and 'auto', check CUDA via torch if available
    cuda_ok = False
    try:
        import torch  # noqa: F401
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if requested == 'gpu':
        if not cuda_ok:
            os.environ['TOTALSEG_DISABLE_GPU'] = '1'
            log_message(log_file, "GPU requested but not available → falling back to CPU")
            selected = 'cpu'
        else:
            selected = 'gpu'
            log_message(log_file, "Device: GPU (CUDA available)")
        return selected

    # requested == 'auto'
    if cuda_ok:
        selected = 'gpu'
        log_message(log_file, "Device: GPU (auto-detected)")
    else:
        os.environ['TOTALSEG_DISABLE_GPU'] = '1'
        selected = 'cpu'
        log_message(log_file, "Device: CPU (no CUDA detected)")
    return selected


def setup_logging(output_dir):
    """Setup logging directory and files"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"totalseg_log_{timestamp}.txt"

    return log_file


def log_message(log_file, message):
    """Log message to file and print"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')


def run_totalseg(input_path, output_dir, roi_subset=None, fast=True, task=None, log_file=None):
    """
    Run TotalSegmentator CLI with detailed logging
    """
    cmd = ["TotalSegmentator", "-i", str(input_path), "-o", str(output_dir)]

    if roi_subset:
        cmd.extend(["--roi_subset"] + roi_subset.split())

    if fast:
        cmd.append("--fast")

    if task:
        cmd.extend(["--task", task])

    log_message(log_file, f"Running: {' '.join(cmd)}")

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Log stdout and stderr
    if result.stdout:
        log_message(log_file, f"STDOUT:\n{result.stdout}")
    if result.stderr:
        log_message(log_file, f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        log_message(log_file, "ERROR: TotalSegmentator failed")
        sys.exit(1)

    log_message(log_file, "✓ TotalSegmentator completed successfully")
    return output_dir


def create_overlay_visualization(
    image_data,
    mask_data,
    output_path,
    title="Segmentation Overlay",
    voxel_spacing=(1.0, 1.0, 1.0),
):
    """
    Create 3-view (axial, sagittal, coronal) overlay visualization
    with proper aspect ratio to account for anisotropic voxel spacing
    """
    # Get middle slices
    ax_slice = image_data.shape[2] // 2
    sag_slice = image_data.shape[0] // 2
    cor_slice = image_data.shape[1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Interpolation settings for production-grade paper figures:
    # - bilinear: balanced sharpness for medical imaging (standard for papers)
    # - nearest for masks: preserve crisp boundaries
    img_interp = 'bilinear'
    mask_interp = 'nearest'
    img_common = dict(origin='lower', interpolation=img_interp)
    mask_common = dict(origin='lower', interpolation=mask_interp)

    # Unpack voxel spacing (size per voxel along x, y, z)
    sx, sy, sz = voxel_spacing

    # Compute extents so that pixel geometry reflects real spacing
    # Note: After .T, array shape becomes (rows, cols) = (dim2, dim1)

    # Axial view (XY plane)
    axial_img = image_data[:, :, ax_slice].T
    axial_msk = mask_data[:, :, ax_slice].T
    axial_extent = [0, image_data.shape[0] * sx, 0, image_data.shape[1] * sy]
    axes[0].imshow(axial_img, cmap='gray', extent=axial_extent, aspect='equal', **img_common)
    axes[0].imshow(axial_msk, cmap='Reds', alpha=0.3, extent=axial_extent, aspect='equal', **mask_common)
    axes[0].set_title(f'Axial (slice {ax_slice})')
    axes[0].axis('off')

    # Sagittal view (YZ plane)
    sag_img = image_data[sag_slice, :, :].T  # (Z, Y)
    sag_msk = mask_data[sag_slice, :, :].T
    sag_extent = [0, image_data.shape[1] * sy, 0, image_data.shape[2] * sz]
    axes[1].imshow(sag_img, cmap='gray', extent=sag_extent, aspect='equal', **img_common)
    axes[1].imshow(sag_msk, cmap='Reds', alpha=0.3, extent=sag_extent, aspect='equal', **mask_common)
    axes[1].set_title(f'Sagittal (slice {sag_slice})')
    axes[1].axis('off')

    # Coronal view (XZ plane)
    cor_img = image_data[:, cor_slice, :].T  # (Z, X)
    cor_msk = mask_data[:, cor_slice, :].T
    cor_extent = [0, image_data.shape[0] * sx, 0, image_data.shape[2] * sz]
    axes[2].imshow(cor_img, cmap='gray', extent=cor_extent, aspect='equal', **img_common)
    axes[2].imshow(cor_msk, cmap='Reds', alpha=0.3, extent=cor_extent, aspect='equal', **mask_common)
    axes[2].set_title(f'Coronal (slice {cor_slice})')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_segmentation_stats(mask_data, output_path, log_file):
    """Save segmentation statistics"""
    stats = {
        'total_voxels': int(mask_data.size),
        'segmented_voxels': int(mask_data.sum()),
        'segmentation_ratio': float(mask_data.sum() / mask_data.size),
        'bounding_box': {
            'x': [int(np.where(mask_data)[0].min()), int(np.where(mask_data)[0].max())],
            'y': [int(np.where(mask_data)[1].min()), int(np.where(mask_data)[1].max())],
            'z': [int(np.where(mask_data)[2].min()), int(np.where(mask_data)[2].max())],
        } if mask_data.sum() > 0 else None
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    log_message(log_file, f"Segmentation stats: {stats['segmented_voxels']:,} / {stats['total_voxels']:,} voxels ({stats['segmentation_ratio']:.2%})")

    return stats


def combine_brain_mask(seg_dir, output_path, log_file):
    """
    Combine brain-related structures into single brain mask
    WITH visualization
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
        log_message(log_file, "WARNING: No brain structures found")
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
            log_message(log_file, f"  Added: {struct_name}")

    log_message(log_file, f"  Combined {count} brain structures")

    # Save combined mask
    out_img = nib.Nifti1Image(combined_mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, output_path)
    log_message(log_file, f"✓ Saved brain mask: {output_path}")

    return output_path


def combine_bone_mask(seg_dir, output_path, log_file):
    """
    Combine bone/skull structures into single bone mask
    WITH visualization
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
        log_message(log_file, "WARNING: No bone structures found")
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
            log_message(log_file, f"  Added: {bone_file}")

    log_message(log_file, f"  Combined {count} bone structures")

    # Save combined mask
    out_img = nib.Nifti1Image(combined_mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, output_path)
    log_message(log_file, f"✓ Saved bone mask: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROI masks using TotalSegmentator with comprehensive visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both brain and bone masks with visualization
  python scripts/make_masks_totalseg_with_viz.py \\
      --mri data/apis/raw/adc/case_001.nii.gz \\
      --ct work/reg/case_001/ct_in_mri.nii.gz \\
      --out work/masks/case_001
        """
    )

    parser.add_argument("--mri", type=str, help="MRI image (for brain mask)")
    parser.add_argument("--ct", type=str, help="CT image (for bone mask)")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (default: True)", default=True)
    parser.add_argument("--no-fast", dest="fast", action="store_false", help="Use full mode (slower, more accurate)")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization images", default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Computation device: auto-detect GPU if available")

    args = parser.parse_args()

    if not args.mri and not args.ct:
        print("ERROR: At least one of --mri or --ct must be provided")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = setup_logging(out_dir)
    log_message(log_file, "=" * 80)
    log_message(log_file, "TotalSegmentator ROI Mask Generation with Visualization")
    log_message(log_file, "=" * 80)
    log_message(log_file, f"Output directory: {out_dir}")
    log_message(log_file, f"Fast mode: {args.fast}")
    log_message(log_file, f"Save visualizations: {args.save_viz}")
    log_message(log_file, f"Requested device: {args.device}")

    # Configure device (GPU/CPU)
    selected_device = configure_device(args.device, log_file)

    # Create visualization directory
    viz_dir = out_dir / "visualizations"
    if args.save_viz:
        viz_dir.mkdir(exist_ok=True)

    # Process MRI (brain mask)
    if args.mri:
        log_message(log_file, "\n" + "="*80)
        log_message(log_file, "Processing MRI for brain mask...")
        log_message(log_file, "="*80)

        mri_path = Path(args.mri)

        if not mri_path.exists():
            log_message(log_file, f"ERROR: MRI file not found: {mri_path}")
            sys.exit(1)

        # Run TotalSegmentator on MRI
        mri_seg_dir = out_dir / "totalseg_mri"
        run_totalseg(
            mri_path,
            mri_seg_dir,
            roi_subset="brain",
            fast=args.fast,
            task=None,
            log_file=log_file
        )

        # Combine into brain mask
        brain_mask_path = out_dir / "brain_mask.nii.gz"
        combine_brain_mask(mri_seg_dir, brain_mask_path, log_file)

        # Create visualization
        if args.save_viz and brain_mask_path.exists():
            log_message(log_file, "Creating brain mask visualization...")
            mri_img = nib.load(mri_path)
            brain_mask_img = nib.load(brain_mask_path)

            create_overlay_visualization(
                mri_img.get_fdata(),
                brain_mask_img.get_fdata(),
                viz_dir / "brain_mask_overlay.png",
                title="Brain Mask on MRI",
                voxel_spacing=mri_img.header.get_zooms()[:3],
            )
            log_message(log_file, f"✓ Saved visualization: {viz_dir / 'brain_mask_overlay.png'}")

            # Save stats
            stats_path = out_dir / "brain_mask_stats.json"
            save_segmentation_stats(brain_mask_img.get_fdata(), stats_path, log_file)

    # Process CT (bone mask)
    if args.ct:
        log_message(log_file, "\n" + "="*80)
        log_message(log_file, "Processing CT for bone mask...")
        log_message(log_file, "="*80)

        ct_path = Path(args.ct)

        if not ct_path.exists():
            log_message(log_file, f"ERROR: CT file not found: {ct_path}")
            sys.exit(1)

        # Run TotalSegmentator on CT
        ct_seg_dir = out_dir / "totalseg_ct"
        run_totalseg(
            ct_path,
            ct_seg_dir,
            roi_subset="skull",
            fast=args.fast,
            task=None,
            log_file=log_file
        )

        # Combine into bone mask
        bone_mask_path = out_dir / "bone_mask.nii.gz"
        combine_bone_mask(ct_seg_dir, bone_mask_path, log_file)

        # Create visualization
        if args.save_viz and bone_mask_path.exists():
            log_message(log_file, "Creating bone mask visualization...")
            ct_img = nib.load(ct_path)
            bone_mask_img = nib.load(bone_mask_path)

            create_overlay_visualization(
                ct_img.get_fdata(),
                bone_mask_img.get_fdata(),
                viz_dir / "bone_mask_overlay.png",
                title="Bone Mask on CT",
                voxel_spacing=ct_img.header.get_zooms()[:3],
            )
            log_message(log_file, f"✓ Saved visualization: {viz_dir / 'bone_mask_overlay.png'}")

            # Save stats
            stats_path = out_dir / "bone_mask_stats.json"
            save_segmentation_stats(bone_mask_img.get_fdata(), stats_path, log_file)

    log_message(log_file, "\n" + "="*80)
    log_message(log_file, "✓ ROI mask generation complete!")
    log_message(log_file, "="*80)
    log_message(log_file, f"\nOutput directory: {out_dir}")

    if args.mri:
        log_message(log_file, "  - brain_mask.nii.gz")
        if args.save_viz:
            log_message(log_file, "  - visualizations/brain_mask_overlay.png")
            log_message(log_file, "  - brain_mask_stats.json")
    if args.ct:
        log_message(log_file, "  - bone_mask.nii.gz")
        if args.save_viz:
            log_message(log_file, "  - visualizations/bone_mask_overlay.png")
            log_message(log_file, "  - bone_mask_stats.json")

    log_message(log_file, f"  - logs/totalseg_log_*.txt")
    log_message(log_file, "")


if __name__ == "__main__":
    main()
