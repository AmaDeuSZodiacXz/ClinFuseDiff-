#!/usr/bin/env python3
"""
APIS Dataset Batch Preprocessing - All 60 Cases
Generates ROI masks (brain + bone) and organizes preprocessed data
"""
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def copy_file(src, dst):
    """Binary copy to avoid Windows/WSL permission errors"""
    with open(src, 'rb') as f_in:
        with open(dst, 'wb') as f_out:
            f_out.write(f_in.read())

def process_case(case_id, raw_dir, preproc_dir, work_dir, project_root, device: str, timeout_s: int):
    """Process single case"""
    result = {"case": case_id, "status": "pending"}
    
    try:
        case_preproc = preproc_dir / case_id
        case_work = work_dir / case_id
        
        case_preproc.mkdir(exist_ok=True, parents=True)
        case_work.mkdir(exist_ok=True, parents=True)
        
        # Input files
        ct_file = raw_dir / "ct" / case_id / "ct.nii.gz"
        adc_file = raw_dir / "adc" / case_id / "adc.nii.gz"
        lesion_file = raw_dir / "lesion_masks" / case_id / "lesion_mask.nii.gz"
        
        # Check inputs
        if not ct_file.exists():
            result["status"] = "error"
            result["reason"] = "missing CT"
            return result
        if not adc_file.exists():
            result["status"] = "error"
            result["reason"] = "missing ADC"
            return result
        
        # Generate ROI masks with TotalSegmentator
        cmd = [
            sys.executable,
            str(project_root / "scripts/make_masks_totalseg_with_viz.py"),
            "--mri", str(adc_file),
            "--ct", str(ct_file),
            "--out", str(case_work),
            "--device", device,
            "--fast",
            "--save-viz"
        ]
        
        proc_result = subprocess.run(cmd, cwd=project_root, capture_output=True, timeout=timeout_s)
        
        if proc_result.returncode != 0:
            result["status"] = "error"
            result["reason"] = "TotalSegmentator failed"
            return result
        
        # Copy files to preprocessing directory
        brain_mask_src = case_work / "brain_mask.nii.gz"
        bone_mask_src = case_work / "bone_mask.nii.gz"
        
        if brain_mask_src.exists():
            copy_file(brain_mask_src, case_preproc / "brain_mask.nii.gz")
        
        if bone_mask_src.exists():
            copy_file(bone_mask_src, case_preproc / "bone_mask.nii.gz")
        
        if lesion_file.exists():
            copy_file(lesion_file, case_preproc / "lesion_mask.nii.gz")
        
        copy_file(ct_file, case_preproc / "ct.nii.gz")
        copy_file(adc_file, case_preproc / "mri.nii.gz")
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["reason"] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="APIS batch preprocessing")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Computation device for TotalSegmentator")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-case timeout in seconds (includes first-time model download)")
    args = parser.parse_args()
    print("="*80)
    print("APIS Batch Preprocessing - All 60 Cases")
    print("="*80)
    print()
    
    # Paths
    project_root = Path(__file__).parent.parent.resolve()
    raw_dir = project_root / "data/apis/raw"
    preproc_dir = project_root / "data/apis/preproc"
    work_dir = project_root / "work" / f"apis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    preproc_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Raw data: {raw_dir}")
    log(f"Preprocessing output: {preproc_dir}")
    log(f"Work directory: {work_dir}")
    print()
    
    # Get all cases
    all_cases = sorted([d.name for d in (raw_dir / "ct").iterdir() if d.is_dir()])
    total = len(all_cases)
    
    log(f"Found {total} cases")
    log(f"Processing with parallel workers (device: {args.device})")
    print()
    
    # Process cases sequentially (TotalSegmentator is CPU-intensive)
    # Parallel processing would compete for CPU resources
    results = []
    
    for i, case_id in enumerate(all_cases, 1):
        log(f"[{i}/{total}] Processing {case_id}...")
        result = process_case(case_id, raw_dir, preproc_dir, work_dir, project_root, args.device, args.timeout)
        results.append(result)
        
        if result["status"] == "success":
            log(f"  ✓ {case_id} complete")
        else:
            log(f"  ✗ {case_id} failed: {result.get('reason', 'unknown')}")
        
        # Progress indicator every 10 cases
        if i % 10 == 0:
            success = sum(1 for r in results if r["status"] == "success")
            log(f"  Progress: {i}/{total} ({100*i//total}%) - {success} successful")
            print()
    
    # Summary
    print()
    log("="*80)
    log("Batch Preprocessing Complete!")
    log("="*80)
    
    success = sum(1 for r in results if r["status"] == "success")
    log(f"Success: {success}/{total} cases")
    
    if success < total:
        errors = [r for r in results if r["status"] == "error"]
        log(f"\nErrors ({len(errors)}):")
        for r in errors[:10]:
            log(f"  - {r['case']}: {r.get('reason', 'unknown')}")
        if len(errors) > 10:
            log(f"  ... and {len(errors)-10} more")
    
    # Save results
    results_file = work_dir / "batch_preprocessing_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved: {results_file}")
    log(f"Preprocessed data: {preproc_dir}")
    
    print()
    print("Next steps:")
    print("  1. Create train/val/test splits (scripts/make_splits.py)")
    print("  2. Verify data loading (src/data/fusion_dataset.py)")
    print("  3. Start training (train.py)")

if __name__ == "__main__":
    main()
