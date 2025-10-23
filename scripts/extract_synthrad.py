#!/usr/bin/env python3
"""
Extract and organize SynthRAD2023 dataset for registration robustness testing
"""
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

print("="*80)
print("SynthRAD2023 Dataset Extraction & Organization")
print("="*80)
print()

# Paths
project_root = Path('/mnt/c/Users/User/Documents/ClinFuseDiff')
synthrad_raw = project_root / 'SynthRAD_data'
synthrad_organized = project_root / 'data' / 'synthrad' / 'raw'

# Create organized directory
synthrad_organized.mkdir(parents=True, exist_ok=True)

print(f"Source: {synthrad_raw}")
print(f"Target: {synthrad_organized}")
print()

# Extract training data
train_zip = synthrad_raw / 'training' / 'Task1.zip?download=1'
if train_zip.exists():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting training data (14GB)...")
    print(f"  This may take 5-10 minutes...")
    
    extract_dir = synthrad_raw / 'training' / 'extracted'
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        # Get total files
        total = len(zip_ref.namelist())
        print(f"  Total files: {total}")
        
        # Extract with progress
        for i, member in enumerate(zip_ref.namelist(), 1):
            zip_ref.extract(member, extract_dir)
            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({100*i//total}%)", end='\r')
        print()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Training data extracted")
else:
    print(f"✗ Training zip not found: {train_zip}")

print()
print("="*80)
print("Next: Organize into paired CT-MRI structure")
print("="*80)
