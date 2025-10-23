#!/usr/bin/env python3
"""
Create deterministic train/val/test splits for APIS dataset
70% train / 15% val / 15% test
"""
import json
from pathlib import Path
import random

print("="*80)
print("APIS Dataset Split Generation")
print("="*80)
print()

# Paths
project_root = Path(__file__).parent.parent
preproc_dir = project_root / "data/apis/preproc"
splits_file = project_root / "data/apis/splits.json"

# Get all preprocessed cases
all_cases = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir()])
total = len(all_cases)

print(f"Found {total} preprocessed cases")
print()

# Split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

train_size = int(total * train_ratio)
val_size = int(total * val_ratio)
test_size = total - train_size - val_size  # Ensure all cases are used

# Deterministic shuffle
random.seed(42)  # For reproducibility
shuffled = all_cases.copy()
random.shuffle(shuffled)

# Split
train_cases = sorted(shuffled[:train_size])
val_cases = sorted(shuffled[train_size:train_size+val_size])
test_cases = sorted(shuffled[train_size+val_size:])

splits = {
    "train": train_cases,
    "val": val_cases,
    "test": test_cases,
    "metadata": {
        "total_cases": total,
        "train_size": len(train_cases),
        "val_size": len(val_cases),
        "test_size": len(test_cases),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "random_seed": 42
    }
}

# Save splits
with open(splits_file, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Split distribution:")
print(f"  Train: {len(train_cases)} cases ({100*len(train_cases)/total:.1f}%)")
print(f"  Val:   {len(val_cases)} cases ({100*len(val_cases)/total:.1f}%)")
print(f"  Test:  {len(test_cases)} cases ({100*len(test_cases)/total:.1f}%)")
print()

# Show first few cases from each split
print("Sample cases:")
print(f"  Train: {', '.join(train_cases[:5])} ...")
print(f"  Val:   {', '.join(val_cases[:3])} ...")
print(f"  Test:  {', '.join(test_cases[:3])} ...")
print()

print(f"✓ Splits saved to: {splits_file}")
print()
print("Next step: Verify data loading with src/data/fusion_dataset.py")
