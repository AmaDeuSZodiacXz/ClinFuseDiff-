#!/usr/bin/env python3
"""
Update splits metadata from current preproc directory (54 lesion cases only).
"""

import json
import random
from pathlib import Path

def main():
    preproc_dir = Path("data/apis/preproc")
    splits_dir = Path("data/apis/splits")

    # Get all cases in preproc
    all_cases = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir() and d.name.startswith('train_')])

    print(f"Found {len(all_cases)} cases in preproc/")
    print(f"Cases: {all_cases[:5]}... (showing first 5)")

    # Set seed for reproducibility
    random.seed(42)
    shuffled_cases = all_cases.copy()
    random.shuffle(shuffled_cases)

    # Split: 70% train, 15% val, 15% test
    n_total = len(shuffled_cases)
    n_train = int(0.70 * n_total)  # 37
    n_val = int(0.15 * n_total)    # 8

    train_cases = sorted(shuffled_cases[:n_train])
    val_cases = sorted(shuffled_cases[n_train:n_train + n_val])
    test_cases = sorted(shuffled_cases[n_train + n_val:])

    print(f"\nSplits:")
    print(f"  Train: {len(train_cases)} cases")
    print(f"  Val:   {len(val_cases)} cases")
    print(f"  Test:  {len(test_cases)} cases")
    print(f"  Total: {len(all_cases)} cases")

    # Write split files
    splits_dir.mkdir(parents=True, exist_ok=True)

    (splits_dir / "train.txt").write_text('\n'.join(train_cases) + '\n')
    (splits_dir / "val.txt").write_text('\n'.join(val_cases) + '\n')
    (splits_dir / "test.txt").write_text('\n'.join(test_cases) + '\n')

    # Create metadata
    metadata = {
        "total_cases": n_total,
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "seed": 42,
        "split_ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15
        },
        "note": "Lesion cases only (54 cases). 6 non-lesion cases excluded."
    }

    with open(splits_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Splits updated in {splits_dir}/")
    print(f"  - train.txt ({len(train_cases)} cases)")
    print(f"  - val.txt ({len(val_cases)} cases)")
    print(f"  - test.txt ({len(test_cases)} cases)")
    print(f"  - split_metadata.json")

    # Verify no non-lesion cases
    NO_LESION = ['train_027', 'train_038', 'train_048', 'train_051', 'train_058', 'train_059']
    all_split_cases = train_cases + val_cases + test_cases
    found_excluded = [c for c in NO_LESION if c in all_split_cases]

    if found_excluded:
        print(f"\n⚠ WARNING: Found excluded cases in splits: {found_excluded}")
    else:
        print(f"\n✓ Verification passed: No non-lesion cases in splits")

if __name__ == '__main__':
    main()