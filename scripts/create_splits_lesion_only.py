"""
Create train/val/test splits excluding cases without lesions

Based on lesion analysis results, remove 6 cases with no lesions:
- train_027, train_038, train_048, train_051, train_058, train_059

Final dataset: 54 cases (all with lesions)
Split: 70% train, 15% val, 15% test
"""

import json
import random
from pathlib import Path
import argparse


# Cases without lesions (from lesion analysis)
NO_LESION_CASES = [
    'train_027',
    'train_038',
    'train_048',
    'train_051',
    'train_058',
    'train_059'
]


def create_splits_lesion_only(preproc_dir, splits_dir, seed=42):
    """Create train/val/test splits with only lesion cases"""

    preproc_dir = Path(preproc_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Get all case IDs
    all_cases = sorted([d.name for d in preproc_dir.iterdir() if d.is_dir()])

    print(f"Total cases found: {len(all_cases)}")
    print(f"Cases without lesions: {len(NO_LESION_CASES)}")

    # Filter out cases without lesions
    lesion_cases = [c for c in all_cases if c not in NO_LESION_CASES]

    print(f"Cases with lesions: {len(lesion_cases)}")
    print(f"\nExcluded cases:")
    for case in NO_LESION_CASES:
        if case in all_cases:
            print(f"  - {case}")
        else:
            print(f"  - {case} (not found in preprocessed data)")

    # Set seed for reproducibility
    random.seed(seed)
    random.shuffle(lesion_cases)

    # Split ratios
    n_total = len(lesion_cases)
    n_train = int(0.70 * n_total)  # 70%
    n_val = int(0.15 * n_total)    # 15%
    # n_test = remaining             # 15%

    train_cases = lesion_cases[:n_train]
    val_cases = lesion_cases[n_train:n_train + n_val]
    test_cases = lesion_cases[n_train + n_val:]

    print(f"\nSplit Statistics:")
    print(f"  Train: {len(train_cases)} ({len(train_cases)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_cases)} ({len(val_cases)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_cases)} ({len(test_cases)/n_total*100:.1f}%)")
    print(f"  Total: {n_total}")

    # Save splits
    splits = {
        'train': train_cases,
        'val': val_cases,
        'test': test_cases
    }

    for split_name, cases in splits.items():
        # Save as text file (one case per line)
        txt_path = splits_dir / f'{split_name}.txt'
        with open(txt_path, 'w') as f:
            for case in cases:
                f.write(f"{case}\n")
        print(f"\n✓ Saved {split_name}: {txt_path} ({len(cases)} cases)")

        # Print first few cases
        print(f"  First 3: {', '.join(cases[:3])}")
        if len(cases) > 3:
            print(f"  Last 3:  {', '.join(cases[-3:])}")

    # Save metadata
    metadata = {
        'total_cases': n_total,
        'excluded_cases': NO_LESION_CASES,
        'splits': {
            'train': {
                'count': len(train_cases),
                'percentage': round(len(train_cases)/n_total*100, 1)
            },
            'val': {
                'count': len(val_cases),
                'percentage': round(len(val_cases)/n_total*100, 1)
            },
            'test': {
                'count': len(test_cases),
                'percentage': round(len(test_cases)/n_total*100, 1)
            }
        },
        'seed': seed,
        'split_ratios': [0.70, 0.15, 0.15],
        'note': 'All cases have lesions (no-lesion cases excluded)'
    }

    metadata_path = splits_dir / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path}")

    return splits, metadata


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits (lesion cases only)')
    parser.add_argument('--preproc-dir', type=str,
                       default='data/apis/preproc',
                       help='Preprocessed data directory')
    parser.add_argument('--splits-dir', type=str,
                       default='data/apis/splits',
                       help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    print("="*80)
    print("CREATE TRAIN/VAL/TEST SPLITS (LESION CASES ONLY)")
    print("="*80)
    print()

    splits, metadata = create_splits_lesion_only(
        args.preproc_dir,
        args.splits_dir,
        args.seed
    )

    print()
    print("="*80)
    print("SPLIT CREATION COMPLETE")
    print("="*80)
    print(f"\nFinal dataset: {metadata['total_cases']} cases (all with lesions)")
    print(f"Excluded: {len(metadata['excluded_cases'])} cases without lesions")
    print(f"\nSplit files saved to: {args.splits_dir}")


if __name__ == '__main__':
    main()