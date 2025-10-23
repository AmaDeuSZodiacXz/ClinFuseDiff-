"""
Analyze lesion presence in APIS preprocessed dataset

Check which cases have lesion masks and compute lesion statistics
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm


def analyze_lesion_presence(preproc_dir):
    """Analyze lesion mask presence and statistics"""

    preproc_dir = Path(preproc_dir)

    results = []

    # Get all case directories
    case_dirs = sorted([d for d in preproc_dir.iterdir() if d.is_dir()])

    print(f"Analyzing {len(case_dirs)} cases...")
    print()

    for case_dir in tqdm(case_dirs):
        case_id = case_dir.name

        # Check file existence
        ct_path = case_dir / "ct.nii.gz"
        mri_path = case_dir / "mri_to_ct.nii.gz"
        brain_mask_path = case_dir / "brain_mask.nii.gz"
        bone_mask_path = case_dir / "bone_mask.nii.gz"
        lesion_mask_path = case_dir / "lesion_mask.nii.gz"

        result = {
            'case_id': case_id,
            'ct_exists': ct_path.exists(),
            'mri_exists': mri_path.exists(),
            'brain_mask_exists': brain_mask_path.exists(),
            'bone_mask_exists': bone_mask_path.exists(),
            'lesion_mask_exists': lesion_mask_path.exists(),
            'has_lesion': False,
            'lesion_volume_ml': 0.0,
            'lesion_voxels': 0,
            'brain_voxels': 0,
            'bone_voxels': 0
        }

        # Load and analyze lesion mask
        if lesion_mask_path.exists():
            try:
                lesion_nii = nib.load(lesion_mask_path)
                lesion_data = lesion_nii.get_fdata()

                # Count lesion voxels
                lesion_voxels = (lesion_data > 0).sum()

                if lesion_voxels > 0:
                    result['has_lesion'] = True
                    result['lesion_voxels'] = int(lesion_voxels)

                    # Compute volume (mm³ → mL)
                    voxel_size = np.prod(lesion_nii.header.get_zooms())  # mm³
                    lesion_volume_mm3 = lesion_voxels * voxel_size
                    lesion_volume_ml = lesion_volume_mm3 / 1000.0

                    result['lesion_volume_ml'] = round(lesion_volume_ml, 3)

            except Exception as e:
                print(f"Warning: Error loading lesion mask for {case_id}: {e}")

        # Load brain mask stats
        if brain_mask_path.exists():
            try:
                brain_nii = nib.load(brain_mask_path)
                brain_data = brain_nii.get_fdata()
                result['brain_voxels'] = int((brain_data > 0).sum())
            except Exception as e:
                print(f"Warning: Error loading brain mask for {case_id}: {e}")

        # Load bone mask stats
        if bone_mask_path.exists():
            try:
                bone_nii = nib.load(bone_mask_path)
                bone_data = bone_nii.get_fdata()
                result['bone_voxels'] = int((bone_data > 0).sum())
            except Exception as e:
                print(f"Warning: Error loading bone mask for {case_id}: {e}")

        results.append(result)

    return results


def print_summary(results):
    """Print summary statistics"""

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("LESION PRESENCE ANALYSIS")
    print("="*80)

    # Overall statistics
    total_cases = len(df)
    cases_with_lesion = df['has_lesion'].sum()
    cases_without_lesion = total_cases - cases_with_lesion

    print(f"\nTotal Cases: {total_cases}")
    print(f"  With Lesion: {cases_with_lesion} ({cases_with_lesion/total_cases*100:.1f}%)")
    print(f"  Without Lesion: {cases_without_lesion} ({cases_without_lesion/total_cases*100:.1f}%)")

    # File existence
    print(f"\nFile Completeness:")
    print(f"  CT exists: {df['ct_exists'].sum()}/{total_cases}")
    print(f"  MRI exists: {df['mri_exists'].sum()}/{total_cases}")
    print(f"  Brain mask exists: {df['brain_mask_exists'].sum()}/{total_cases}")
    print(f"  Bone mask exists: {df['bone_mask_exists'].sum()}/{total_cases}")
    print(f"  Lesion mask file exists: {df['lesion_mask_exists'].sum()}/{total_cases}")

    # Lesion volume statistics
    if cases_with_lesion > 0:
        lesion_cases = df[df['has_lesion']]

        print(f"\nLesion Volume Statistics (n={cases_with_lesion}):")
        print(f"  Mean: {lesion_cases['lesion_volume_ml'].mean():.2f} mL")
        print(f"  Median: {lesion_cases['lesion_volume_ml'].median():.2f} mL")
        print(f"  Std: {lesion_cases['lesion_volume_ml'].std():.2f} mL")
        print(f"  Min: {lesion_cases['lesion_volume_ml'].min():.3f} mL")
        print(f"  Max: {lesion_cases['lesion_volume_ml'].max():.2f} mL")

        # Lesion size categories
        small_lesions = (lesion_cases['lesion_volume_ml'] < 5).sum()
        medium_lesions = ((lesion_cases['lesion_volume_ml'] >= 5) &
                         (lesion_cases['lesion_volume_ml'] < 20)).sum()
        large_lesions = (lesion_cases['lesion_volume_ml'] >= 20).sum()

        print(f"\nLesion Size Distribution:")
        print(f"  Small (<5 mL): {small_lesions} ({small_lesions/cases_with_lesion*100:.1f}%)")
        print(f"  Medium (5-20 mL): {medium_lesions} ({medium_lesions/cases_with_lesion*100:.1f}%)")
        print(f"  Large (≥20 mL): {large_lesions} ({large_lesions/cases_with_lesion*100:.1f}%)")

    # ROI mask statistics
    if df['brain_voxels'].sum() > 0:
        print(f"\nBrain ROI Statistics:")
        print(f"  Mean voxels: {df[df['brain_voxels']>0]['brain_voxels'].mean():.0f}")
        print(f"  Median voxels: {df[df['brain_voxels']>0]['brain_voxels'].median():.0f}")

    if df['bone_voxels'].sum() > 0:
        print(f"\nBone ROI Statistics:")
        print(f"  Mean voxels: {df[df['bone_voxels']>0]['bone_voxels'].mean():.0f}")
        print(f"  Median voxels: {df[df['bone_voxels']>0]['bone_voxels'].median():.0f}")

    # Cases without lesions
    if cases_without_lesion > 0:
        print(f"\nCases WITHOUT Lesions ({cases_without_lesion} cases):")
        no_lesion = df[~df['has_lesion']]
        for idx, row in no_lesion.iterrows():
            status = "file missing" if not row['lesion_mask_exists'] else "empty mask"
            print(f"  - {row['case_id']}: {status}")

    print("\n" + "="*80)

    return df


def save_results(df, output_dir):
    """Save results to files"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / "lesion_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Save JSON
    json_path = output_dir / "lesion_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)
    print(f"✓ Saved JSON: {json_path}")

    # Save summary statistics
    summary = {
        'total_cases': len(df),
        'cases_with_lesion': int(df['has_lesion'].sum()),
        'cases_without_lesion': int((~df['has_lesion']).sum()),
        'lesion_volume_stats': {
            'mean_ml': float(df[df['has_lesion']]['lesion_volume_ml'].mean()) if df['has_lesion'].any() else 0,
            'median_ml': float(df[df['has_lesion']]['lesion_volume_ml'].median()) if df['has_lesion'].any() else 0,
            'std_ml': float(df[df['has_lesion']]['lesion_volume_ml'].std()) if df['has_lesion'].any() else 0,
            'min_ml': float(df[df['has_lesion']]['lesion_volume_ml'].min()) if df['has_lesion'].any() else 0,
            'max_ml': float(df[df['has_lesion']]['lesion_volume_ml'].max()) if df['has_lesion'].any() else 0,
        },
        'size_distribution': {
            'small_count': int((df[df['has_lesion']]['lesion_volume_ml'] < 5).sum()) if df['has_lesion'].any() else 0,
            'medium_count': int(((df[df['has_lesion']]['lesion_volume_ml'] >= 5) &
                                (df[df['has_lesion']]['lesion_volume_ml'] < 20)).sum()) if df['has_lesion'].any() else 0,
            'large_count': int((df[df['has_lesion']]['lesion_volume_ml'] >= 20).sum()) if df['has_lesion'].any() else 0,
        },
        'cases_without_lesions': df[~df['has_lesion']]['case_id'].tolist()
    }

    summary_path = output_dir / "lesion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze lesion presence in APIS dataset')
    parser.add_argument('--preproc-dir', type=str,
                       default='data/apis/preproc',
                       help='Preprocessed data directory')
    parser.add_argument('--output-dir', type=str,
                       default='eda_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Analyze
    results = analyze_lesion_presence(args.preproc_dir)

    # Print summary
    df = print_summary(results)

    # Save results
    save_results(df, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()