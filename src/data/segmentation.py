"""Segmentation utilities using TotalSegmentator for brain MRI/CT"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, List
import numpy as np
import nibabel as nib
import SimpleITK as sitk


class TotalSegmentatorWrapper:
    """Wrapper for TotalSegmentator to segment brain structures"""

    def __init__(
        self,
        device: str = "gpu",
        fast: bool = False,
        task: str = "total",  # or "total_mr" for MRI
        ml: bool = True,  # multilabel output
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        quiet: bool = False
    ):
        """
        Args:
            device: 'gpu' or 'cpu'
            fast: use faster but less accurate model
            task: segmentation task ('total' for CT, 'total_mr' for MRI, 'brain_structures')
            ml: output multilabel segmentation (single file with all structures)
            nr_thr_resamp: number of threads for resampling
            nr_thr_saving: number of threads for saving
            quiet: suppress output
        """
        self.device = device
        self.fast = fast
        self.task = task
        self.ml = ml
        self.nr_thr_resamp = nr_thr_resamp
        self.nr_thr_saving = nr_thr_saving
        self.quiet = quiet

        # Check if TotalSegmentator is installed
        try:
            import totalsegmentator
            self.use_python_api = True
        except ImportError:
            self.use_python_api = False
            print("Warning: TotalSegmentator Python API not found. Using command-line interface.")

    def segment(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        roi_subset: Optional[List[str]] = None
    ) -> Union[str, np.ndarray]:
        """
        Segment anatomical structures from CT/MRI image

        Args:
            input_path: path to input NIfTI file (.nii or .nii.gz)
            output_path: path to output directory or file
            roi_subset: list of specific ROIs to segment (e.g., ['brain', 'skull'])

        Returns:
            output_path or segmentation mask array
        """
        input_path = Path(input_path)

        # Create temporary output if not specified
        if output_path is None:
            output_dir = tempfile.mkdtemp()
            output_path = Path(output_dir) / "segmentation.nii.gz"
        else:
            output_path = Path(output_path)

        if self.use_python_api:
            return self._segment_python_api(input_path, output_path, roi_subset)
        else:
            return self._segment_cli(input_path, output_path, roi_subset)

    def _segment_python_api(
        self,
        input_path: Path,
        output_path: Path,
        roi_subset: Optional[List[str]] = None
    ) -> str:
        """Segment using Python API"""
        from totalsegmentator.python_api import totalsegmentator

        # Prepare arguments
        kwargs = {
            'task': self.task,
            'ml': self.ml,
            'device': self.device,
            'fast': self.fast,
            'nr_thr_resamp': self.nr_thr_resamp,
            'nr_thr_saving': self.nr_thr_saving,
            'quiet': self.quiet
        }

        if roi_subset is not None:
            kwargs['roi_subset'] = roi_subset

        # Run segmentation
        totalsegmentator(
            input=str(input_path),
            output=str(output_path),
            **kwargs
        )

        return str(output_path)

    def _segment_cli(
        self,
        input_path: Path,
        output_path: Path,
        roi_subset: Optional[List[str]] = None
    ) -> str:
        """Segment using command-line interface"""
        cmd = [
            'TotalSegmentator',
            '-i', str(input_path),
            '-o', str(output_path),
            '--task', self.task,
            '--device', self.device
        ]

        if self.fast:
            cmd.append('--fast')

        if self.ml:
            cmd.append('--ml')

        if self.quiet:
            cmd.append('--quiet')

        if roi_subset is not None:
            cmd.extend(['--roi_subset'] + roi_subset)

        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed: {result.stderr}")

        return str(output_path)

    def load_segmentation(self, seg_path: Union[str, Path]) -> np.ndarray:
        """Load segmentation mask from file"""
        seg_path = Path(seg_path)

        if seg_path.is_dir():
            # If directory, look for multilabel file
            ml_file = seg_path / "segmentations.nii.gz"
            if ml_file.exists():
                seg_path = ml_file
            else:
                raise ValueError(f"No multilabel segmentation found in {seg_path}")

        # Load with nibabel
        seg_nii = nib.load(str(seg_path))
        seg_data = seg_nii.get_fdata()

        return seg_data


class BrainSegmentationPreprocessor:
    """Preprocessor for brain MRI/CT with segmentation"""

    def __init__(
        self,
        segmentator: Optional[TotalSegmentatorWrapper] = None,
        target_spacing: tuple = (1.0, 1.0, 1.0),
        target_size: tuple = (128, 128, 128),
        normalize: bool = True,
        use_skull_stripping: bool = False,
        brain_roi_names: Optional[List[str]] = None
    ):
        """
        Args:
            segmentator: TotalSegmentator wrapper instance
            target_spacing: target voxel spacing in mm
            target_size: target image size
            normalize: apply intensity normalization
            use_skull_stripping: remove skull using segmentation
            brain_roi_names: specific brain ROI names to extract
        """
        if segmentator is None:
            self.segmentator = TotalSegmentatorWrapper(task='brain_structures')
        else:
            self.segmentator = segmentator

        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize = normalize
        self.use_skull_stripping = use_skull_stripping
        self.brain_roi_names = brain_roi_names

    def preprocess(
        self,
        image_path: Union[str, Path],
        return_segmentation: bool = True,
        cache_dir: Optional[Path] = None
    ) -> dict:
        """
        Preprocess brain image with optional segmentation

        Args:
            image_path: path to input NIfTI image
            return_segmentation: whether to return segmentation mask
            cache_dir: directory to cache segmentation results

        Returns:
            dict with 'image', 'segmentation', 'metadata'
        """
        image_path = Path(image_path)

        # Load image
        image_sitk = sitk.ReadImage(str(image_path))
        image_array = sitk.GetArrayFromImage(image_sitk)
        original_spacing = image_sitk.GetSpacing()
        original_size = image_sitk.GetSize()

        # Segment if needed
        segmentation = None
        if return_segmentation or self.use_skull_stripping:
            # Check cache
            if cache_dir is not None:
                cache_dir = Path(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                seg_cache_path = cache_dir / f"{image_path.stem}_seg.nii.gz"

                if seg_cache_path.exists():
                    segmentation = self.segmentator.load_segmentation(seg_cache_path)
                else:
                    seg_path = self.segmentator.segment(
                        image_path,
                        seg_cache_path,
                        roi_subset=self.brain_roi_names
                    )
                    segmentation = self.segmentator.load_segmentation(seg_path)
            else:
                seg_path = self.segmentator.segment(
                    image_path,
                    roi_subset=self.brain_roi_names
                )
                segmentation = self.segmentator.load_segmentation(seg_path)

        # Skull stripping
        if self.use_skull_stripping and segmentation is not None:
            brain_mask = segmentation > 0
            image_array = image_array * brain_mask

        # Convert to SimpleITK for resampling
        image_sitk = sitk.GetImageFromArray(image_array)
        image_sitk.SetSpacing(original_spacing)

        # Resample to target spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(self.target_spacing)

        # Calculate new size
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]
        resampler.SetSize(new_size)

        image_resampled = resampler.Execute(image_sitk)
        image_array = sitk.GetArrayFromImage(image_resampled)

        # Resize to target size
        image_array = self._resize_3d(image_array, self.target_size)

        # Normalize intensity
        if self.normalize:
            image_array = self._normalize_intensity(image_array)

        # Resample segmentation if provided
        if segmentation is not None:
            seg_sitk = sitk.GetImageFromArray(segmentation.astype(np.float32))
            seg_sitk.SetSpacing(original_spacing)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            seg_resampled = resampler.Execute(seg_sitk)
            segmentation = sitk.GetArrayFromImage(seg_resampled)
            segmentation = self._resize_3d(segmentation, self.target_size)

        return {
            'image': image_array.astype(np.float32),
            'segmentation': segmentation.astype(np.int32) if segmentation is not None else None,
            'metadata': {
                'original_spacing': original_spacing,
                'original_size': original_size,
                'target_spacing': self.target_spacing,
                'target_size': self.target_size
            }
        }

    @staticmethod
    def _resize_3d(array: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize 3D array to target size using SimpleITK"""
        image = sitk.GetImageFromArray(array)

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetInterpolator(sitk.sitkLinear)

        # Calculate output spacing
        input_size = image.GetSize()
        input_spacing = image.GetSpacing()
        output_spacing = [
            input_spacing[i] * (input_size[i] / target_size[i])
            for i in range(3)
        ]
        resampler.SetOutputSpacing(output_spacing)

        resized = resampler.Execute(image)
        return sitk.GetArrayFromImage(resized)

    @staticmethod
    def _normalize_intensity(array: np.ndarray, percentiles: tuple = (1, 99)) -> np.ndarray:
        """Normalize intensity using percentile-based clipping"""
        p_low, p_high = np.percentile(array[array > 0], percentiles)
        array = np.clip(array, p_low, p_high)
        array = (array - p_low) / (p_high - p_low + 1e-8)
        return array


def extract_brain_roi_features(
    segmentation: np.ndarray,
    image: np.ndarray,
    roi_id: int
) -> dict:
    """
    Extract features from specific brain ROI

    Args:
        segmentation: segmentation mask with ROI labels
        image: intensity image
        roi_id: ROI label ID

    Returns:
        dict of features (volume, mean_intensity, std_intensity, etc.)
    """
    roi_mask = segmentation == roi_id

    if not roi_mask.any():
        return {
            'volume': 0,
            'mean_intensity': 0,
            'std_intensity': 0,
            'max_intensity': 0,
            'min_intensity': 0
        }

    roi_voxels = image[roi_mask]

    features = {
        'volume': np.sum(roi_mask),
        'mean_intensity': np.mean(roi_voxels),
        'std_intensity': np.std(roi_voxels),
        'max_intensity': np.max(roi_voxels),
        'min_intensity': np.min(roi_voxels)
    }

    return features
