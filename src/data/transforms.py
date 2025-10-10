"""Data augmentation transforms for medical imaging"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import torchvision.transforms.functional as TF


class RandomFlip3D:
    """Randomly flip 3D image along specified axes"""

    def __init__(self, axes: List[int] = [0, 1, 2], p: float = 0.5):
        """
        Args:
            axes: list of axes to potentially flip (0, 1, 2 for z, y, x)
            p: probability of flipping each axis
        """
        self.axes = axes
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (C, D, H, W) tensor
        Returns:
            flipped image
        """
        for axis in self.axes:
            if np.random.rand() < self.p:
                image = torch.flip(image, dims=[axis + 1])  # +1 for channel dim
        return image


class RandomRotation3D:
    """Random 3D rotation in axial plane"""

    def __init__(self, max_angle: float = 15, p: float = 0.5):
        """
        Args:
            max_angle: maximum rotation angle in degrees
            p: probability of applying rotation
        """
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (C, D, H, W) tensor
        Returns:
            rotated image
        """
        if np.random.rand() > self.p:
            return image

        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Rotate each axial slice
        C, D, H, W = image.shape
        rotated = torch.zeros_like(image)

        for d in range(D):
            for c in range(C):
                rotated[c, d] = TF.rotate(image[c, d], angle, fill=0)

        return rotated


class RandomAffine3D:
    """Random affine transformation for 3D images"""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 10,
        translation_range: float = 0.1,
        p: float = 0.5
    ):
        """
        Args:
            scale_range: min and max scaling factors
            rotation_range: max rotation in degrees
            translation_range: max translation as fraction of image size
            p: probability of applying transform
        """
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random affine transformation"""
        if np.random.rand() > self.p:
            return image

        from scipy.ndimage import affine_transform

        # Generate random parameters
        scale = np.random.uniform(*self.scale_range)
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        translation = [
            np.random.uniform(-self.translation_range, self.translation_range) * s
            for s in image.shape[1:]
        ]

        # Create affine matrix
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 3D affine matrix (rotation around z-axis + scaling + translation)
        matrix = np.array([
            [scale * cos_theta, -scale * sin_theta, 0, translation[2]],
            [scale * sin_theta, scale * cos_theta, 0, translation[1]],
            [0, 0, scale, translation[0]],
            [0, 0, 0, 1]
        ])

        # Apply to each channel
        transformed = torch.zeros_like(image)
        for c in range(image.shape[0]):
            transformed[c] = torch.from_numpy(
                affine_transform(
                    image[c].numpy(),
                    matrix[:3, :3],
                    offset=matrix[:3, 3],
                    order=1,
                    mode='constant',
                    cval=0
                )
            )

        return transformed


class RandomNoise:
    """Add random Gaussian noise"""

    def __init__(self, std_range: Tuple[float, float] = (0, 0.1), p: float = 0.3):
        """
        Args:
            std_range: range of noise standard deviation
            p: probability of adding noise
        """
        self.std_range = std_range
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to image"""
        if np.random.rand() > self.p:
            return image

        std = np.random.uniform(*self.std_range)
        noise = torch.randn_like(image) * std
        return image + noise


class RandomGammaCorrection:
    """Random gamma correction for intensity augmentation"""

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        """
        Args:
            gamma_range: range of gamma values
            p: probability of applying gamma correction
        """
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction"""
        if np.random.rand() > self.p:
            return image

        gamma = np.random.uniform(*self.gamma_range)

        # Normalize to [0, 1]
        img_min = image.min()
        img_max = image.max()
        image_norm = (image - img_min) / (img_max - img_min + 1e-8)

        # Apply gamma
        image_corrected = torch.pow(image_norm, gamma)

        # Denormalize
        return image_corrected * (img_max - img_min) + img_min


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment"""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.3
    ):
        """
        Args:
            brightness_range: range for brightness multiplier
            contrast_range: range for contrast multiplier
            p: probability of applying transform
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Adjust brightness and contrast"""
        if np.random.rand() > self.p:
            return image

        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)

        # Adjust contrast around mean
        mean = image.mean()
        image = (image - mean) * contrast + mean

        # Adjust brightness
        image = image * brightness

        return image


class RandomElasticDeformation:
    """Random elastic deformation for medical images"""

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0, 30),
        sigma: float = 5,
        p: float = 0.2
    ):
        """
        Args:
            alpha_range: range for deformation strength
            sigma: smoothness of deformation
            p: probability of applying deformation
        """
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation"""
        if np.random.rand() > self.p:
            return image

        from scipy.ndimage import gaussian_filter, map_coordinates

        alpha = np.random.uniform(*self.alpha_range)
        shape = image.shape[1:]  # Exclude channel dimension

        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * alpha

        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * alpha

        dz = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * alpha

        # Create meshgrid
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )

        indices = [
            np.reshape(z + dz, (-1,)),
            np.reshape(y + dy, (-1,)),
            np.reshape(x + dx, (-1,))
        ]

        # Apply deformation to each channel
        deformed = torch.zeros_like(image)
        for c in range(image.shape[0]):
            deformed_c = map_coordinates(
                image[c].numpy(),
                indices,
                order=1,
                mode='reflect'
            ).reshape(shape)
            deformed[c] = torch.from_numpy(deformed_c)

        return deformed


class MultiModalCompose:
    """Compose transforms for multimodal data"""

    def __init__(self, transforms: List):
        """
        Args:
            transforms: list of transform functions
        """
        self.transforms = transforms

    def __call__(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to all imaging modalities

        Args:
            modality_data: dict of {modality_name: tensor}
        Returns:
            transformed modality data
        """
        # Apply same transform to all imaging modalities
        # (to maintain spatial correspondence)

        imaging_modalities = {
            k: v for k, v in modality_data.items()
            if v is not None and k != 'clinical'
        }

        if len(imaging_modalities) == 0:
            return modality_data

        # Generate random state for consistent transforms
        random_state = np.random.get_state()

        transformed_data = {}

        for modality, data in modality_data.items():
            if modality == 'clinical' or data is None:
                transformed_data[modality] = data
            else:
                # Reset random state for consistent transforms across modalities
                np.random.set_state(random_state)
                torch.manual_seed(int(random_state[1][0]))

                transformed = data
                for transform in self.transforms:
                    transformed = transform(transformed)

                transformed_data[modality] = transformed

        return transformed_data


def get_train_transforms(
    use_flip: bool = True,
    use_rotation: bool = True,
    use_noise: bool = True,
    use_gamma: bool = True,
    use_brightness_contrast: bool = True,
    use_elastic: bool = False
) -> MultiModalCompose:
    """
    Get training transforms

    Args:
        use_flip: use random flipping
        use_rotation: use random rotation
        use_noise: use random noise
        use_gamma: use gamma correction
        use_brightness_contrast: use brightness/contrast adjustment
        use_elastic: use elastic deformation (slow)

    Returns:
        MultiModalCompose transform
    """
    transforms = []

    if use_flip:
        transforms.append(RandomFlip3D(axes=[1, 2], p=0.5))  # Flip H and W

    if use_rotation:
        transforms.append(RandomRotation3D(max_angle=15, p=0.3))

    if use_noise:
        transforms.append(RandomNoise(std_range=(0, 0.05), p=0.3))

    if use_gamma:
        transforms.append(RandomGammaCorrection(gamma_range=(0.8, 1.2), p=0.3))

    if use_brightness_contrast:
        transforms.append(
            RandomBrightnessContrast(
                brightness_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                p=0.3
            )
        )

    if use_elastic:
        transforms.append(
            RandomElasticDeformation(alpha_range=(0, 20), sigma=5, p=0.2)
        )

    return MultiModalCompose(transforms)


def get_val_transforms() -> MultiModalCompose:
    """Get validation transforms (no augmentation)"""
    return MultiModalCompose([])
