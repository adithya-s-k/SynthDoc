"""
Augmentation module for SynthDoc - applies various transformations to documents and images.

This module provides two main approaches:
1. Dataset-level augmentation: Takes a dataset and applies augmentations on top of it
2. Direct image augmentation: Can be fed directly into dataset generators to perform augmentation on images

Both approaches take a list of augmentations that can be applied.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

from .models import AugmentationType

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for individual augmentation parameters."""

    probability: float = 0.5  # Probability of applying this augmentation
    intensity: float = 1.0  # Intensity/strength of the augmentation
    parameters: Dict[str, Any] = (
        None  # Additional parameters specific to each augmentation
    )

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BaseAugmentation:
    """Base class for all augmentation operations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def should_apply(self) -> bool:
        """Determine if augmentation should be applied based on probability."""
        return random.random() < self.config.probability

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply the augmentation to an image."""
        raise NotImplementedError("Subclasses must implement apply method")


class RotationAugmentation(BaseAugmentation):
    """Applies rotation to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        max_angle = self.config.parameters.get("max_angle", 15)
        angle = random.uniform(-max_angle, max_angle) * self.config.intensity

        # Rotate with white background fill
        rotated = image.rotate(angle, expand=True, fillcolor="white")
        logger.debug(f"Applied rotation: {angle:.2f} degrees")
        return rotated


class ScalingAugmentation(BaseAugmentation):
    """Applies scaling to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        min_scale = self.config.parameters.get("min_scale", 0.8)
        max_scale = self.config.parameters.get("max_scale", 1.2)

        scale_factor = random.uniform(min_scale, max_scale)
        scale_factor = 1.0 + (scale_factor - 1.0) * self.config.intensity

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Applied scaling: {scale_factor:.2f}x")
        return scaled


class NoiseAugmentation(BaseAugmentation):
    """Adds noise to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        # Convert to numpy array
        img_array = np.array(image)

        noise_std = self.config.parameters.get("noise_std", 10)
        noise_std *= self.config.intensity

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        noisy_image = Image.fromarray(noisy_array)
        logger.debug(f"Applied noise: std={noise_std:.2f}")
        return noisy_image


class BlurAugmentation(BaseAugmentation):
    """Applies blur to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        blur_radius = self.config.parameters.get("blur_radius", 2.0)
        blur_radius *= self.config.intensity

        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        logger.debug(f"Applied blur: radius={blur_radius:.2f}")
        return blurred


class ColorShiftAugmentation(BaseAugmentation):
    """Applies color shifts to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        # Brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        brightness_factor = 1.0 + (brightness_factor - 1.0) * self.config.intensity

        # Contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        contrast_factor = 1.0 + (contrast_factor - 1.0) * self.config.intensity

        # Apply enhancements
        enhanced = ImageEnhance.Brightness(image).enhance(brightness_factor)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)

        logger.debug(
            f"Applied color shift: brightness={brightness_factor:.2f}, contrast={contrast_factor:.2f}"
        )
        return enhanced


class CroppingAugmentation(BaseAugmentation):
    """Applies random cropping to images."""

    def apply(self, image: Image.Image) -> Image.Image:
        if not self.should_apply():
            return image

        crop_ratio = self.config.parameters.get("crop_ratio", 0.1)
        crop_ratio *= self.config.intensity

        width, height = image.size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        # Random crop coordinates
        left = random.randint(0, crop_width)
        top = random.randint(0, crop_height)
        right = width - random.randint(0, crop_width)
        bottom = height - random.randint(0, crop_height)

        cropped = image.crop((left, top, right, bottom))
        logger.debug(f"Applied cropping: ({left}, {top}, {right}, {bottom})")
        return cropped


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations in sequence."""

    def __init__(self):
        self.augmentations: List[BaseAugmentation] = []

    def add_augmentation(self, augmentation: BaseAugmentation):
        """Add an augmentation to the pipeline."""
        self.augmentations.append(augmentation)

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply all augmentations in the pipeline."""
        result = image.copy()

        for augmentation in self.augmentations:
            try:
                result = augmentation.apply(result)
            except Exception as e:
                logger.warning(f"Failed to apply {type(augmentation).__name__}: {e}")
                # Continue with original image if augmentation fails
                continue

        return result


class DatasetAugmenter:
    """
    Takes a dataset and applies augmentations on top of it.

    This class processes entire datasets, applying augmentations to create
    augmented versions of the original data.
    """

    def __init__(
        self, augmentation_configs: Dict[AugmentationType, AugmentationConfig]
    ):
        self.augmentation_configs = augmentation_configs
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> AugmentationPipeline:
        """Build augmentation pipeline from configs."""
        pipeline = AugmentationPipeline()

        augmentation_map = {
            AugmentationType.ROTATION: RotationAugmentation,
            AugmentationType.SCALING: ScalingAugmentation,
            AugmentationType.NOISE: NoiseAugmentation,
            AugmentationType.BLUR: BlurAugmentation,
            AugmentationType.COLOR_SHIFT: ColorShiftAugmentation,
            AugmentationType.CROPPING: CroppingAugmentation,
        }

        for aug_type, config in self.augmentation_configs.items():
            if aug_type in augmentation_map:
                augmentation_class = augmentation_map[aug_type]
                pipeline.add_augmentation(augmentation_class(config))
            else:
                logger.warning(f"Unknown augmentation type: {aug_type}")

        return pipeline

    def augment_dataset(
        self, dataset: Dict[str, Any], num_augmentations: int = 1
    ) -> Dict[str, Any]:
        """
        Apply augmentations to a dataset.

        Args:
            dataset: Dictionary containing dataset with 'images' key
            num_augmentations: Number of augmented versions to create per image

        Returns:
            Augmented dataset with original and augmented samples
        """
        augmented_dataset = {"images": [], "metadata": [], "labels": []}

        # Copy original data
        original_images = dataset.get("images", [])
        original_metadata = dataset.get("metadata", [])
        original_labels = dataset.get("labels", [])

        # Add original samples
        augmented_dataset["images"].extend(original_images)
        augmented_dataset["metadata"].extend(original_metadata)
        augmented_dataset["labels"].extend(original_labels)

        # Generate augmented samples
        for i, image_path in enumerate(original_images):
            try:
                # Load image
                if isinstance(image_path, str):
                    image = Image.open(image_path)
                else:
                    image = image_path  # Assume it's already a PIL Image

                # Create augmented versions
                for aug_idx in range(num_augmentations):
                    augmented_image = self.pipeline.apply(image)

                    # Add to dataset
                    augmented_dataset["images"].append(augmented_image)

                    # Copy and modify metadata
                    if i < len(original_metadata):
                        aug_metadata = original_metadata[i].copy()
                        aug_metadata["is_augmented"] = True
                        aug_metadata["augmentation_id"] = aug_idx
                        aug_metadata["original_index"] = i
                        augmented_dataset["metadata"].append(aug_metadata)

                    # Copy labels
                    if i < len(original_labels):
                        augmented_dataset["labels"].append(original_labels[i])

            except Exception as e:
                logger.error(f"Failed to augment image {i}: {e}")
                continue

        logger.info(
            f"Dataset augmentation complete. Original: {len(original_images)}, "
            f"Total: {len(augmented_dataset['images'])}"
        )

        return augmented_dataset


class ImageAugmenter:
    """
    Direct image augmentation that can be fed into dataset generators.

    This class provides methods for augmenting individual images during
    the dataset generation process.
    """

    @staticmethod
    def create_augmenter(
        augmentation_types: List[AugmentationType],
        intensity: float = 1.0,
        probability: float = 0.5,
    ) -> AugmentationPipeline:
        """
        Create an augmentation pipeline for direct image processing.

        Args:
            augmentation_types: List of augmentation types to apply
            intensity: Global intensity multiplier for all augmentations
            probability: Global probability for applying augmentations

        Returns:
            Configured augmentation pipeline
        """
        pipeline = AugmentationPipeline()

        # Default configurations for each augmentation type
        default_configs = {
            AugmentationType.ROTATION: AugmentationConfig(
                probability=probability,
                intensity=intensity,
                parameters={"max_angle": 10},
            ),
            AugmentationType.SCALING: AugmentationConfig(
                probability=probability,
                intensity=intensity,
                parameters={"min_scale": 0.9, "max_scale": 1.1},
            ),
            AugmentationType.NOISE: AugmentationConfig(
                probability=probability,
                intensity=intensity,
                parameters={"noise_std": 5},
            ),
            AugmentationType.BLUR: AugmentationConfig(
                probability=probability,
                intensity=intensity,
                parameters={"blur_radius": 1.0},
            ),
            AugmentationType.COLOR_SHIFT: AugmentationConfig(
                probability=probability, intensity=intensity, parameters={}
            ),
            AugmentationType.CROPPING: AugmentationConfig(
                probability=probability,
                intensity=intensity,
                parameters={"crop_ratio": 0.05},
            ),
        }

        augmentation_map = {
            AugmentationType.ROTATION: RotationAugmentation,
            AugmentationType.SCALING: ScalingAugmentation,
            AugmentationType.NOISE: NoiseAugmentation,
            AugmentationType.BLUR: BlurAugmentation,
            AugmentationType.COLOR_SHIFT: ColorShiftAugmentation,
            AugmentationType.CROPPING: CroppingAugmentation,
        }

        for aug_type in augmentation_types:
            if aug_type in augmentation_map and aug_type in default_configs:
                augmentation_class = augmentation_map[aug_type]
                config = default_configs[aug_type]
                pipeline.add_augmentation(augmentation_class(config))
            else:
                logger.warning(f"Unknown or unsupported augmentation type: {aug_type}")

        return pipeline

    @staticmethod
    def augment_image(
        image: Image.Image,
        augmentation_types: List[AugmentationType],
        intensity: float = 1.0,
        probability: float = 0.5,
    ) -> Image.Image:
        """
        Apply augmentations to a single image.

        Args:
            image: PIL Image to augment
            augmentation_types: List of augmentation types to apply
            intensity: Intensity multiplier for augmentations
            probability: Probability of applying each augmentation

        Returns:
            Augmented PIL Image
        """
        pipeline = ImageAugmenter.create_augmenter(
            augmentation_types, intensity, probability
        )
        return pipeline.apply(image)


# Utility functions for common augmentation patterns
def get_default_augmentation_config() -> Dict[AugmentationType, AugmentationConfig]:
    """Get default augmentation configuration for common use cases."""
    return {
        AugmentationType.ROTATION: AugmentationConfig(
            probability=0.3, intensity=0.8, parameters={"max_angle": 5}
        ),
        AugmentationType.SCALING: AugmentationConfig(
            probability=0.3,
            intensity=0.8,
            parameters={"min_scale": 0.95, "max_scale": 1.05},
        ),
        AugmentationType.NOISE: AugmentationConfig(
            probability=0.2, intensity=0.5, parameters={"noise_std": 3}
        ),
        AugmentationType.BLUR: AugmentationConfig(
            probability=0.2, intensity=0.5, parameters={"blur_radius": 0.5}
        ),
        AugmentationType.COLOR_SHIFT: AugmentationConfig(
            probability=0.3, intensity=0.7, parameters={}
        ),
    }


def get_aggressive_augmentation_config() -> Dict[AugmentationType, AugmentationConfig]:
    """Get aggressive augmentation configuration for data augmentation."""
    return {
        AugmentationType.ROTATION: AugmentationConfig(
            probability=0.7, intensity=1.2, parameters={"max_angle": 15}
        ),
        AugmentationType.SCALING: AugmentationConfig(
            probability=0.6,
            intensity=1.0,
            parameters={"min_scale": 0.8, "max_scale": 1.2},
        ),
        AugmentationType.NOISE: AugmentationConfig(
            probability=0.5, intensity=1.0, parameters={"noise_std": 10}
        ),
        AugmentationType.BLUR: AugmentationConfig(
            probability=0.4, intensity=1.0, parameters={"blur_radius": 2.0}
        ),
        AugmentationType.COLOR_SHIFT: AugmentationConfig(
            probability=0.6, intensity=1.0, parameters={}
        ),
        AugmentationType.CROPPING: AugmentationConfig(
            probability=0.3, intensity=0.8, parameters={"crop_ratio": 0.1}
        ),
    }
