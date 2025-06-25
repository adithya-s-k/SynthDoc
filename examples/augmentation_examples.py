#!/usr/bin/env python3
"""
Examples demonstrating the augmentation functionality in SynthDoc.
"""

from PIL import Image
import numpy as np

from synthdoc.models import AugmentationType
from synthdoc.augmentations import (
    DatasetAugmenter,
    ImageAugmenter,
    AugmentationConfig,
    get_default_augmentation_config,
    get_aggressive_augmentation_config,
)


def create_sample_image() -> Image.Image:
    """Create a sample image for testing."""
    # Create a simple test image with text-like patterns
    img_array = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White background

    # Add some black rectangles to simulate text
    img_array[50:70, 50:250] = 0  # Horizontal line
    img_array[100:120, 50:200] = 0  # Another line
    img_array[150:170, 75:225] = 0  # Third line

    return Image.fromarray(img_array)


def example_single_image_augmentation():
    """Example: Apply augmentations to a single image."""
    print("=== Single Image Augmentation Example ===")

    # Create a sample image
    original_image = create_sample_image()

    # Define augmentation types to apply
    augmentation_types = [
        AugmentationType.ROTATION,
        AugmentationType.NOISE,
        AugmentationType.COLOR_SHIFT,
    ]

    # Apply augmentations
    augmented_image = ImageAugmenter.augment_image(
        image=original_image,
        augmentation_types=augmentation_types,
        intensity=0.8,
        probability=0.7,
    )

    print(f"Original image size: {original_image.size}")
    print(f"Augmented image size: {augmented_image.size}")

    # Save images for inspection (optional)
    # original_image.save("original.png")
    # augmented_image.save("augmented.png")

    return original_image, augmented_image


def example_dataset_augmentation():
    """Example: Apply augmentations to a dataset."""
    print("\n=== Dataset Augmentation Example ===")

    # Create a mock dataset
    dataset = {
        "images": [create_sample_image() for _ in range(3)],
        "metadata": [
            {"filename": "sample_1.png", "label": "document"},
            {"filename": "sample_2.png", "label": "document"},
            {"filename": "sample_3.png", "label": "document"},
        ],
        "labels": ["doc", "doc", "doc"],
    }

    # Create augmentation configuration
    augmentation_configs = get_default_augmentation_config()

    # Create augmenter
    augmenter = DatasetAugmenter(augmentation_configs)

    # Apply augmentations
    augmented_dataset = augmenter.augment_dataset(
        dataset=dataset,
        num_augmentations=2,  # Create 2 augmented versions per original image
    )

    print(f"Original dataset size: {len(dataset['images'])}")
    print(f"Augmented dataset size: {len(augmented_dataset['images'])}")
    print(
        f"Sample metadata: {augmented_dataset['metadata'][3]}"
    )  # First augmented sample

    return augmented_dataset


def example_custom_augmentation_pipeline():
    """Example: Create a custom augmentation pipeline."""
    print("\n=== Custom Augmentation Pipeline Example ===")

    # Create custom configuration
    custom_configs = {
        AugmentationType.ROTATION: AugmentationConfig(
            probability=1.0,  # Always apply
            intensity=1.5,  # High intensity
            parameters={"max_angle": 20},
        ),
        AugmentationType.SCALING: AugmentationConfig(
            probability=0.8,
            intensity=1.2,
            parameters={"min_scale": 0.7, "max_scale": 1.3},
        ),
        AugmentationType.BLUR: AugmentationConfig(
            probability=0.6, intensity=1.0, parameters={"blur_radius": 3.0}
        ),
    }

    # Create augmenter with custom config
    augmenter = DatasetAugmenter(custom_configs)

    # Test on a single image
    original_image = create_sample_image()
    augmented_image = augmenter.pipeline.apply(original_image)

    print("Applied custom augmentation pipeline")
    print(f"Pipeline has {len(augmenter.pipeline.augmentations)} augmentations")

    return augmented_image


def example_aggressive_augmentation():
    """Example: Use aggressive augmentation settings."""
    print("\n=== Aggressive Augmentation Example ===")

    # Use aggressive configuration
    aggressive_configs = get_aggressive_augmentation_config()

    # Create augmenter
    augmenter = DatasetAugmenter(aggressive_configs)

    # Apply to a single image
    original_image = create_sample_image()
    augmented_image = augmenter.pipeline.apply(original_image)

    print("Applied aggressive augmentation settings")
    print("This creates more dramatic transformations suitable for robust training")

    return augmented_image


def example_direct_pipeline_usage():
    """Example: Use ImageAugmenter for direct pipeline creation."""
    print("\n=== Direct Pipeline Usage Example ===")

    # Create augmentation pipeline
    pipeline = ImageAugmenter.create_augmenter(
        augmentation_types=[
            AugmentationType.ROTATION,
            AugmentationType.NOISE,
            AugmentationType.COLOR_SHIFT,
            AugmentationType.SCALING,
        ],
        intensity=0.9,
        probability=0.6,
    )

    # Apply to multiple images
    original_images = [create_sample_image() for _ in range(5)]
    augmented_images = []

    for img in original_images:
        augmented = pipeline.apply(img)
        augmented_images.append(augmented)

    print(f"Processed {len(original_images)} images through the pipeline")
    print(f"Pipeline contains {len(pipeline.augmentations)} augmentation steps")

    return augmented_images


if __name__ == "__main__":
    """Run all augmentation examples."""
    print("SynthDoc Augmentation Examples")
    print("=" * 50)

    try:
        # Run examples
        example_single_image_augmentation()
        example_dataset_augmentation()
        example_custom_augmentation_pipeline()
        example_aggressive_augmentation()
        example_direct_pipeline_usage()

        print("\n" + "=" * 50)
        print("All augmentation examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()
