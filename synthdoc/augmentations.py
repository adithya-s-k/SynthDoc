#!/usr/bin/env python3
"""
Image Augmentation Script using Augraphy
=========================================

This script applies various document augmentations to images in a folder and creates
augmented copies with configurable ratios. It uses the Augraphy library for
realistic document distortions.

Usage:
    python augmentation.py --input_folder ./images --output_folder ./augmented_images
"""

import os
import argparse
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from augraphy import *

# Performance data for available augmentations only (Img/sec)
PERFORMANCE_DATA = {
    "BleedThrough": 0.58,
    "Brightness": 4.92,
    "BrightnessTexturize": 1.83,
    "ColorPaper": 4.83,
    "ColorShift": 0.79,
    "DirtyDrum": 0.83,
    "DirtyRollers": 1.47,
    "Folding": 3.18,
    "InkColorSwap": 3.47,
    "InkMottling": 5.41,
    "Letterpress": 0.35,
    "LightingGradient": 0.37,
    "LinesDegradation": 1.28,
    "LowInkPeriodicLines": 5.17,
    "Markup": 2.33,
    "NoiseTexturize": 0.83,
    "NoisyLines": 0.89,
    "PatternGenerator": 0.76,
    "Scribbles": 1.11,
    "ShadowCast": 0.75,
    "Squish": 0.72,
    "WaterMark": 2.09,
}


def calculate_custom_distribution_weights():
    """Calculate custom weights for augmentations with equal distribution."""
    # Get all available augmentation types (convert to lowercase)
    aug_types = []
    for aug_type in PERFORMANCE_DATA.keys():
        normalized_name = aug_type.lower()
        aug_types.append(normalized_name)

    # Original gets 300 (30%), remaining 700 distributed equally among available augmentations
    original_weight = 300
    remaining_weight = 1000 - original_weight

    # Distribute remaining weight equally among all available augmentations
    equal_weight = remaining_weight // len(aug_types)
    remainder = remaining_weight % len(aug_types)

    # Create weights dictionary
    weights = {}

    # Assign equal weights to all augmentations
    for i, aug_type in enumerate(aug_types):
        # Distribute remainder among first few augmentations
        extra = 1 if i < remainder else 0
        weights[aug_type] = equal_weight + extra

    return weights


# Calculate the weights
calculated_weights = calculate_custom_distribution_weights()

# Augmentation configuration with original at 30% and rest distributed equally
AUGMENTATION_CONFIG = {
    "original": 300,  # 30% kept as original
    **calculated_weights,
}

# Verify the ratios add up to 1000
assert sum(AUGMENTATION_CONFIG.values()) == 1000, (
    f"Ratios must sum to 1000, got {sum(AUGMENTATION_CONFIG.values())}"
)


def get_augmentation_pipeline(aug_type: str):
    """
    Get the augmentation pipeline for a specific augmentation type.
    Only includes augmentations that are confirmed to be available.

    Args:
        aug_type: The type of augmentation to apply

    Returns:
        Augraphy pipeline object
    """
    pipelines = {
        "bleedthrough": BleedThrough(),
        "brightness": Brightness(),
        "brightnesstexturize": BrightnessTexturize(),
        "colorpaper": ColorPaper(),
        "colorshift": ColorShift(),
        "dirtydrum": DirtyDrum(),
        "dirtyrollers": DirtyRollers(),
        "folding": Folding(),
        "inkcolorswap": InkColorSwap(),
        "inkmottling": InkMottling(),
        "letterpress": Letterpress(),
        "lightinggradient": LightingGradient(),
        "linesdegradation": LinesDegradation(),
        "lowinkperiodiclines": LowInkPeriodicLines(),
        "markup": Markup(),
        "noisetexturize": NoiseTexturize(),
        "noisylines": NoisyLines(),
        "patterngenerator": PatternGenerator(),
        "scribbles": Scribbles(),
        "shadowcast": ShadowCast(),
        "squish": Squish(),
        "watermark": WaterMark(),
    }

    return pipelines.get(aug_type, None)


def get_supported_image_extensions():
    """Get list of supported image file extensions."""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.

    Args:
        image: Image array to save
        output_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image)
        return success
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def generate_augmentation_plan(total_images: int, config: Dict[str, int]) -> List[str]:
    """
    Generate a plan for which augmentation to apply to each image.

    Args:
        total_images: Total number of input images
        config: Configuration dictionary with augmentation ratios

    Returns:
        List of augmentation types for each image
    """
    plan = []

    # Calculate actual counts based on ratios
    for aug_type, ratio in config.items():
        count = int((ratio / 1000.0) * total_images)
        plan.extend([aug_type] * count)

    # Handle any rounding differences
    while len(plan) < total_images:
        # Add the most common augmentation for remaining images
        most_common = max(config.items(), key=lambda x: x[1])[0]
        plan.append(most_common)

    # Shuffle the plan for random distribution
    random.shuffle(plan)

    return plan[:total_images]


def apply_padding_to_original(image: np.ndarray) -> np.ndarray:
    """
    Apply random top and right padding to original images.

    Args:
        image: Input image

    Returns:
        Image with random top and right padding (0-15px)
    """
    # Generate random padding values
    top_padding = random.randint(0, 15)
    right_padding = random.randint(0, 15)

    # Apply padding (top, bottom, left, right)
    # We only add top and right padding, so bottom and left are 0
    padded_image = cv2.copyMakeBorder(
        image,
        top_padding,
        0,  # top, bottom
        0,
        right_padding,  # left, right
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],  # White padding
    )

    return padded_image


def apply_augmentation(
    image: np.ndarray, aug_type: str
) -> Tuple[np.ndarray, bool, float]:
    """
    Apply a specific augmentation to an image.

    Args:
        image: Input image
        aug_type: Type of augmentation to apply

    Returns:
        Tuple of (augmented_image, success_flag, processing_time_seconds)
    """
    start_time = time.time()

    if aug_type == "original":
        # Apply random top and right padding to original images
        result = apply_padding_to_original(image)
        end_time = time.time()
        return result, True, end_time - start_time

    try:
        pipeline = get_augmentation_pipeline(aug_type)
        if pipeline is None:
            print(f"Warning: Unknown augmentation type '{aug_type}', keeping original")
            end_time = time.time()
            return image.copy(), False, end_time - start_time

        augmented = pipeline(image)
        end_time = time.time()
        return augmented, True, end_time - start_time

    except Exception as e:
        print(f"Error applying {aug_type} augmentation: {e}")
        end_time = time.time()
        return image.copy(), False, end_time - start_time


def process_images(
    input_folder: str,
    output_folder: str,
    config: Dict[str, int],
    max_samples: int = None,
):
    """
    Process all images in the input folder and create augmented versions.

    Args:
        input_folder: Path to input image folder
        output_folder: Path to output folder
        config: Augmentation configuration
        max_samples: Maximum number of images to process (None for all)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Get all image files
    supported_extensions = get_supported_image_extensions()
    image_files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    # Limit the number of images if max_samples is specified
    if max_samples is not None and max_samples > 0:
        if max_samples < len(image_files):
            print(f"Limiting to {max_samples} images out of {len(image_files)} found")
            # Shuffle to get a random sample
            random.shuffle(image_files)
            image_files = image_files[:max_samples]
        else:
            print(
                f"max_samples ({max_samples}) is >= total images ({len(image_files)}), processing all images"
            )

    print(f"Found {len(image_files)} images to process")

    # Generate augmentation plan
    augmentation_plan = generate_augmentation_plan(len(image_files), config)

    # Print augmentation distribution
    print("\nAugmentation Distribution:")
    aug_counts = {}
    for aug in augmentation_plan:
        aug_counts[aug] = aug_counts.get(aug, 0) + 1

    for aug_type, count in sorted(aug_counts.items()):
        percentage = (count / len(image_files)) * 100
        print(f"  {aug_type}: {count} images ({percentage:.1f}%)")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each image
    successful_augmentations = 0
    failed_augmentations = 0
    timing_stats = {}  # Track timing for each augmentation type

    print("\nProcessing images...")

    with tqdm(total=len(image_files), desc="Augmenting images") as pbar:
        for i, image_file in enumerate(image_files):
            pbar.set_description(f"Processing {image_file.name}")

            # Load image
            image = load_image(str(image_file))
            if image is None:
                pbar.update(1)
                failed_augmentations += 1
                continue

            # Get augmentation type for this image
            aug_type = augmentation_plan[i]

            # Apply augmentation with timing
            augmented_image, success, processing_time = apply_augmentation(
                image, aug_type
            )

            # Track timing statistics
            if aug_type not in timing_stats:
                timing_stats[aug_type] = []
            timing_stats[aug_type].append(processing_time)

            # Generate output filename with augmentation suffix
            file_stem = image_file.stem
            file_extension = image_file.suffix
            output_filename = f"{file_stem}{file_extension}"

            output_file_path = output_path / output_filename

            # Save augmented image
            if save_image(augmented_image, str(output_file_path)):
                if success:
                    successful_augmentations += 1
                else:
                    # Still count as successful if image was saved (even if it's original)
                    successful_augmentations += 1
            else:
                failed_augmentations += 1

            pbar.update(1)

    # Print processing results
    print("\nProcessing complete!")
    print(
        f"Successfully processed: {successful_augmentations}/{len(image_files)} images"
    )
    print(f"Failed: {failed_augmentations}/{len(image_files)} images")
    print(f"Output saved to: {output_folder}")

    # Print timing report
    print("\n" + "=" * 80)
    print("TIMING REPORT")
    print("=" * 80)

    total_time = 0
    for aug_type, times in sorted(timing_stats.items()):
        if times:  # Only if we have timing data
            avg_time = sum(times) / len(times)
            total_aug_time = sum(times)
            total_time += total_aug_time

            print(
                f"{aug_type:25} | Count: {len(times):3} | "
                f"Avg: {avg_time:6.3f}s | Total: {total_aug_time:6.2f}s | "
                f"Rate: {len(times) / total_aug_time:5.1f} img/s"
            )

    print("-" * 80)
    print(
        f"{'TOTAL':25} | Count: {successful_augmentations:3} | "
        f"Total: {total_time:6.2f}s | Rate: {successful_augmentations / total_time:5.1f} img/s"
    )
    print("=" * 80)


def main():
    """Main function to handle command line arguments and run the augmentation process."""
    parser = argparse.ArgumentParser(
        description="Apply document augmentations to images with configurable ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python augmentation.py --input_folder ./images --output_folder ./augmented
  python augmentation.py -i ./data/images -o ./data/augmented --seed 42
  python augmentation.py -i ./images -o ./augmented --max_samples 100
  
Available Augmentations ({len(PERFORMANCE_DATA)} total):
  {", ".join(sorted(PERFORMANCE_DATA.keys()))}
  
Current Distribution (out of 1000):
  - original: 300 (30%) - kept as original with random top/right padding (0-15px)
  - Each augmentation: ~{700 // len(PERFORMANCE_DATA)} (~{(700 // len(PERFORMANCE_DATA)) / 10:.1f}%) - equally distributed
  
Note: All available augmentations are equally distributed. Original images get 30% 
with random top and right padding of 0-15 pixels.

Use --max_samples for testing with a limited number of images (e.g., 100).
        """,
    )

    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Path to input folder containing images",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Path to output folder for augmented images",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible results"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (JSON format with augmentation ratios)",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show augmentation plan without processing images",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of images to process (useful for testing, e.g., 100)",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Use default configuration
    config = AUGMENTATION_CONFIG

    # Load custom configuration if provided
    if args.config:
        try:
            import json

            with open(args.config, "r") as f:
                custom_config = json.load(f)

            # Validate that ratios sum to 1000
            if sum(custom_config.values()) != 1000:
                print(
                    f"Warning: Custom config ratios sum to {sum(custom_config.values())}, should be 1000"
                )

            config = custom_config
            print(f"Using custom configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading custom config: {e}")
            print("Using default configuration")

    # Verify input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return

    if args.dry_run:
        # Show what would be done without actually processing
        input_path = Path(args.input_folder)
        supported_extensions = get_supported_image_extensions()
        image_files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not image_files:
            print(f"No image files found in {args.input_folder}")
            return

        # Limit the number of images if max_samples is specified
        if args.max_samples is not None and args.max_samples > 0:
            if args.max_samples < len(image_files):
                print(
                    f"Limiting to {args.max_samples} images out of {len(image_files)} found for dry run"
                )
                # Shuffle to get a random sample
                random.shuffle(image_files)
                image_files = image_files[: args.max_samples]
            else:
                print(
                    f"max_samples ({args.max_samples}) is >= total images ({len(image_files)}), showing all images"
                )

        plan = generate_augmentation_plan(len(image_files), config)

        print(f"Dry run mode - would process {len(image_files)} images")
        print("Augmentation distribution:")

        aug_counts = {}
        for aug in plan:
            aug_counts[aug] = aug_counts.get(aug, 0) + 1

        for aug_type, count in sorted(aug_counts.items()):
            percentage = (count / len(image_files)) * 100
            print(f"  {aug_type}: {count} images ({percentage:.1f}%)")
    else:
        # Process images
        process_images(args.input_folder, args.output_folder, config, args.max_samples)


if __name__ == "__main__":
    main()


# python augmentation.py --input_folder ./datasets/nayana_sections_en_en_para/images --output_folder ./datasets/nayana_sections_en_en_para_augment/images --max_samples 1000

# python augmentation.py --input_folder ./datasets/nayana_sections_kn_kn_para/images --output_folder ./datasets/nayana_sections_kn_kn_para_augment/images --max_samples 1000
