"""
Augmentation processor for datasets and image folders.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
from datasets import Dataset
from tqdm import tqdm

from .engine import AugmentationEngine
from .config import (
    calculate_augmentation_ratios,
    detect_image_column,
    AVAILABLE_AUGMENTATIONS
)


class AugmentationProcessor:
    """
    Processor for applying augmentations to datasets or image folders.
    Acts as a filter that takes input data and returns augmented data.
    """
    
    def __init__(self):
        self.engine = AugmentationEngine()
    
    def process_dataset(
        self,
        dataset: Dataset,
        config,  # AugmentationConfig type hint removed to avoid circular import
        image_column: Optional[str] = None
    ) -> Dataset:
        """
        Apply augmentations to a HuggingFace dataset.
        
        Args:
            dataset: Input HuggingFace dataset
            config: Augmentation configuration
            image_column: Name of image column (auto-detected if None)
            
        Returns:
            New dataset with augmented images
        """
        # Auto-detect image column if not provided
        if image_column is None:
            image_column = detect_image_column(dataset)
        
        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Calculate augmentation ratios
        if isinstance(config.augmentations, list):
            ratios = calculate_augmentation_ratios(
                config.augmentations, 
                config.original_ratio
            )
        else:
            ratios = config.augmentations
        
        # Limit samples if specified
        num_samples = len(dataset)
        if config.max_samples is not None and config.max_samples < num_samples:
            # Take random sample
            indices = random.sample(range(num_samples), config.max_samples)
            dataset = dataset.select(indices)
            num_samples = config.max_samples
        
        # Generate augmentation plan
        augmentation_plan = self.engine.generate_augmentation_plan(num_samples, ratios)
        
        print(f"Processing {num_samples} images with augmentations:")
        self._print_augmentation_distribution(augmentation_plan)
        
        # Process each sample
        augmented_samples = []
        timing_stats = {}
        
        with tqdm(total=num_samples, desc="Applying augmentations") as pbar:
            for i, sample in enumerate(dataset):
                aug_type = augmentation_plan[i]
                
                # Get image
                image = sample[image_column]
                if isinstance(image, str):
                    # Image path
                    image = Image.open(image)
                
                # Apply augmentation
                augmented_image, success, processing_time = self.engine.apply_augmentation(
                    image, aug_type
                )
                
                # Track timing
                if aug_type not in timing_stats:
                    timing_stats[aug_type] = []
                timing_stats[aug_type].append(processing_time)
                
                # Create new sample
                new_sample = dict(sample)
                new_sample[image_column] = augmented_image
                
                # Add augmentation metadata if requested
                if config.add_augmentation_metadata:
                    new_sample["augmentation_type"] = aug_type
                    new_sample["augmentation_success"] = success
                    new_sample["processing_time"] = processing_time
                
                augmented_samples.append(new_sample)
                pbar.update(1)
        
        # Print timing report
        self._print_timing_report(timing_stats, num_samples)
        
        # Create new dataset
        return Dataset.from_list(augmented_samples)
    
    def process_folder(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path], 
        config  # AugmentationConfig type hint removed to avoid circular import
    ) -> Dataset:
        """
        Apply augmentations to images in a folder.
        
        Args:
            input_folder: Path to input image folder
            output_folder: Path to output folder
            config: Augmentation configuration
            
        Returns:
            HuggingFace dataset with augmented images
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in {input_folder}")
        
        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Limit samples if specified
        if config.max_samples is not None and config.max_samples < len(image_files):
            random.shuffle(image_files)
            image_files = image_files[:config.max_samples]
        
        # Calculate augmentation ratios
        if isinstance(config.augmentations, list):
            ratios = calculate_augmentation_ratios(
                config.augmentations,
                config.original_ratio
            )
        else:
            ratios = config.augmentations
        
        # Generate augmentation plan
        augmentation_plan = self.engine.generate_augmentation_plan(len(image_files), ratios)
        
        print(f"Processing {len(image_files)} images from {input_folder}")
        self._print_augmentation_distribution(augmentation_plan)
        
        # Process images
        augmented_samples = []
        timing_stats = {}
        
        with tqdm(total=len(image_files), desc="Augmenting images") as pbar:
            for i, image_file in enumerate(image_files):
                pbar.set_description(f"Processing {image_file.name}")
                
                # Load image
                try:
                    image = Image.open(image_file)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
                    continue
                
                # Get augmentation type
                aug_type = augmentation_plan[i]
                
                # Apply augmentation
                augmented_image, success, processing_time = self.engine.apply_augmentation(
                    image, aug_type
                )
                
                # Track timing
                if aug_type not in timing_stats:
                    timing_stats[aug_type] = []
                timing_stats[aug_type].append(processing_time)
                
                # Save augmented image
                output_filename = f"{image_file.stem}{image_file.suffix}"
                output_path_file = output_path / output_filename
                
                try:
                    augmented_image.save(output_path_file)
                except Exception as e:
                    print(f"Error saving {output_path_file}: {e}")
                    continue
                
                # Create sample for dataset
                sample = {
                    "image": augmented_image,
                    "image_path": str(output_path_file),
                    "original_filename": image_file.name,
                    "original_path": str(image_file),
                }
                
                # Add augmentation metadata if requested
                if config.add_augmentation_metadata:
                    sample["augmentation_type"] = aug_type
                    sample["augmentation_success"] = success
                    sample["processing_time"] = processing_time
                
                augmented_samples.append(sample)
                pbar.update(1)
        
        # Print timing report
        self._print_timing_report(timing_stats, len(augmented_samples))
        
        print(f"Augmented images saved to: {output_folder}")
        
        # Return as dataset
        return Dataset.from_list(augmented_samples)
    
    def _print_augmentation_distribution(self, augmentation_plan: List[str]):
        """Print the distribution of augmentations."""
        aug_counts = {}
        for aug in augmentation_plan:
            aug_counts[aug] = aug_counts.get(aug, 0) + 1
        
        print("Augmentation Distribution:")
        for aug_type, count in sorted(aug_counts.items()):
            percentage = (count / len(augmentation_plan)) * 100
            print(f"  {aug_type}: {count} images ({percentage:.1f}%)")
    
    def _print_timing_report(self, timing_stats: Dict[str, List[float]], total_images: int):
        """Print timing performance report."""
        print("\n" + "=" * 60)
        print("AUGMENTATION TIMING REPORT")
        print("=" * 60)
        
        total_time = 0
        for aug_type, times in sorted(timing_stats.items()):
            if times:
                avg_time = sum(times) / len(times)
                total_aug_time = sum(times)
                total_time += total_aug_time
                
                print(
                    f"{aug_type:20} | Count: {len(times):3} | "
                    f"Avg: {avg_time:6.3f}s | Total: {total_aug_time:6.2f}s | "
                    f"Rate: {len(times) / total_aug_time:5.1f} img/s"
                )
        
        if total_time > 0:
            print("-" * 60)
            print(
                f"{'TOTAL':20} | Count: {total_images:3} | "
                f"Total: {total_time:6.2f}s | Rate: {total_images / total_time:5.1f} img/s"
            )
        print("=" * 60)
    
    def validate_augmentations(self, augmentations: Union[List[str], Dict[str, float]]) -> bool:
        """
        Validate that all requested augmentations are available.
        
        Args:
            augmentations: List of augmentation names or dict with ratios
            
        Returns:
            True if all augmentations are valid
            
        Raises:
            ValueError: If invalid augmentations are found
        """
        if isinstance(augmentations, list):
            aug_list = augmentations
        else:
            aug_list = list(augmentations.keys())
        
        # Check for invalid augmentations (excluding "original")
        invalid_augs = [
            aug for aug in aug_list 
            if aug != "original" and aug not in AVAILABLE_AUGMENTATIONS
        ]
        
        if invalid_augs:
            raise ValueError(
                f"Unsupported augmentations: {invalid_augs}. "
                f"Available: {AVAILABLE_AUGMENTATIONS}"
            )
        
        return True