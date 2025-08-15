"""
SynthDoc Augmentation Module
============================

This module provides document image augmentation capabilities using Augraphy.
It acts as a filter/processor that can augment existing datasets or image folders.
"""

from .config import (
    AVAILABLE_AUGMENTATIONS,
    LIGHT_AUGMENTATIONS,
    BALANCED_AUGMENTATIONS,
    HEAVY_AUGMENTATIONS,
    DOCUMENT_QUALITY_AUGMENTATIONS,
    calculate_augmentation_ratios,
    detect_image_column,
)
from .processor import AugmentationProcessor

__all__ = [
    "AugmentationProcessor",
    "AVAILABLE_AUGMENTATIONS",
    "LIGHT_AUGMENTATIONS",
    "BALANCED_AUGMENTATIONS", 
    "HEAVY_AUGMENTATIONS",
    "DOCUMENT_QUALITY_AUGMENTATIONS",
    "calculate_augmentation_ratios",
    "detect_image_column",
]