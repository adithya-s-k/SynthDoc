"""
Augmentation configuration module with pythonic list-based configurations.
"""

from typing import Dict, List, Union
from dataclasses import dataclass

# All available augmentations from your script
AVAILABLE_AUGMENTATIONS = [
    "bleedthrough",
    "brightness", 
    "brightnesstexturize",
    "colorpaper",
    "colorshift",
    "dirtydrum",
    "dirtyrollers", 
    "folding",
    "inkcolorswap",
    "inkmottling",
    "letterpress",
    "lightinggradient",
    "linesdegradation",
    "lowinkperiodiclines",
    "markup",
    "noisetexturize",
    "noisylines",
    "patterngenerator",
    "scribbles",
    "shadowcast",
    "squish",
    "watermark",
]

# Predefined augmentation presets
LIGHT_AUGMENTATIONS = ["brightness", "colorpaper", "original"]

BALANCED_AUGMENTATIONS = [
    "brightness", 
    "folding", 
    "watermark", 
    "colorshift",
    "inkmottling",
    "original"
]

HEAVY_AUGMENTATIONS = [
    "brightness",
    "colorshift", 
    "folding",
    "watermark",
    "bleedthrough",
    "dirtydrum",
    "shadowcast",
    "squish",
    "original"
]

DOCUMENT_QUALITY_AUGMENTATIONS = [
    "brightness",
    "colorpaper", 
    "inkmottling",
    "lowinkperiodiclines",
    "original"
]

# Performance data from your script
AUGMENTATION_PERFORMANCE = {
    "bleedthrough": 0.58,
    "brightness": 4.92,
    "brightnesstexturize": 1.83,
    "colorpaper": 4.83,
    "colorshift": 0.79,
    "dirtydrum": 0.83,
    "dirtyrollers": 1.47,
    "folding": 3.18,
    "inkcolorswap": 3.47,
    "inkmottling": 5.41,
    "letterpress": 0.35,
    "lightinggradient": 0.37,
    "linesdegradation": 1.28,
    "lowinkperiodiclines": 5.17,
    "markup": 2.33,
    "noisetexturize": 0.83,
    "noisylines": 0.89,
    "patterngenerator": 0.76,
    "scribbles": 1.11,
    "shadowcast": 0.75,
    "squish": 0.72,
    "watermark": 2.09,
}


@dataclass
class AugmentationConfig:
    """Configuration for document augmentation."""
    
    # Input configuration
    augmentations: Union[List[str], Dict[str, float]] = None
    original_ratio: float = 0.3  # 30% kept as original
    max_samples: int = None
    
    # Processing configuration  
    random_seed: int = None
    preserve_metadata: bool = True
    
    # Output configuration
    output_format: str = "dataset"  # "dataset" or "folder"
    add_augmentation_metadata: bool = True
    
    def __post_init__(self):
        """Validate and process configuration."""
        if self.augmentations is None:
            self.augmentations = BALANCED_AUGMENTATIONS.copy()
        
        # Convert string presets to lists
        if isinstance(self.augmentations, str):
            preset_map = {
                "light": LIGHT_AUGMENTATIONS,
                "balanced": BALANCED_AUGMENTATIONS,
                "heavy": HEAVY_AUGMENTATIONS,
                "document_quality": DOCUMENT_QUALITY_AUGMENTATIONS,
            }
            self.augmentations = preset_map.get(self.augmentations, BALANCED_AUGMENTATIONS)
        
        # Validate augmentations
        if isinstance(self.augmentations, list):
            # Ensure "original" is in the list if not specified
            if "original" not in self.augmentations:
                self.augmentations.append("original")
            
            # Validate all augmentations are supported
            invalid_augs = [
                aug for aug in self.augmentations 
                if aug != "original" and aug not in AVAILABLE_AUGMENTATIONS
            ]
            if invalid_augs:
                raise ValueError(f"Unsupported augmentations: {invalid_augs}")
        
        elif isinstance(self.augmentations, dict):
            # Validate custom ratios
            if not all(isinstance(v, (int, float)) for v in self.augmentations.values()):
                raise ValueError("All augmentation ratios must be numeric")
            
            # Ensure ratios sum to 1.0
            total_ratio = sum(self.augmentations.values())
            if abs(total_ratio - 1.0) > 0.001:
                raise ValueError(f"Augmentation ratios must sum to 1.0, got {total_ratio}")


def calculate_augmentation_ratios(
    augmentations: List[str], 
    original_ratio: float = 0.3
) -> Dict[str, float]:
    """
    Calculate augmentation ratios from a list of augmentations.
    
    Args:
        augmentations: List of augmentation names
        original_ratio: Ratio to keep as original (default 0.3 = 30%)
    
    Returns:
        Dictionary with augmentation ratios that sum to 1.0
    """
    if "original" not in augmentations:
        augmentations = augmentations + ["original"]
    
    # Calculate remaining ratio for other augmentations
    other_augmentations = [aug for aug in augmentations if aug != "original"]
    remaining_ratio = 1.0 - original_ratio
    
    if not other_augmentations:
        return {"original": 1.0}
    
    # Distribute remaining ratio equally
    equal_ratio = remaining_ratio / len(other_augmentations)
    
    ratios = {"original": original_ratio}
    for aug in other_augmentations:
        ratios[aug] = equal_ratio
    
    return ratios


def detect_image_column(dataset) -> str:
    """
    Auto-detect the image column in a HuggingFace dataset.
    
    Args:
        dataset: HuggingFace Dataset
        
    Returns:
        Name of the image column
        
    Raises:
        ValueError: If no image column is found
    """
    # Common image column names to check
    possible_columns = ["image", "images", "picture", "pictures", "img", "photo", "document"]
    
    for col_name in possible_columns:
        if col_name in dataset.column_names:
            # Check if it's actually an image column
            sample = dataset[0]
            if col_name in sample:
                # Check if it's PIL Image or image path
                value = sample[col_name]
                if hasattr(value, 'save') or isinstance(value, str):
                    return col_name
    
    raise ValueError(
        f"No image column found. Available columns: {dataset.column_names}. "
        f"Expected one of: {possible_columns}"
    )