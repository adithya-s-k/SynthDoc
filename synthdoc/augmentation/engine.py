"""
Core augmentation engine adapted from the original augmentation script.
"""

import time
import random
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    from augraphy import *
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False


class AugmentationEngine:
    """Core engine for applying document augmentations using Augraphy."""
    
    def __init__(self):
        if not AUGRAPHY_AVAILABLE:
            raise ImportError(
                "Augraphy is required for augmentations. Install with: pip install augraphy"
            )
    
    def get_augmentation_pipeline(self, aug_type: str):
        """
        Get the augmentation pipeline for a specific augmentation type.
        
        Args:
            aug_type: The type of augmentation to apply
            
        Returns:
            Augraphy pipeline object or None if not found
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
        return pipelines.get(aug_type.lower(), None)
    
    def apply_padding_to_original(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random top and right padding to original images.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image with random top and right padding (0-15px)
        """
        top_padding = random.randint(0, 15)
        right_padding = random.randint(0, 15)
        
        padded_image = cv2.copyMakeBorder(
            image,
            top_padding, 0,  # top, bottom
            0, right_padding,  # left, right
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],  # White padding
        )
        return padded_image
    
    def apply_augmentation(
        self, image: Union[Image.Image, np.ndarray], aug_type: str
    ) -> Tuple[Image.Image, bool, float]:
        """
        Apply a specific augmentation to an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            aug_type: Type of augmentation to apply
            
        Returns:
            Tuple of (augmented_image_pil, success_flag, processing_time_seconds)
        """
        start_time = time.time()
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original_pil = image
        else:
            image_np = image
            original_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if aug_type == "original":
            # Apply random padding to original images
            result_np = self.apply_padding_to_original(image_np)
            result_pil = Image.fromarray(cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB))
            end_time = time.time()
            return result_pil, True, end_time - start_time
        
        try:
            pipeline = self.get_augmentation_pipeline(aug_type)
            if pipeline is None:
                print(f"Warning: Unknown augmentation type '{aug_type}', keeping original")
                end_time = time.time()
                return original_pil, False, end_time - start_time
            
            # Apply augmentation
            augmented_np = pipeline(image_np)
            
            # Convert back to PIL
            result_pil = Image.fromarray(cv2.cvtColor(augmented_np, cv2.COLOR_BGR2RGB))
            
            end_time = time.time()
            return result_pil, True, end_time - start_time
            
        except Exception as e:
            print(f"Error applying {aug_type} augmentation: {e}")
            end_time = time.time()
            return original_pil, False, end_time - start_time
    
    def generate_augmentation_plan(
        self, total_images: int, ratios: Dict[str, float]
    ) -> List[str]:
        """
        Generate a plan for which augmentation to apply to each image.
        
        Args:
            total_images: Total number of input images
            ratios: Dictionary with augmentation ratios (must sum to 1.0)
            
        Returns:
            List of augmentation types for each image
        """
        plan = []
        
        # Calculate actual counts based on ratios
        for aug_type, ratio in ratios.items():
            count = int(ratio * total_images)
            plan.extend([aug_type] * count)
        
        # Handle any rounding differences
        while len(plan) < total_images:
            # Add the most common augmentation for remaining images
            most_common = max(ratios.items(), key=lambda x: x[1])[0]
            plan.append(most_common)
        
        # Shuffle the plan for random distribution
        random.shuffle(plan)
        
        return plan[:total_images]