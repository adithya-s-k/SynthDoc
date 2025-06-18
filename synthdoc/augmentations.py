"""
Document augmentation utilities.

This module provides various augmentation techniques for documents including
visual transformations, noise addition, and layout modifications.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from enum import Enum
import numpy as np 
import random
from PIL import Image, ImageEnhance, ImageFilter

class AugmentationType(Enum):
    """Available augmentation techniques."""

    ROTATION = "rotation"
    # SCALING = "scaling"
    # CROPPING = "cropping"
    NOISE = "noise"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    BLUR = "blur"
    # PERSPECTIVE = "perspective"
    # ELASTIC = "elastic"
    # COLOR_SHIFT = "color_shift"


class Augmentor:
    """Document augmentation engine."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_augmentations = [aug.value for aug in AugmentationType]

    def apply_augmentations(self, image: Image.Image, augmentations: List[AugmentationType]) -> Image.Image:
        """Apply image augmentations"""
        if not augmentations:
            return image 
        
        augmented_image = image.copy()

        for aug in augmentations:
            if aug == AugmentationType.ROTATION:
                angle = random.uniform(-3, 3)
                augmented_image = augmented_image.rotate(angle, fillcolor='white', expand=True)
            
            elif aug == AugmentationType.NOISE:
                np_img = np.array(augmented_image)
                noise = np.random.normal(0, 5, np_img.shape).astype(np.uint8)
                noisy_img = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                augmented_image = Image.fromarray(noisy_img)
            
            elif aug == AugmentationType.BLUR:
                augmented_image = augmented_image.filter(ImageFilter.GaussianBlur(radius=0.3))
            
            elif aug == AugmentationType.BRIGHTNESS:
                enhancer = ImageEnhance.Brightness(augmented_image)
                factor = random.uniform(0.85, 1.15)
                augmented_image = enhancer.enhance(factor)
                
            elif aug == AugmentationType.CONTRAST:
                enhancer = ImageEnhance.Contrast(augmented_image)
                factor = random.uniform(0.9, 1.1)
                augmented_image = enhancer.enhance(factor)

        return augmented_image