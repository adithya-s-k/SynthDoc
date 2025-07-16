"""
Document augmentation utilities.

This module provides various augmentation techniques for documents including
visual transformations, noise addition, and layout modifications.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from enum import Enum
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class AugmentationType(Enum):
    """Available augmentation techniques."""

    ROTATION = "rotation"
    SCALING = "scaling"
    CROPPING = "cropping"
    NOISE = "noise"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    BLUR = "blur"
    PERSPECTIVE = "perspective"
    ELASTIC = "elastic"
    COLOR_SHIFT = "color_shift"


class Augmentor:
    """Document augmentation engine with actual image processing implementations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_augmentations = [aug.value for aug in AugmentationType]
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available. Some augmentations will be limited.")
        if not CV2_AVAILABLE:
            self.logger.warning("OpenCV not available. Some augmentations will be limited.")

    def apply_augmentations(
        self,
        documents: List[Dict[str, Any]],
        augmentations: List[str],
        intensity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Apply augmentations to a list of documents.

        Args:
            documents: List of documents to augment
            augmentations: List of augmentation types to apply
            intensity: Augmentation intensity (0.0 to 1.0)

        Returns:
            List of augmented documents
        """
        self.logger.info(
            f"Applying {len(augmentations)} augmentations to {len(documents)} documents"
        )

        augmented_docs = []

        for doc in documents:
            for aug_type in augmentations:
                if aug_type in self.available_augmentations:
                    augmented_doc = self._apply_single_augmentation(
                        doc, aug_type, intensity
                    )
                    if augmented_doc:
                        augmented_docs.append(augmented_doc)
                else:
                    self.logger.warning(f"Unknown augmentation type: {aug_type}")

        return augmented_docs

    def _apply_single_augmentation(
        self, document: Dict[str, Any], augmentation: str, intensity: float
    ) -> Optional[Dict[str, Any]]:
        """Apply a single augmentation to a document."""
        image = document.get("image")
        if image is None:
            self.logger.warning(f"No image found in document {document.get('id')}")
            return None
        
        try:
            # Apply the specific augmentation
            if augmentation == AugmentationType.ROTATION.value:
                augmented_image = self.apply_rotation(image, intensity)
            elif augmentation == AugmentationType.SCALING.value:
                augmented_image = self.apply_scaling(image, intensity)
            elif augmentation == AugmentationType.CROPPING.value:
                augmented_image = self.apply_cropping(image, intensity)
            elif augmentation == AugmentationType.NOISE.value:
                augmented_image = self.apply_noise(image, intensity)
            elif augmentation == AugmentationType.BRIGHTNESS.value:
                augmented_image = self.apply_brightness(image, intensity)
            elif augmentation == AugmentationType.CONTRAST.value:
                augmented_image = self.apply_contrast(image, intensity)
            elif augmentation == AugmentationType.BLUR.value:
                augmented_image = self.apply_blur(image, intensity)
            elif augmentation == AugmentationType.PERSPECTIVE.value:
                augmented_image = self.apply_perspective(image, intensity)
            elif augmentation == AugmentationType.ELASTIC.value:
                augmented_image = self.apply_elastic(image, intensity)
            elif augmentation == AugmentationType.COLOR_SHIFT.value:
                augmented_image = self.apply_color_shift(image, intensity)
            else:
                self.logger.warning(f"Unsupported augmentation: {augmentation}")
                return None
            
            # Create augmented document
            augmented_doc = document.copy()
            augmented_doc["image"] = augmented_image
            augmented_doc["augmentation"] = {
                "type": augmentation,
                "intensity": intensity,
                "original_id": document.get("id"),
            }

            # Update document ID to reflect augmentation
            original_id = document.get("id", "unknown")
            augmented_doc["id"] = f"{original_id}_{augmentation}"
            
            # Update metadata if present
            if "metadata" in augmented_doc:
                augmented_doc["metadata"]["augmentation"] = {
                    "type": augmentation,
                    "intensity": intensity,
                    "image_width": augmented_image.width,
                    "image_height": augmented_image.height,
                }

            return augmented_doc
            
        except Exception as e:
            self.logger.error(f"Failed to apply {augmentation}: {e}")
            return None

    def apply_rotation(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply rotation augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for rotation")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate rotation angle based on intensity (-15 to 15 degrees)
        max_angle = 15
        angle = np.random.uniform(-max_angle * intensity, max_angle * intensity)
        
        self.logger.debug(f"Applying rotation: {angle:.2f} degrees")
        
        # Apply rotation with white background
        rotated = image.rotate(angle, expand=True, fillcolor='white')
        return rotated

    def apply_scaling(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply scaling augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for scaling")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate scale factor based on intensity (0.8 to 1.2)
        min_scale = 1.0 - 0.2 * intensity
        max_scale = 1.0 + 0.2 * intensity
        scale = np.random.uniform(min_scale, max_scale)
        
        self.logger.debug(f"Applying scaling: {scale:.2f}")
        
        # Apply scaling
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        scaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        return scaled

    def apply_cropping(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply random cropping augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for cropping")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate crop size based on intensity (keep 70%-95% of original)
        crop_ratio = 1.0 - (0.3 * intensity)
        new_width = int(image.width * crop_ratio)
        new_height = int(image.height * crop_ratio)
        
        # Random crop position
        max_x = image.width - new_width
        max_y = image.height - new_height
        left = np.random.randint(0, max(1, max_x))
        top = np.random.randint(0, max(1, max_y))
        
        self.logger.debug(f"Applying cropping: {new_width}x{new_height} at ({left}, {top})")
        
        # Apply crop
        cropped = image.crop((left, top, left + new_width, top + new_height))
        
        # Resize back to original size
        resized = cropped.resize((image.width, image.height), Image.LANCZOS)
        
        return resized

    def apply_noise(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply noise augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for noise")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for noise application
        img_array = np.array(image)
        
        # Calculate noise level based on intensity
        noise_level = 25 * intensity  # Max noise standard deviation of 25
        
        self.logger.debug(f"Applying noise: level {noise_level:.2f}")
        
        # Generate and apply noise
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.float32)
        noisy_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)

    def apply_brightness(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply brightness augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for brightness")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate brightness factor based on intensity (0.5 to 1.5)
        min_factor = 1.0 - 0.5 * intensity
        max_factor = 1.0 + 0.5 * intensity
        factor = np.random.uniform(min_factor, max_factor)
        
        self.logger.debug(f"Applying brightness: factor {factor:.2f}")
        
        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
        
        return enhanced

    def apply_contrast(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply contrast augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for contrast")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate contrast factor based on intensity (0.5 to 1.5)
        min_factor = 1.0 - 0.5 * intensity
        max_factor = 1.0 + 0.5 * intensity
        factor = np.random.uniform(min_factor, max_factor)
        
        self.logger.debug(f"Applying contrast: factor {factor:.2f}")
        
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
        
        return enhanced

    def apply_blur(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply blur augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for blur")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Calculate blur radius based on intensity
        max_radius = 3.0
        radius = max_radius * intensity
        
        self.logger.debug(f"Applying blur: radius {radius:.2f}")
        
        if radius > 0.1:  # Only apply if radius is meaningful
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
            return blurred
        else:
            return image

    def apply_perspective(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply perspective transformation."""
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            self.logger.warning("OpenCV and PIL required for perspective transformation")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        
        # Calculate perspective points based on intensity
        max_distortion = min(w, h) * 0.1 * intensity  # Max 10% distortion
        
        # Original corner points
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Randomly distorted corner points
        dst_points = np.float32([
            [np.random.uniform(-max_distortion, max_distortion), 
             np.random.uniform(-max_distortion, max_distortion)],
            [w + np.random.uniform(-max_distortion, max_distortion), 
             np.random.uniform(-max_distortion, max_distortion)],
            [w + np.random.uniform(-max_distortion, max_distortion), 
             h + np.random.uniform(-max_distortion, max_distortion)],
            [np.random.uniform(-max_distortion, max_distortion), 
             h + np.random.uniform(-max_distortion, max_distortion)]
        ])
        
        self.logger.debug(f"Applying perspective: distortion {max_distortion:.2f}")
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(cv_image, matrix, (w, h), borderValue=(255, 255, 255))
        
        # Convert back to PIL
        result = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)

    def apply_elastic(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply elastic deformation."""
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            self.logger.warning("OpenCV and PIL required for elastic deformation")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Parameters for elastic deformation based on intensity
        alpha = 100 * intensity  # Displacement strength
        sigma = 20 * intensity   # Smoothness of displacement field
        
        self.logger.debug(f"Applying elastic deformation: alpha={alpha:.2f}, sigma={sigma:.2f}")
        
        # Generate displacement fields
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32) * alpha
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply elastic deformation
        if len(img_array.shape) == 3:  # Color image
            deformed = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        else:  # Grayscale image
            deformed = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
        
        return Image.fromarray(deformed)

    def apply_color_shift(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply color shift augmentation."""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available for color shift")
            return image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for color manipulation
        img_array = np.array(image).astype(np.float32)
        
        if len(img_array.shape) != 3:  # Skip if not color image
            return image
        
        # Calculate color shift amounts based on intensity
        max_shift = 30 * intensity  # Max shift of 30 color values
        
        # Random shifts for each channel
        r_shift = np.random.uniform(-max_shift, max_shift)
        g_shift = np.random.uniform(-max_shift, max_shift)
        b_shift = np.random.uniform(-max_shift, max_shift)
        
        self.logger.debug(f"Applying color shift: R={r_shift:.1f}, G={g_shift:.1f}, B={b_shift:.1f}")
        
        # Apply shifts
        img_array[:, :, 0] += r_shift  # Red channel
        img_array[:, :, 1] += g_shift  # Green channel
        img_array[:, :, 2] += b_shift  # Blue channel
        
        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def apply_random_augmentation(self, image: Union[Image.Image, np.ndarray], intensity: float = 0.5) -> Image.Image:
        """Apply a random augmentation from available options."""
        available_augs = self.get_available_augmentations()
        if not available_augs:
            return image
        
        aug_type = np.random.choice(available_augs)
        return self._apply_single_augmentation({"image": image}, aug_type, intensity)["image"]

    def get_available_augmentations(self) -> List[str]:
        """Get list of available augmentation types."""
        return self.available_augmentations

    def batch_augment_images(
        self, 
        images: List[Union[Image.Image, np.ndarray]], 
        augmentations: List[str],
        intensity: float = 0.5
    ) -> List[Image.Image]:
        """Apply augmentations to a batch of images."""
        augmented_images = []
        
        for image in images:
            for aug_type in augmentations:
                try:
                    if aug_type == AugmentationType.ROTATION.value:
                        aug_image = self.apply_rotation(image, intensity)
                    elif aug_type == AugmentationType.SCALING.value:
                        aug_image = self.apply_scaling(image, intensity)
                    elif aug_type == AugmentationType.NOISE.value:
                        aug_image = self.apply_noise(image, intensity)
                    elif aug_type == AugmentationType.BRIGHTNESS.value:
                        aug_image = self.apply_brightness(image, intensity)
                    elif aug_type == AugmentationType.CONTRAST.value:
                        aug_image = self.apply_contrast(image, intensity)
                    elif aug_type == AugmentationType.BLUR.value:
                        aug_image = self.apply_blur(image, intensity)
                    elif aug_type == AugmentationType.PERSPECTIVE.value:
                        aug_image = self.apply_perspective(image, intensity)
                    elif aug_type == AugmentationType.ELASTIC.value:
                        aug_image = self.apply_elastic(image, intensity)
                    elif aug_type == AugmentationType.COLOR_SHIFT.value:
                        aug_image = self.apply_color_shift(image, intensity)
                    else:
                        aug_image = image
                    
                    augmented_images.append(aug_image)
                except Exception as e:
                    self.logger.error(f"Failed to apply {aug_type} to image: {e}")
                    augmented_images.append(image)  # Add original if augmentation fails
        
        return augmented_images
