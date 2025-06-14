"""
Configuration settings for SynthDoc.

This module contains default configuration values that can be overridden.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DocumentConfig:
    """Document generation configuration."""

    default_language: str = "en"
    max_pages_per_document: int = 10
    default_prompt: str = "Generate diverse document content"

    # Image settings
    image_dpi: int = 300
    image_format: str = "PNG"
    image_quality: int = 95

    # Output settings
    output_format: str = "huggingface"
    save_intermediate: bool = True


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""

    default_augmentations: List[str] = field(
        default_factory=lambda: ["rotation", "scaling"]
    )
    intensity_range: tuple = (0.3, 0.7)

    # Specific augmentation settings
    rotation_range: tuple = (-15, 15)  # degrees
    scale_range: tuple = (0.8, 1.2)
    noise_range: tuple = (0.0, 0.1)
    brightness_range: tuple = (0.8, 1.2)


@dataclass
class VQAConfig:
    """VQA generation configuration."""

    default_question_types: List[str] = field(
        default_factory=lambda: ["factual", "reasoning"]
    )
    default_difficulty_levels: List[str] = field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    hard_negative_ratio: float = 0.3
    questions_per_document: int = 5


@dataclass
class SynthDocConfig:
    """Main configuration class."""

    document: DocumentConfig = field(default_factory=DocumentConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    vqa: VQAConfig = field(default_factory=VQAConfig)

    # Logging
    log_level: str = "INFO"

    # Performance
    max_workers: int = 4
    batch_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "document": {
                "default_language": self.document.default_language,
                "max_pages_per_document": self.document.max_pages_per_document,
                "image_dpi": self.document.image_dpi,
                "image_format": self.document.image_format,
                "output_format": self.document.output_format,
            },
            "augmentation": {
                "default_augmentations": self.augmentation.default_augmentations,
                "intensity_range": self.augmentation.intensity_range,
                "rotation_range": self.augmentation.rotation_range,
                "scale_range": self.augmentation.scale_range,
            },
            "vqa": {
                "default_question_types": self.vqa.default_question_types,
                "hard_negative_ratio": self.vqa.hard_negative_ratio,
                "questions_per_document": self.vqa.questions_per_document,
            },
            "log_level": self.log_level,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
        }


# Default configuration instance
DEFAULT_CONFIG = SynthDocConfig()
