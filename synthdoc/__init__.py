"""
SynthDoc - A library for generating synthetic documents for ML training.

A comprehensive library for generating synthetic documents designed for training
and evaluating models in document understanding tasks.
"""

from .core import SynthDoc
from .languages import LanguageSupport
from .generators import DocumentGenerator, LayoutGenerator, VQAGenerator
from .augmentations import Augmentor
from .config import SynthDocConfig, DEFAULT_CONFIG
from .fonts import FontManager
from .dataset_manager import DatasetManager, create_dataset_manager
from .workflows import (
    WorkflowDatasetBuilder,
    ImageCaptioningWorkflow,
    VQAWorkflow,
    OCRWorkflow,
    DocumentUnderstandingWorkflow,
    ObjectDetectionWorkflow,
    ClassificationWorkflow,
    create_image_captioning_workflow,
    create_vqa_workflow,
    create_ocr_workflow,
    create_document_workflow,
    create_detection_workflow,
    create_classification_workflow,
)
from .models import (
    DatasetConfig,
    DatasetItem,
    SplitType,
    DatasetType,
    MetadataFormat,
    HubUploadConfig,
    create_image_captioning_metadata,
    create_object_detection_metadata,
    create_vqa_metadata,
    create_classification_metadata,
    create_ocr_metadata,
    create_document_metadata,
)

__version__ = "0.1.0"
__author__ = "Adithya Skolavi"

__all__ = [
    # Core components
    "SynthDoc",
    "LanguageSupport",
    "DocumentGenerator",
    "LayoutGenerator",
    "VQAGenerator",
    "Augmentor",
    "SynthDocConfig",
    "DEFAULT_CONFIG",
    "FontManager",
    # Dataset management
    "DatasetManager",
    "create_dataset_manager",
    # Workflow classes
    "WorkflowDatasetBuilder",
    "ImageCaptioningWorkflow",
    "VQAWorkflow",
    "OCRWorkflow",
    "DocumentUnderstandingWorkflow",
    "ObjectDetectionWorkflow",
    "ClassificationWorkflow",
    # Workflow factory functions
    "create_image_captioning_workflow",
    "create_vqa_workflow",
    "create_ocr_workflow",
    "create_document_workflow",
    "create_detection_workflow",
    "create_classification_workflow",
    # Models and types
    "DatasetConfig",
    "DatasetItem",
    "SplitType",
    "DatasetType",
    "MetadataFormat",
    "HubUploadConfig",
    # Metadata factory functions
    "create_image_captioning_metadata",
    "create_object_detection_metadata",
    "create_vqa_metadata",
    "create_classification_metadata",
    "create_ocr_metadata",
    "create_document_metadata",
]
