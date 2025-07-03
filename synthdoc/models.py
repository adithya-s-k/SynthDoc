"""
Pydantic models for SynthDoc dataset management.

This module contains all the Pydantic models used for type-safe dataset management,
including metadata structures, validation schemas, and configuration models.
"""

from typing import Dict, List, Optional, Union, Any, Literal
from pathlib import Path
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import UUID4
from .languages import Language


class SplitType(str, Enum):
    """Supported dataset splits."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    DEV = "dev"


class MetadataFormat(str, Enum):
    """Supported metadata formats."""

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class DatasetType(str, Enum):
    """Types of datasets supported."""

    IMAGE_CAPTIONING = "image_captioning"
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    VQA = "vqa"
    OCR = "ocr"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    LAYOUT_ANALYSIS = "layout_analysis"
    CUSTOM = "custom"


class BoundingBox(BaseModel):
    """Bounding box for object detection."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., ge=0, description="X coordinate of top-left corner")
    y: float = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: float = Field(..., gt=0, description="Width of the bounding box")
    height: float = Field(..., gt=0, description="Height of the bounding box")


class ObjectDetectionMetadata(BaseModel):
    """Metadata for object detection tasks."""

    model_config = ConfigDict(frozen=True)

    bbox: List[List[float]] = Field(
        ..., description="List of bounding boxes [x, y, width, height]"
    )
    categories: List[int] = Field(
        ..., description="List of category IDs corresponding to bboxes"
    )

    @validator("bbox")
    def validate_bbox_format(cls, v):
        """Validate bounding box format."""
        for box in v:
            if len(box) != 4:
                raise ValueError(
                    "Each bounding box must have exactly 4 values [x, y, width, height]"
                )
            if any(val < 0 for val in box[:2]):  # x, y should be >= 0
                raise ValueError("Bounding box coordinates must be non-negative")
            if any(val <= 0 for val in box[2:]):  # width, height should be > 0
                raise ValueError("Bounding box width and height must be positive")
        return v

    @validator("categories")
    def validate_categories_length(cls, v, values):
        """Validate that categories match bbox count."""
        if "bbox" in values and len(v) != len(values["bbox"]):
            raise ValueError("Number of categories must match number of bounding boxes")
        return v


class ImageCaptioningMetadata(BaseModel):
    """Metadata for image captioning tasks."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="Caption text for the image")


class VQAMetadata(BaseModel):
    """Metadata for Visual Question Answering tasks."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1, description="Question about the image")
    answer: str = Field(..., min_length=1, description="Answer to the question")
    question_type: Optional[str] = Field(
        None, description="Type of question (factual, reasoning, etc.)"
    )
    difficulty: Optional[str] = Field(
        None, description="Difficulty level (easy, medium, hard)"
    )


class ClassificationMetadata(BaseModel):
    """Metadata for classification tasks."""

    model_config = ConfigDict(frozen=True)

    label: Union[str, int] = Field(..., description="Class label")
    confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence score"
    )


class OCRMetadata(BaseModel):
    """Metadata for OCR tasks."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="OCR text content")
    words: Optional[List[Dict[str, Any]]] = Field(
        None, description="Word-level annotations"
    )
    lines: Optional[List[Dict[str, Any]]] = Field(
        None, description="Line-level annotations"
    )
    language: Optional[str] = Field(None, description="Detected language")


class DocumentMetadata(BaseModel):
    """Metadata for document understanding tasks."""

    model_config = ConfigDict(frozen=True)

    text: Optional[str] = Field(None, description="Full text content")
    layout: Optional[Dict[str, Any]] = Field(None, description="Layout structure")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Named entities")
    document_type: Optional[str] = Field(None, description="Type of document")


class BaseItemMetadata(BaseModel):
    """Base metadata that all items must have."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_name: str = Field(..., description="Name of the image file")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    source: Optional[str] = Field(None, description="Source of the data")
    dataset_type: Optional[DatasetType] = Field(None, description="Type of dataset")

    # Task-specific metadata (one of these should be present)
    image_captioning: Optional[ImageCaptioningMetadata] = None
    object_detection: Optional[ObjectDetectionMetadata] = None
    vqa: Optional[VQAMetadata] = None
    classification: Optional[ClassificationMetadata] = None
    ocr: Optional[OCRMetadata] = None
    document: Optional[DocumentMetadata] = None

    # Custom metadata for extensibility
    custom: Optional[Dict[str, Any]] = Field(None, description="Custom metadata fields")


class MultiImageMetadata(BaseModel):
    """Metadata for items with multiple images."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_file_name: Optional[str] = Field(None, description="Input image filename")
    output_file_name: Optional[str] = Field(None, description="Output image filename")
    frames_file_names: Optional[List[str]] = Field(
        None, description="List of frame filenames"
    )

    # Metadata for the sequence/pair
    label: Optional[str] = Field(None, description="Label for the image sequence/pair")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DatasetItem(BaseModel):
    """A single item to be added to the dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Union[str, Path] = Field(..., description="Path to the source image")
    metadata: Union[BaseItemMetadata, MultiImageMetadata, Dict[str, Any]] = Field(
        ..., description="Item metadata"
    )
    split: SplitType = Field(default=SplitType.TRAIN, description="Dataset split")
    custom_filename: Optional[str] = Field(
        None, description="Custom filename for the image"
    )


class DatasetStats(BaseModel):
    """Statistics for a dataset."""

    model_config = ConfigDict(frozen=True)

    split_counts: Dict[str, int] = Field(..., description="Number of items per split")
    total_items: int = Field(..., description="Total number of items")
    dataset_type: Optional[DatasetType] = Field(
        None, description="Primary dataset type"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )


class ValidationResult(BaseModel):
    """Result of dataset validation."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether the dataset is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    stats: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Per-split statistics"
    )


class DatasetConfig(BaseModel):
    """Configuration for dataset creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_root: Path = Field(..., description="Root directory for datasets")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")
    splits: List[SplitType] = Field(
        default_factory=lambda: [SplitType.TRAIN, SplitType.TEST, SplitType.VALIDATION],
        description="Dataset splits to create",
    )
    metadata_format: MetadataFormat = Field(
        default=MetadataFormat.JSONL, description="Format for metadata files"
    )
    auto_create_splits: bool = Field(
        default=True, description="Whether to auto-create split directories"
    )
    copy_images: bool = Field(
        default=True, description="Whether to copy images or create symlinks"
    )
    batch_size: int = Field(default=100, ge=1, description="Batch size for processing")


class HubUploadConfig(BaseModel):
    """Configuration for uploading to HuggingFace Hub."""

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(..., description="Repository ID on HuggingFace Hub")
    private: bool = Field(
        default=False, description="Whether to create a private repository"
    )
    token: Optional[str] = Field(None, description="HuggingFace token")
    commit_message: Optional[str] = Field(None, description="Custom commit message")
    create_pr: bool = Field(
        default=False, description="Whether to create a pull request"
    )


class DatasetSummary(BaseModel):
    """Summary information about a dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_path: Path = Field(..., description="Path to the dataset")
    created_at: datetime = Field(..., description="Creation timestamp")
    splits: List[str] = Field(..., description="Available splits")
    stats: DatasetStats = Field(..., description="Dataset statistics")
    validation: ValidationResult = Field(..., description="Validation results")
    dataset_type: Optional[DatasetType] = Field(
        None, description="Primary dataset type"
    )


# Helper functions for creating typed metadata


def create_image_captioning_metadata(
    file_name: str, caption: str, source: Optional[str] = None, **kwargs
) -> BaseItemMetadata:
    """Create metadata for image captioning tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.IMAGE_CAPTIONING,
        image_captioning=ImageCaptioningMetadata(text=caption),
        source=source,
        custom=kwargs if kwargs else None,
    )


def create_object_detection_metadata(
    file_name: str,
    bboxes: List[List[float]],
    categories: List[int],
    source: Optional[str] = None,
    **kwargs,
) -> BaseItemMetadata:
    """Create metadata for object detection tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.OBJECT_DETECTION,
        object_detection=ObjectDetectionMetadata(bbox=bboxes, categories=categories),
        source=source,
        custom=kwargs if kwargs else None,
    )


def create_vqa_metadata(
    file_name: str,
    question: str,
    answer: str,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    source: Optional[str] = None,
    **kwargs,
) -> BaseItemMetadata:
    """Create metadata for VQA tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.VQA,
        vqa=VQAMetadata(
            question=question,
            answer=answer,
            question_type=question_type,
            difficulty=difficulty,
        ),
        source=source,
        custom=kwargs if kwargs else None,
    )


def create_classification_metadata(
    file_name: str,
    label: Union[str, int],
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    **kwargs,
) -> BaseItemMetadata:
    """Create metadata for classification tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.CLASSIFICATION,
        classification=ClassificationMetadata(label=label, confidence=confidence),
        source=source,
        custom=kwargs if kwargs else None,
    )


def create_ocr_metadata(
    file_name: str,
    text: str,
    words: Optional[List[Dict[str, Any]]] = None,
    lines: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = None,
    source: Optional[str] = None,
    **kwargs,
) -> BaseItemMetadata:
    """Create metadata for OCR tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.OCR,
        ocr=OCRMetadata(text=text, words=words, lines=lines, language=language),
        source=source,
        custom=kwargs if kwargs else None,
    )


def create_document_metadata(
    file_name: str,
    text: Optional[str] = None,
    layout: Optional[Dict[str, Any]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    document_type: Optional[str] = None,
    source: Optional[str] = None,
    **kwargs,
) -> BaseItemMetadata:
    """Create metadata for document understanding tasks."""
    return BaseItemMetadata(
        file_name=file_name,
        dataset_type=DatasetType.DOCUMENT_UNDERSTANDING,
        document=DocumentMetadata(
            text=text, layout=layout, entities=entities, document_type=document_type
        ),
        source=source,
        custom=kwargs if kwargs else None,
    )


class AugmentationType(Enum):
    ROTATION = "rotation"
    SCALING = "scaling"
    NOISE = "noise"
    BLUR = "blur"
    COLOR_SHIFT = "color_shift"
    CROPPING = "cropping"


class LayoutType(Enum):
    SINGLE_COLUMN = "SINGLE_COLUMN"
    TWO_COLUMN = "TWO_COLUMN"
    THREE_COLUMN = "THREE_COLUMN"
    FOUR_COLUMN = "FOUR_COLUMN"
    NEWSLETTER = "NEWSLETTER"
    MAGAZINE = "MAGAZINE"
    ACADEMIC = "ACADEMIC"
    NEWSPAPER = "NEWSPAPER"
    BROCHURE = "BROCHURE"
    REPORT = "REPORT"


class OutputFormat(Enum):
    HUGGINGFACE = "huggingface"


# Raw Document Generation Workflow
class RawDocumentGenerationConfig(BaseModel):
    """Configuration for generating synthetic documents from scratch using LLMs."""

    language: Language = Field(
        default=Language.EN, description="Target language for content generation"
    )
    num_pages: int = Field(default=1, ge=1, description="Number of pages to generate")
    prompt: Optional[str] = Field(
        default=None, description="Custom prompt for content generation"
    )
    layout_type: LayoutType = Field(
        default=LayoutType.SINGLE_COLUMN, description="Document layout type"
    )
    include_graphs: bool = Field(
        default=False, description="Include graphs in generated documents"
    )
    include_tables: bool = Field(
        default=False, description="Include tables in generated documents"
    )
    include_ai_images: bool = Field(
        default=False, description="Include AI-generated images"
    )
    augmentations: Optional[List[AugmentationType]] = Field(
        default=None, description="List of augmentations to apply"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )


# Layout Augmentation Workflow
class LayoutAugmentationConfig(BaseModel):
    """Configuration for applying layout transformations to existing documents."""

    documents: List[Union[str, Path]] = Field(
        description="List of PDF files or images to process"
    )
    languages: List[Language] = Field(
        default=[Language.EN], description="Target languages for text content"
    )
    fonts: Optional[List[str]] = Field(
        default=None, description="Font families to apply"
    )
    augmentations: Optional[List[AugmentationType]] = Field(
        default=None, description="Visual augmentations"
    )
    layout_templates: Optional[List[str]] = Field(
        default=None, description="Predefined layout templates"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )


# PDF Augmentation Workflow
class PDFAugmentationConfig(BaseModel):
    """Configuration for augmenting existing PDF documents."""

    pdf_files: List[Union[str, Path]] = Field(
        description="List of PDF files to augment"
    )
    extraction_elements: Optional[List[str]] = Field(
        default=None, description="Types of elements to extract (text_blocks, headers, images, tables, etc.)"
    )
    combination_strategy: str = Field(
        default="random", description="Method for combining extracted elements"
    )
    num_generated_docs: int = Field(
        default=5, ge=1, description="Number of new documents to generate"
    )
    augmentations: Optional[List[AugmentationType]] = Field(
        default=None, description="Augmentations to apply"
    )
    preserve_text: bool = Field(
        default=True, description="Whether to preserve original text"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )


# VQA Generation Workflow
class VQAGenerationConfig(BaseModel):
    """Configuration for generating visual question-answering datasets."""

    documents: List[Union[str, Path]] = Field(
        description="Source documents for VQA generation"
    )
    num_questions_per_doc: int = Field(
        default=5, ge=1, description="Number of questions per document"
    )
    include_hard_negatives: bool = Field(
        default=True, description="Include hard negative examples"
    )
    question_types: Optional[List[str]] = Field(
        default=None, description="Types of questions to generate"
    )
    difficulty_levels: Optional[List[str]] = Field(
        default=None,
        description="Difficulty levels for generated questions (easy, medium, hard, etc.)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )


# Handwriting Generation Workflow
class HandwritingGenerationConfig(BaseModel):
    """Configuration for generating handwritten documents."""

    text_content: Optional[str] = Field(
        default=None, description="Text content to handwrite"
    )
    handwriting_style: str = Field(
        default="default", description="Handwriting style to use"
    )
    paper_template: str = Field(
        default="lined", description="Paper template (lined, grid, blank)"
    )
    language: Language = Field(default=Language.EN, description="Language for handwriting")
    num_samples: int = Field(
        default=1, ge=1, description="Number of handwriting samples to generate"
    )
    augmentations: Optional[List[AugmentationType]] = Field(
        default=None, description="Augmentations to apply"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )


# Base Workflow Result
class DocumentTranslationConfig(BaseModel):
    """Configuration for document translation workflow."""

    input_images: Optional[List[Union[str, Path]]] = Field(
        default=None, description="List of image files or directories to translate"
    )
    input_dataset: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Input dataset with images to translate"
    )
    target_languages: List[str] = Field(
        default=["hi"], description="Target languages for translation (e.g., ['hi', 'zh', 'fr'])"
    )
    yolo_model_path: str = Field(
        default="/SynthDoc/model-doclayout-yolo.pt",
        description="Path to the YOLO layout detection model"
    )
    font_path: str = Field(
        description="Path to directory containing language-specific fonts"
    )
    confidence_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Confidence threshold for layout detection"
    )
    image_size: int = Field(
        default=1024, gt=0, description="Input image size for YOLO model"
    )
    preserve_layout: bool = Field(
        default=True, description="Whether to preserve original document layout"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HUGGINGFACE, description="Output format"
    )

    @validator("target_languages")
    def validate_languages(cls, v):
        """Validate that at least one target language is specified."""
        if not v:
            raise ValueError("At least one target language must be specified")
        return v

    @validator("yolo_model_path")
    def validate_model_path(cls, v):
        """Validate that YOLO model path exists."""
        if not Path(v).exists():
            # For the default model path, provide a helpful warning instead of failing
            if v == "/SynthDoc/model-doclayout-yolo.pt":
                print(f"⚠️  Warning: Default YOLO model not found at {v}")
                print("   Download the model from: https://huggingface.co/vikp/doclayout-yolo/tree/main")
                print("   Or provide a custom yolo_model_path parameter")
            else:
                raise ValueError(f"YOLO model path does not exist: {v}")
        return v

    @validator("font_path")
    def validate_font_path(cls, v):
        """Validate that font path exists."""
        if not Path(v).exists():
            raise ValueError(f"Font path does not exist: {v}")
        return v

    @validator('input_dataset')
    def validate_input_provided(cls, v, values):
        """Validate that either input_images or input_dataset is provided."""
        input_images = values.get('input_images')
        if not input_images and not v:
            raise ValueError("Either input_images or input_dataset must be provided")
        return v


class WorkflowResult(BaseModel):
    """Standard result format for all workflows."""
    
    model_config = {"arbitrary_types_allowed": True}

    dataset: Any = Field(description="HuggingFace dataset or dict")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    num_samples: int = Field(description="Number of generated samples")
    output_files: List[str] = Field(
        default_factory=list, description="List of generated file paths"
    )
