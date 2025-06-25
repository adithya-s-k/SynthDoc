from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from pathlib import Path


class AugmentationType(Enum):
    ROTATION = "rotation"
    SCALING = "scaling"
    NOISE = "noise"
    BLUR = "blur"
    COLOR_SHIFT = "color_shift"
    CROPPING = "cropping"


class OutputFormat(Enum):
    HUGGINGFACE = "huggingface"


# Raw Document Generation Workflow
class RawDocumentGenerationConfig(BaseModel):
    """Configuration for generating synthetic documents from scratch using LLMs."""

    language: str = Field(
        default="en", description="Target language for content generation"
    )
    num_pages: int = Field(default=1, ge=1, description="Number of pages to generate")
    prompt: Optional[str] = Field(
        default=None, description="Custom prompt for content generation"
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
    languages: List[str] = Field(
        default=["en"], description="Target languages for text content"
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
    language: str = Field(default="en", description="Language for handwriting")
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
class WorkflowResult(BaseModel):
    """Standard result format for all workflows."""

    dataset: Dict[str, Any] = Field(description="HuggingFace dataset format")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    num_samples: int = Field(description="Number of generated samples")
