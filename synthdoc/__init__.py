"""
SynthDoc - A comprehensive library for generating synthetic documents.
"""

from .models import (
    RawDocumentGenerationConfig,
    LayoutAugmentationConfig,
    PDFAugmentationConfig,
    VQAGenerationConfig,
    HandwritingGenerationConfig,
    AugmentationType,
    OutputFormat,
    WorkflowResult,
)

from .workflows import (
    BaseWorkflow,
    RawDocumentGenerator,
    LayoutAugmenter,
    PDFAugmenter,
    VQAGenerator,
    HandwritingGenerator,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "RawDocumentGenerationConfig",
    "LayoutAugmentationConfig",
    "PDFAugmentationConfig",
    "VQAGenerationConfig",
    "HandwritingGenerationConfig",
    "AugmentationType",
    "OutputFormat",
    "WorkflowResult",
    # Workflows
    "BaseWorkflow",
    "RawDocumentGenerator",
    "LayoutAugmenter",
    "PDFAugmenter",
    "VQAGenerator",
    "HandwritingGenerator",
]
