#!/usr/bin/env python3
"""
Minimal workflow examples for SynthDoc.
"""

from synthdoc.models import (
    RawDocumentGenerationConfig,
    LayoutAugmentationConfig,
    PDFAugmentationConfig,
    VQAGenerationConfig,
    HandwritingGenerationConfig,
    AugmentationType,
)
from synthdoc.workflows import (
    RawDocumentGenerator,
    LayoutAugmenter,
    PDFAugmenter,
    VQAGenerator,
    HandwritingGenerator,
)


def example_raw_document_generation():
    """Example: Generate synthetic documents from scratch."""
    config = RawDocumentGenerationConfig(
        language="en",
        num_pages=3,
        augmentations=[AugmentationType.ROTATION, AugmentationType.NOISE],
    )

    generator = RawDocumentGenerator()
    result = generator.process(config)

    print(f"Generated {result.num_samples} samples")
    print(f"Dataset format: {list(result.dataset.keys())}")
    return result


def example_layout_augmentation():
    """Example: Apply layout transformations to documents."""
    config = LayoutAugmentationConfig(
        documents=["document1.pdf", "document2.pdf"],
        languages=["en", "es"],
        fonts=["Arial", "Times New Roman"],
        augmentations=[AugmentationType.SCALING, AugmentationType.COLOR_SHIFT],
    )

    augmenter = LayoutAugmenter()
    result = augmenter.process(config)

    print(f"Augmented {result.num_samples} documents")
    return result


def example_pdf_augmentation():
    """Example: Augment existing PDF files."""
    config = PDFAugmentationConfig(
        pdf_files=["input1.pdf", "input2.pdf"],
        augmentations=[AugmentationType.BLUR, AugmentationType.CROPPING],
        preserve_text=True,
    )

    augmenter = PDFAugmenter()
    result = augmenter.process(config)

    print(f"Processed {result.num_samples} PDF files")
    return result


def example_vqa_generation():
    """Example: Generate VQA dataset."""
    config = VQAGenerationConfig(
        documents=["doc1.pdf", "doc2.pdf"],
        num_questions_per_doc=5,
        include_hard_negatives=True,
        question_types=["factual", "reasoning", "visual"],
    )

    generator = VQAGenerator()
    result = generator.process(config)

    print(f"Generated {result.num_samples} VQA samples")
    return result


def example_handwriting_generation():
    """Example: Generate handwritten documents."""
    config = HandwritingGenerationConfig(
        text_content="Hello world! This is a handwriting sample.",
        handwriting_style="cursive",
        language="en",
        num_samples=10,
        augmentations=[AugmentationType.ROTATION],
    )

    generator = HandwritingGenerator()
    result = generator.process(config)

    print(f"Generated {result.num_samples} handwriting samples")
    return result


if __name__ == "__main__":
    print("=== SynthDoc Workflow Examples ===\n")

    print("1. Raw Document Generation:")
    example_raw_document_generation()
    print()

    print("2. Layout Augmentation:")
    example_layout_augmentation()
    print()

    print("3. PDF Augmentation:")
    example_pdf_augmentation()
    print()

    print("4. VQA Generation:")
    example_vqa_generation()
    print()

    print("5. Handwriting Generation:")
    example_handwriting_generation()
    print()

    print("All workflows complete!")
