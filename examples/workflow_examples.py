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
from synthdoc.languages import Language


def example_raw_document_generation():
    """Example: Generate synthetic documents from scratch with standardized output structure."""
    
    # Configuration for raw document generation
    config = RawDocumentGenerationConfig(
        language=Language.EN,
        num_pages=3,
        prompt="Write about artificial intelligence and machine learning advances",
        include_graphs=True,
        include_tables=True,
        include_ai_images=False  # Set to True if you have stable diffusion setup
    )

    # Initialize generator with custom output directory
    generator = RawDocumentGenerator(save_dir="example_raw_documents_output")
    
    # Generate documents
    result = generator.process(config)

    # Display results
    print(f"✅ Generated {result.num_samples} document pages")
    print(f"📊 Dataset columns: {result.dataset.column_names if result.dataset else 'N/A'}")
    print(f"🆔 Document ID: {result.metadata.get('document_id', 'N/A')}")
    print(f"⏱️  Processing time: {result.metadata.get('processing_time', 0):.2f}s")
    print(f"💰 Total cost: ${result.metadata.get('total_cost', 0):.6f}")
    
    # Show output structure
    output_structure = result.metadata.get('output_structure', {})
    print(f"\n📁 Output structure:")
    print(f"   - Output directory: {output_structure.get('output_dir', 'N/A')}")
    print(f"   - Images folder: {output_structure.get('images_dir', 'N/A')}")
    print(f"   - Metadata file: {output_structure.get('metadata_file', 'N/A')}")
    print(f"   - Total images: {output_structure.get('total_images', 0)}")
    
    # Show dataset sample if available
    if result.dataset and len(result.dataset) > 0:
        sample = result.dataset[0]
        print(f"\n📄 Sample metadata:")
        print(f"   - File name: {sample.get('file_name', 'N/A')}")
        print(f"   - Language: {sample.get('language', 'N/A')}")
        print(f"   - Page number: {sample.get('page_number', 'N/A')}")
        print(f"   - Layout type: {sample.get('layout_type', 'N/A')}")
        print(f"   - Has graphs: {sample.get('has_graphs', False)}")
        print(f"   - Has tables: {sample.get('has_tables', False)}")
    
    print(f"\n💡 The output follows the standardized structure:")
    print(f"   - images/ folder containing all generated document images")
    print(f"   - metadata.jsonl file with one JSON entry per image")
    print(f"   - Compatible with HuggingFace datasets")
    
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

    print("1. Raw Document Generation (Standardized):")
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
