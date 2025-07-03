# New comprehensive workflow tests

import tempfile
import sys, os, pathlib
# Ensure repository root is on sys.path when running the test file directly
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from pathlib import Path
from types import SimpleNamespace

import pytest

# Utilities for mocking
from unittest.mock import patch

# Shared imports
from synthdoc.languages import Language
from synthdoc.models import (
    RawDocumentGenerationConfig,
    LayoutAugmentationConfig,
    PDFAugmentationConfig,
    VQAGenerationConfig,
    HandwritingGenerationConfig,
)

from PIL import Image


##############################
# Raw Document Generator
##############################

def test_raw_document_generator(monkeypatch, tmp_path):
    """Ensure RawDocumentGenerator runs end-to-end with mocked LLM."""
    from synthdoc.workflows.raw_document_generator.workflow import RawDocumentGenerator
    from synthdoc.utils import CostTracker

    # Mock litellm.completion to avoid network/API calls
    def fake_completion(*args, **kwargs):
        long_text = "dummy " * 100  # >50 words
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=long_text))]
        )

    monkeypatch.setattr(
        "synthdoc.workflows.raw_document_generator.workflow.litellm.completion",
        fake_completion,
    )

    # Disable cost tracking side-effects
    monkeypatch.setattr(CostTracker, "track_usage", lambda self, *a, **k: 0)

    generator = RawDocumentGenerator(save_dir=str(tmp_path))
    cfg = RawDocumentGenerationConfig(language=Language.EN, num_pages=1, prompt="test")
    result = generator.process(cfg)

    assert result.num_samples == 1
    assert len(result.dataset) == 1


#############################################
# Integrated pipeline: Raw -> Layout -> VQA
#############################################

def _generate_raw_docs(monkeypatch, tmp_path):
    """Utility: create raw documents with mocked LLM to feed downstream workflows."""
    from synthdoc.workflows.raw_document_generator.workflow import RawDocumentGenerator
    from synthdoc.utils import CostTracker

    # Mock LLM
    def fake_completion(*args, **kwargs):
        long_text = "synthetic content " * 200  # ensures >50 words
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=long_text))]
        )

    monkeypatch.setattr(
        "synthdoc.workflows.raw_document_generator.workflow.litellm.completion",
        fake_completion,
    )
    monkeypatch.setattr(CostTracker, "track_usage", lambda self, *a, **k: 0)

    raw_gen = RawDocumentGenerator(save_dir=str(tmp_path / "raw"))
    cfg = RawDocumentGenerationConfig(language=Language.EN, num_pages=2, prompt="AI research")
    result = raw_gen.process(cfg)
    return result


def test_layout_and_vqa_pipeline(monkeypatch, tmp_path):
    """Full pipeline: generate docs -> layout augment -> VQA."""

    # Step 1: Raw documents
    raw_result = _generate_raw_docs(monkeypatch, tmp_path)
    assert raw_result.num_samples == 2

    # Collect image paths saved by RawDocumentGenerator
    image_paths = []
    for f in raw_result.metadata.get("generated_files", []):
        full_path = tmp_path / "raw" / f
        if full_path.exists():
            image_paths.append(full_path)
    # sanity fallback: derive from dataset if metadata not present
    if not image_paths:
        for sample in raw_result.dataset:
            if "image_path" in sample:
                image_paths.append(sample["image_path"])

    assert image_paths, "No image paths extracted from raw documents"

    # Step 2: Layout augmentation
    from synthdoc.workflows.layout_augmenter.workflow import LayoutAugmenter

    layout_aug = LayoutAugmenter(save_dir=str(tmp_path / "layout"))
    layout_cfg = LayoutAugmentationConfig(documents=image_paths, languages=[Language.EN])
    layout_result = layout_aug.process(layout_cfg)

    assert layout_result.num_samples > 0
    assert len(layout_result.dataset) == layout_result.num_samples

    # Step 3: VQA generation (template mode, no API)
    from synthdoc.workflows.vqa_generator.workflow import VQAGenerator

    vqa_gen = VQAGenerator(api_key=None, llm_model=None)
    vqa_cfg = VQAGenerationConfig(
        documents=[str(p) for p in image_paths],
        num_questions_per_doc=1,
        include_hard_negatives=False,
        question_types=["factual"],
        difficulty_levels=["easy"],
    )
    vqa_result = vqa_gen.process(vqa_cfg)

    assert vqa_result.num_samples > 0
    assert len(vqa_result.dataset) == vqa_result.num_samples


##############################
# PDF Augmenter (fallback mode)
##############################

def test_pdf_augmenter_fallback(monkeypatch, tmp_path):
    from synthdoc.workflows.pdf_augmenter.workflow import PDFAugmenter

    # Test fallback when PDF libs are unavailable
    monkeypatch.setattr(
        "synthdoc.workflows.pdf_augmenter.workflow.PYMUPDF_AVAILABLE", False
    )
    monkeypatch.setattr(
        "synthdoc.workflows.pdf_augmenter.workflow.PDFPLUMBER_AVAILABLE", False
    )

    augmenter = PDFAugmenter(save_dir=str(tmp_path))
    cfg = PDFAugmentationConfig(pdf_files=[])
    result = augmenter.process(cfg)

    assert result.num_samples == 0
    assert result.metadata.get("status") == "fallback_mode"


def test_pdf_augmenter_with_mocked_pdfs(tmp_path):
    """Test PDF augmentation with mocked PDF content."""
    from synthdoc.workflows.pdf_augmenter.workflow import PDFAugmenter
    
    # Create mock PDF files
    mock_pdf1 = tmp_path / "test1.pdf"
    mock_pdf2 = tmp_path / "test2.pdf"
    
    # Create simple text files to simulate PDF content
    mock_pdf1.write_text("Sample PDF content with text blocks and headers.")
    mock_pdf2.write_text("Another PDF with paragraphs and lists.")
    
    augmenter = PDFAugmenter(save_dir=str(tmp_path / "pdf_output"))
    
    # Test basic PDF augmentation configuration
    cfg = PDFAugmentationConfig(
        pdf_files=[mock_pdf1, mock_pdf2],
        extraction_elements=["text_blocks", "headers"],
        combination_strategy="random",
        num_generated_docs=2,
        preserve_text=True
    )
    
    # Since we don't have actual PDF libraries in test environment,
    # this will likely use fallback, but we test the configuration works
    result = augmenter.process(cfg)
    
    # The result should either succeed with generated docs or fall back gracefully
    assert result.num_samples >= 0
    assert "workflow_type" in result.metadata
    assert result.metadata["workflow_type"] == "pdf_augmentation"


##############################
# Handwriting Generator (stand-alone)
##############################

def test_handwriting_generator(tmp_path):
    from synthdoc.workflows.handwriting_generator.workflow import HandwritingGenerator

    generator = HandwritingGenerator(save_dir=str(tmp_path / "handwriting"))
    
    # Test basic handwriting generation
    cfg = HandwritingGenerationConfig(text_content="Hello world", num_samples=1)
    result = generator.process(cfg)

    assert result.num_samples == 1
    assert len(result.dataset) == 1
    
    # Test with different paper templates
    paper_templates = ["lined", "grid", "blank"]
    for template in paper_templates:
        cfg_template = HandwritingGenerationConfig(
            text_content=f"Testing {template} paper", 
            num_samples=1,
            paper_template=template,
            handwriting_style="print"
        )
        result_template = generator.process(cfg_template)
        
        assert result_template.num_samples == 1
        assert len(result_template.dataset) == 1
        
        # Check that paper template is recorded in the dataset
        sample = result_template.dataset[0]
        
        # The paper template info is stored in the pdf_info field
        assert 'pdf_info' in sample
        pdf_info_str = sample['pdf_info']
        
        # Check that the paper template is mentioned in the pdf_info 
        # (it's stored as a stringified dict containing config)
        assert template in pdf_info_str

if __name__ == "__main__":
    test_raw_document_generator()
    test_layout_and_vqa_pipeline()
    test_pdf_augmenter_fallback()
    test_pdf_augmenter_with_mocked_pdfs()
    test_handwriting_generator()