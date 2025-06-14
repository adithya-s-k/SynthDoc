"""
Test suite for SynthDoc library.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from synthdoc import SynthDoc, LanguageSupport
from synthdoc.languages import LanguageInfo, ScriptType


class TestLanguageSupport:
    """Test language support functionality."""

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = LanguageSupport.get_supported_languages()
        assert len(languages) == 22
        assert "en" in languages
        assert "hi" in languages
        assert "zh" in languages

    def test_get_language_info(self):
        """Test getting language information."""
        en_info = LanguageSupport.get_language("en")
        assert en_info is not None
        assert en_info.code == "en"
        assert en_info.name == "English"
        assert en_info.script == ScriptType.LATIN
        assert en_info.category == "Base"
        assert not en_info.rtl

    def test_get_languages_by_category(self):
        """Test filtering languages by category."""
        indic_languages = LanguageSupport.get_languages_by_category("Indic")
        assert len(indic_languages) > 0
        assert "hi" in indic_languages
        assert "kn" in indic_languages

        base_languages = LanguageSupport.get_languages_by_category("Base")
        assert len(base_languages) == 1
        assert "en" in base_languages

    def test_rtl_language_detection(self):
        """Test RTL language detection."""
        assert not LanguageSupport.is_rtl_language("en")
        assert not LanguageSupport.is_rtl_language("hi")
        assert LanguageSupport.is_rtl_language("ar")

    def test_get_default_fonts(self):
        """Test getting default fonts for languages."""
        en_fonts = LanguageSupport.get_default_fonts("en")
        assert len(en_fonts) > 0
        assert "Arial" in en_fonts

        hi_fonts = LanguageSupport.get_default_fonts("hi")
        assert len(hi_fonts) > 0
        assert any("Devanagari" in font for font in hi_fonts)


class TestSynthDoc:
    """Test main SynthDoc class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.synth = SynthDoc(output_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test SynthDoc initialization."""
        assert self.synth.output_dir.exists()
        assert self.synth.language_support is not None
        assert self.synth.doc_generator is not None

    def test_generate_raw_docs(self):
        """Test raw document generation."""
        docs = self.synth.generate_raw_docs(
            language="en", num_pages=2, prompt="Test prompt"
        )

        assert len(docs) == 2
        assert all(doc["language"] == "en" for doc in docs)
        assert all("content" in doc for doc in docs)
        assert all("metadata" in doc for doc in docs)

    def test_generate_raw_docs_invalid_language(self):
        """Test error handling for invalid language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            self.synth.generate_raw_docs(language="invalid")

    def test_get_supported_languages(self):
        """Test getting supported languages through SynthDoc."""
        languages = self.synth.get_supported_languages()
        assert len(languages) == 22
        assert "en" in languages

    def test_get_language_info(self):
        """Test getting language info through SynthDoc."""
        en_info = self.synth.get_language_info("en")
        assert en_info is not None
        assert en_info["code"] == "en"
        assert en_info["name"] == "English"
        assert en_info["script"] == "Latin"

        invalid_info = self.synth.get_language_info("invalid")
        assert invalid_info is None

    def test_augment_layout(self):
        """Test layout augmentation."""
        # Generate some test documents first
        docs = self.synth.generate_raw_docs(language="en", num_pages=1)

        dataset = self.synth.augment_layout(
            documents=docs,
            languages=["en"],
            fonts=["Arial"],
            augmentations=["rotation"],
        )

        assert "images" in dataset
        assert "annotations" in dataset
        assert "metadata" in dataset

    def test_generate_vqa(self):
        """Test VQA generation."""
        # Generate some test documents first
        docs = self.synth.generate_raw_docs(language="en", num_pages=1)

        vqa_data = self.synth.generate_vqa(
            source_documents=docs,
            question_types=["factual"],
            difficulty_levels=["easy"],
        )

        assert "questions" in vqa_data
        assert "answers" in vqa_data
        assert "metadata" in vqa_data

    def test_generate_handwriting(self):
        """Test handwriting generation."""
        handwritten = self.synth.generate_handwriting(
            content="Test content", language="en", writing_style="print"
        )

        assert "content" in handwritten
        assert "style" in handwritten
        assert "language" in handwritten
        assert handwritten["language"] == "en"


class TestAugmentations:
    """Test augmentation functionality."""

    def setup_method(self):
        """Setup test environment."""
        from synthdoc.augmentations import Augmentor

        self.augmentor = Augmentor()

    def test_available_augmentations(self):
        """Test getting available augmentations."""
        augmentations = self.augmentor.get_available_augmentations()
        assert len(augmentations) > 0
        assert "rotation" in augmentations
        assert "scaling" in augmentations
        assert "noise" in augmentations

    def test_apply_augmentations(self):
        """Test applying augmentations."""
        test_docs = [
            {"id": "doc_001", "content": "Test content 1"},
            {"id": "doc_002", "content": "Test content 2"},
        ]

        augmented = self.augmentor.apply_augmentations(
            documents=test_docs, augmentations=["rotation", "scaling"], intensity=0.5
        )

        # Should have 2 docs × 2 augmentations = 4 results
        assert len(augmented) == 4
        assert all("augmentation" in doc for doc in augmented)


if __name__ == "__main__":
    pytest.main([__file__])
