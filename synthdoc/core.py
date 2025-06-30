"""
Core SynthDoc class - Main entry point for the library.

This module provides the main SynthDoc class that orchestrates all document generation workflows.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .languages import LanguageSupport
from .generators import DocumentGenerator, LayoutGenerator, VQAGenerator
from .augmentations import Augmentor
from .utils import setup_logging
from .config import load_env_config, get_api_key, get_llm_model


class SynthDoc:
    """
    Main SynthDoc class for synthetic document generation.

    This class provides a unified interface for all document generation workflows:
    - Raw document generation
    - Layout augmentation
    - PDF augmentation
    - VQA generation
    - Handwriting synthesis
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        load_dotenv: bool = True,
        env_file: Optional[str] = None,
    ):
        """
        Initialize SynthDoc with automatic .env loading.

        Args:
            output_dir: Directory for output files (default from .env or ./output)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            llm_model: Model name for LiteLLM (auto-detected from .env if None)
            api_key: API key for the model provider (auto-detected from .env if None)
            load_dotenv: Whether to load from .env file (default: True)
            env_file: Specific .env file path (auto-detected if None)
        """
        # Load environment configuration
        if load_dotenv:
            self.config = load_env_config(env_file)
        else:
            from .config import SynthDocConfig
            self.config = SynthDocConfig()

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.default_output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = setup_logging(log_level)

        # Determine API key and model
        if api_key is None:
            api_key = get_api_key("auto")
        
        if llm_model is None:
            llm_model = get_llm_model("auto")
        
        # Store for later use
        self.api_key = api_key
        self.llm_model = llm_model

        # Initialize components with LLM configuration
        self.language_support = LanguageSupport()
        self.doc_generator = DocumentGenerator(model=llm_model, api_key=api_key)
        self.layout_generator = LayoutGenerator()
        self.vqa_generator = VQAGenerator(model=llm_model, api_key=api_key)
        self.augmentor = Augmentor()

        # Log initialization status
        if api_key:
            self.logger.info(f"SynthDoc initialized successfully with model: {llm_model}")
        else:
            self.logger.warning("SynthDoc initialized without API key - limited functionality")
            self.logger.info("Set API keys in .env file for full LLM features")

    def generate_raw_docs(
        self,
        language: str = "en",
        num_pages: int = 1,
        prompt: Optional[str] = None,
        augmentations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic documents from scratch using LLMs.

        Args:
            language: Target language code (e.g., 'en', 'hi', 'zh')
            num_pages: Number of pages to generate
            prompt: Custom prompt for content generation
            augmentations: List of augmentation techniques

        Returns:
            List of generated documents with metadata
        """
        self.logger.info(f"Generating {num_pages} raw documents in {language}")

        # Validate language
        if not self.language_support.get_language(language):
            raise ValueError(f"Unsupported language: {language}")

        return self.doc_generator.generate_raw_documents(
            language=language,
            num_pages=num_pages,
            prompt=prompt,
            augmentations=augmentations or [],
        )

    def augment_layout(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        document_paths: Optional[List[Union[str, Path]]] = None,
        languages: Optional[List[str]] = None,
        fonts: Optional[List[str]] = None,
        augmentations: Optional[List[str]] = None,
        layout_templates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply layout transformations to documents.

        Args:
            documents: Pre-generated documents
            document_paths: Paths to existing documents
            languages: Target languages
            fonts: Font families to apply
            augmentations: Visual augmentation techniques
            layout_templates: Predefined layout templates

        Returns:
            HuggingFace dataset with layout annotations
        """
        self.logger.info("Starting layout augmentation")

        return self.layout_generator.augment_layouts(
            documents=documents,
            document_paths=document_paths,
            languages=languages or ["en"],
            fonts=fonts,
            augmentations=augmentations or [],
            layout_templates=layout_templates,
        )

    def augment_pdfs(
        self,
        corpus_paths: List[Union[str, Path]],
        extraction_elements: Optional[List[str]] = None,
        combination_strategy: str = "random",
        output_layout_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create new documents by recombining elements from existing documents.

        Args:
            corpus_paths: Paths to source documents
            extraction_elements: Types of elements to extract
            combination_strategy: Method for combining elements
            output_layout_types: Target layout styles

        Returns:
            HuggingFace dataset with recombined documents
        """
        self.logger.info("Starting PDF augmentation")

        return self.layout_generator.augment_pdfs(
            corpus_paths=corpus_paths,
            extraction_elements=extraction_elements or ["text", "images", "tables"],
            combination_strategy=combination_strategy,
            output_layout_types=output_layout_types,
        )

    def generate_vqa(
        self,
        source_documents: List[Dict[str, Any]],
        question_types: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None,
        hard_negative_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate VQA datasets with hard negatives.

        Args:
            source_documents: Documents to generate questions about
            question_types: Types of questions to generate
            difficulty_levels: Question complexity levels
            hard_negative_ratio: Ratio of hard negative examples

        Returns:
            Extended dataset with VQA annotations
        """
        self.logger.info("Generating VQA dataset")

        return self.vqa_generator.generate_vqa_dataset(
            documents=source_documents,
            question_types=question_types or ["factual", "reasoning"],
            difficulty_levels=difficulty_levels or ["easy", "medium", "hard"],
            hard_negative_ratio=hard_negative_ratio,
        )

    def generate_handwriting(
        self,
        content: Optional[str] = None,
        language: str = "en",
        handwriting_template: Optional[str] = None,
        writing_style: str = "print",
        paper_template: str = "blank",
    ) -> Dict[str, Any]:
        """
        Generate handwritten documents.

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style

        Returns:
            HuggingFace dataset with handwriting samples
        """
        self.logger.info("Generating handwriting documents")
        
        # For now, return a placeholder
        return {
            "dataset": {},
            "metadata": {
                "handwriting_style": writing_style,
                "paper_template": paper_template,
                "language": language
            }
        }
    
    def create_handwriting(
        self,
        content: Optional[str] = None,
        language: str = "en",
        handwriting_template: Optional[str] = None,
        writing_style: str = "print",
        paper_template: str = "blank",
    ) -> Dict[str, Any]:
        """
        Generate handwritten documents (alias for generate_handwriting).

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style

        Returns:
            HuggingFace dataset with handwriting samples
        """
        return self.generate_handwriting(
            content=content,
            language=language,
            handwriting_template=handwriting_template,
            writing_style=writing_style,
            paper_template=paper_template
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return self.language_support.get_supported_languages()

    def get_language_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a language."""
        lang_info = self.language_support.get_language(code)
        if not lang_info:
            return None
        
        return {
            "code": lang_info.code,
            "name": lang_info.name,
            "script": lang_info.script.value,
            "category": lang_info.category,
            "rtl": lang_info.rtl,
            "fonts": lang_info.font_families,
        }
