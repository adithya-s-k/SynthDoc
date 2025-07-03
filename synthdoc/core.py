"""
Core SynthDoc class - Main entry point for the library.

This module provides the main SynthDoc class that orchestrates all document generation workflows.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .languages import LanguageSupport, Language
from .workflows import RawDocumentGenerator, LayoutAugmenter, VQAGenerator, HandwritingGenerator
from .workflows.document_translator import DocumentTranslator
from .models import (
    RawDocumentGenerationConfig, LayoutAugmentationConfig, 
    VQAGenerationConfig, HandwritingGenerationConfig,
    DocumentTranslationConfig, AugmentationType,
)
from .utils import setup_logging
from .config import load_env_config, get_api_key, get_llm_model
from datasets import Dataset
from .config import SynthDocConfig


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
        config: "SynthDocConfig" = None,
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
            config: Optional SynthDocConfig instance
        """
        # Load configuration (priority: explicit config param -> .env -> defaults)
        if config is not None:
            # User supplied a SynthDocConfig instance in code (as shown in QUICKSTART.md)
            self.config = config
        else:
            if load_dotenv:
                self.config = load_env_config(env_file)
            else:
                from .config import SynthDocConfig as _SDC
                self.config = _SDC()

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

        # Initialize workflows with proper configuration
        self.language_support = LanguageSupport()
        self.raw_doc_generator = RawDocumentGenerator(api_key, str(self.output_dir / "raw_docs"))
        # Backwards-compatibility alias (tests & older code reference self.doc_generator)
        self.doc_generator = self.raw_doc_generator
        self.layout_augmenter = LayoutAugmenter(str(self.output_dir / "layout"))
        self.vqa_generator = VQAGenerator(api_key, llm_model)
        self.handwriting_generator = HandwritingGenerator(str(self.output_dir / "handwriting"))
        self.document_translator = DocumentTranslator(str(self.output_dir / "document_translation"))

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
    ) -> Dataset:
        """
        Generate synthetic documents from scratch using LLMs.

        Args:
            language: Target language code (e.g., 'en', 'hi', 'zh')
            num_pages: Number of pages to generate
            prompt: Custom prompt for content generation
            augmentations: List of augmentation techniques

        Returns:
            HuggingFace Dataset with comprehensive document schema
        """
        self.logger.info(f"Generating {num_pages} raw documents in {language}")

        # Validate language
        if not self.language_support.get_language(language):
            raise ValueError(f"Unsupported language: {language}")

        # Convert string language to Language enum
        try:
            lang_enum = Language(language.upper())
        except ValueError:
            # Fallback for common language codes
            lang_mapping = {
                'en': Language.EN, 'hi': Language.HI, 'zh': Language.ZH,
                'es': Language.ES, 'fr': Language.FR, 'ar': Language.AR
            }
            lang_enum = lang_mapping.get(language.lower(), Language.EN)

        # Convert augmentation strings to enums if provided
        aug_enums = None
        if augmentations:
            aug_enums = []
            for aug in augmentations:
                try:
                    aug_enums.append(AugmentationType(aug.lower()))
                except ValueError:
                    # Ignore unknown augmentation types but warn
                    self.logger.warning(f"Unknown augmentation type ignored: {aug}")

        # Create configuration
        config = RawDocumentGenerationConfig(
            language=lang_enum,
            num_pages=num_pages,
            prompt=prompt,
            augmentations=aug_enums,
        )

        # Use workflow to generate documents
        result = self.raw_doc_generator.process(config)
        return result.dataset

    def augment_layout(
        self,
        documents: Optional[Union[List[Dict[str, Any]], Dataset]] = None,
        document_paths: Optional[List[Union[str, Path]]] = None,
        languages: Optional[List[str]] = None,
        fonts: Optional[List[str]] = None,
        augmentations: Optional[List[str]] = None,
        layout_templates: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Apply layout transformations to documents.

        Args:
            documents: Pre-generated documents (list or HuggingFace Dataset)
            document_paths: Paths to existing documents
            languages: Target languages
            fonts: Font families to apply
            augmentations: Visual augmentation techniques
            layout_templates: Predefined layout templates

        Returns:
            HuggingFace Dataset with comprehensive layout annotations schema
        """
        self.logger.info("Starting layout augmentation")

        # Convert languages to Language enums
        lang_enums = []
        if languages:
            for lang in languages:
                try:
                    lang_enum = Language(lang.upper())
                except ValueError:
                    lang_mapping = {
                        'en': Language.EN, 'hi': Language.HI, 'zh': Language.ZH,
                        'es': Language.ES, 'fr': Language.FR, 'ar': Language.AR
                    }
                    lang_enum = lang_mapping.get(lang.lower(), Language.EN)
                lang_enums.append(lang_enum)
        else:
            lang_enums = [Language.EN]

        # Handle documents input - extract paths if it's a Dataset
        doc_paths = document_paths or []
        if documents is not None:
            if isinstance(documents, Dataset):
                # Extract image paths from dataset
                for i, sample in enumerate(documents):
                    if isinstance(sample, dict) and 'image' in sample and hasattr(sample['image'], 'save'):
                        # Create temp path for the image
                        temp_path = self.output_dir / "temp" / f"doc_{i}.png"
                        temp_path.parent.mkdir(exist_ok=True)
                        sample['image'].save(temp_path)
                        doc_paths.append(str(temp_path))
                    elif isinstance(sample, dict) and 'image_path' in sample:
                        doc_paths.append(sample['image_path'])
            else:
                # Handle list of documents
                for i, doc in enumerate(documents):
                    if isinstance(doc, dict):
                        if 'image_path' in doc:
                            doc_paths.append(doc['image_path'])
                        elif 'image' in doc and hasattr(doc['image'], 'save'):
                            temp_path = self.output_dir / "temp" / f"doc_{i}.png"
                            temp_path.parent.mkdir(exist_ok=True)
                            doc['image'].save(temp_path)
                            doc_paths.append(str(temp_path))

        # Map augmentation strings to enums
        aug_enums = None
        if augmentations:
            aug_enums = []
            for aug in augmentations:
                try:
                    aug_enums.append(AugmentationType(aug.lower()))
                except ValueError:
                    self.logger.warning(f"Unknown augmentation type ignored: {aug}")

        # Create configuration
        config = LayoutAugmentationConfig(
            documents=doc_paths,
            languages=lang_enums,
            fonts=fonts or ["Arial", "Times New Roman"],
            augmentations=aug_enums,
        )

        # Use workflow to augment layouts
        result = self.layout_augmenter.process(config)
        return result.dataset

    def augment_pdfs(
        self,
        corpus_paths: List[Union[str, Path]],
        extraction_elements: Optional[List[str]] = None,
        combination_strategy: str = "random",
        output_layout_types: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Create new documents by recombining elements from existing documents.

        Args:
            corpus_paths: Paths to source documents
            extraction_elements: Types of elements to extract
            combination_strategy: Method for combining elements
            output_layout_types: Target layout styles

        Returns:
            HuggingFace Dataset with comprehensive recombined document schema
        """
        self.logger.info("Starting PDF augmentation")

        # Convert paths to Path objects
        pdf_paths = [Path(p) for p in corpus_paths]
        
        # Import PDF augmenter if not already imported
        from .workflows.pdf_augmenter.workflow import PDFAugmenter
        from .models import PDFAugmentationConfig, AugmentationType
        
        # Create PDF augmenter
        pdf_augmenter = PDFAugmenter(save_dir=str(self.output_dir / "pdf_augmentation"))
        
        # Map extraction elements to supported types
        supported_elements = extraction_elements or ["text_blocks", "headers", "paragraphs"]
        
        # Map output layout types to augmentation types
        aug_enums = []
        if output_layout_types:
            for layout_type in output_layout_types:
                try:
                    if layout_type.lower() in ["rotation", "scaling", "brightness", "noise"]:
                        aug_enums.append(AugmentationType(layout_type.lower()))
                except ValueError:
                    self.logger.warning(f"Unknown augmentation type ignored: {layout_type}")
        
        # Create configuration
        config = PDFAugmentationConfig(
            pdf_files=pdf_paths,
            extraction_elements=supported_elements,
            combination_strategy=combination_strategy,
            num_generated_docs=min(10, len(pdf_paths) * 2),  # Generate 2 docs per input by default
            preserve_text=True,
            augmentations=aug_enums
        )

        # Use workflow to augment PDFs
        try:
            result = pdf_augmenter.process(config)
            self.logger.info(f"Successfully generated {result.num_samples} PDF-augmented documents")
            return result.dataset
        except Exception as e:
            self.logger.error(f"PDF augmentation failed: {e}")
            # Return fallback dataset with error information
            from datasets import Dataset
            return Dataset.from_dict({
                'error': [str(e)],
                'note': ['PDF augmentation failed - check PDF processing library installation'],
                'fallback': [True]
            })

    def generate_vqa(
        self,
        source_documents: Union[List[Dict[str, Any]], Dataset],
        question_types: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None,
        hard_negative_ratio: float = 0.3,
    ) -> Dataset:
        """
        Generate VQA datasets with hard negatives.

        Args:
            source_documents: Documents to generate questions about (list or Dataset)
            question_types: Types of questions to generate
            difficulty_levels: Question complexity levels
            hard_negative_ratio: Ratio of hard negative examples

        Returns:
            HuggingFace Dataset with comprehensive VQA schema
        """
        self.logger.info("Generating VQA dataset")

        # Extract document paths from various input formats
        doc_paths = []
        if isinstance(source_documents, Dataset):
            # Extract paths from HuggingFace Dataset
            for i, sample in enumerate(source_documents):
                if 'image' in sample and hasattr(sample['image'], 'save'):
                    temp_path = self.output_dir / "temp" / f"vqa_doc_{i}.png"
                    temp_path.parent.mkdir(exist_ok=True)
                    sample['image'].save(temp_path)
                    doc_paths.append(str(temp_path))
                elif 'image_path' in sample:
                    doc_paths.append(sample['image_path'])
        else:
            # Handle list of documents
            for i, doc in enumerate(source_documents):
                if 'image_path' in doc:
                    doc_paths.append(doc['image_path'])
                elif 'image' in doc and hasattr(doc['image'], 'save'):
                    temp_path = self.output_dir / "temp" / f"vqa_doc_{i}.png"
                    temp_path.parent.mkdir(exist_ok=True)
                    doc['image'].save(temp_path)
                    doc_paths.append(str(temp_path))

        # Create configuration
        config = VQAGenerationConfig(
            documents=doc_paths,
            question_types=question_types or ["factual", "reasoning", "comparative"],
            include_hard_negatives=hard_negative_ratio > 0,
            num_questions_per_doc=5,  # Default from README
            difficulty_levels=difficulty_levels,
        )

        # Use workflow to generate VQA
        result = self.vqa_generator.process(config)
        return result.dataset

    def generate_handwriting(
        self,
        content: Optional[str] = None,
        language: str = "en",
        handwriting_template: Optional[str] = None,
        writing_style: str = "print",
        paper_template: str = "blank",
    ) -> Dataset:
        """
        Generate handwritten documents.

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style (lined, grid, blank)

        Returns:
            HuggingFace Dataset with comprehensive handwriting schema
        """
        self.logger.info("Generating handwriting documents")
        
        # Convert string language to Language enum
        try:
            lang_enum = Language(language.upper())
        except ValueError:
            lang_mapping = {
                'en': Language.EN, 'hi': Language.HI, 'zh': Language.ZH,
                'es': Language.ES, 'fr': Language.FR, 'ar': Language.AR
            }
            lang_enum = lang_mapping.get(language.lower(), Language.EN)

        # Create configuration
        config = HandwritingGenerationConfig(
            text_content=content or "Sample handwriting text for document generation.",
            language=lang_enum,
            handwriting_style=writing_style,
            paper_template=paper_template,
            num_samples=1  # Default to 1 sample
        )

        # Use workflow to generate handwriting
        result = self.handwriting_generator.process(config)
        return result.dataset
    
    def create_handwriting(
        self,
        content: Optional[str] = None,
        language: str = "en",
        handwriting_template: Optional[str] = None,
        writing_style: str = "print",
        paper_template: str = "blank",
    ) -> Dataset:
        """
        Generate handwritten documents (alias for generate_handwriting).

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style (lined, grid, blank)

        Returns:
            HuggingFace Dataset with comprehensive handwriting schema
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

    def translate_documents(
        self,
        input_images: Optional[Union[List[Union[str, Path]], str, Path]] = None,
        input_dataset: Optional[Union[List[Dict[str, Any]], Dataset]] = None,
        target_languages: List[str] = ["hi"],
        yolo_model_path: Optional[str] = None,
        font_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        image_size: int = 1024,
        preserve_layout: bool = True,
    ) -> Dataset:
        """
        Translate documents to different languages while preserving layout.

        This method uses YOLO for layout detection, OCR for text extraction,
        translation APIs for text conversion, and smart font rendering for target languages.

        Args:
            input_images: Image files or directories to translate, or single image path
            input_dataset: Input dataset with images to translate
            target_languages: Target languages for translation (e.g., ['hi', 'zh', 'fr'])
            yolo_model_path: Path to YOLO layout detection model (default: '/SynthDoc/model-doclayout-yolo.pt')
            font_path: Path to directory with language-specific fonts
            confidence_threshold: Confidence threshold for layout detection (0.0-1.0)
            image_size: Input image size for YOLO model
            preserve_layout: Whether to preserve original document layout

        Returns:
            HuggingFace Dataset with translated documents

        Example:
            >>> # Translate a single image (using default YOLO model)
            >>> result = synthdoc.translate_documents(
            ...     input_images="document.png",
            ...     target_languages=["hi", "fr"]
            ... )
            
            >>> # Translate all images in a directory with custom model
            >>> result = synthdoc.translate_documents(
            ...     input_images="./documents/",
            ...     target_languages=["zh", "ar"],
            ...     yolo_model_path="path/to/custom_yolo_model.pt"
            ... )
            
            >>> # Translate from existing dataset
            >>> result = synthdoc.translate_documents(
            ...     input_dataset=existing_dataset,
            ...     target_languages=["es", "de"]
            ... )
        """
        self.logger.info(f"🌍 Starting document translation to languages: {target_languages}")

        # Handle single image path input
        if isinstance(input_images, (str, Path)):
            input_images = [input_images]

        # Set default paths if not provided
        if yolo_model_path is None:
            # Use the default YOLO model path
            yolo_model_path = "/SynthDoc/model-doclayout-yolo.pt"
            
        if font_path is None:
            # Use SynthDoc's built-in font path
            font_path = Path(__file__).parent / "fonts"
            if not font_path.exists():
                raise ValueError(f"Font path not found: {font_path}. Please provide valid font_path.")

        # Convert dataset input format
        dataset_list = None
        if input_dataset is not None:
            if isinstance(input_dataset, Dataset):
                dataset_list = [dict(sample) for sample in input_dataset]
            else:
                dataset_list = input_dataset

        # Create configuration
        config = DocumentTranslationConfig(
            input_images=input_images,
            input_dataset=dataset_list,
            target_languages=target_languages,
            yolo_model_path=yolo_model_path,
            font_path=str(font_path),
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            preserve_layout=preserve_layout,
        )

        try:
            # Use workflow to translate documents
            result = self.document_translator.process(config)
            self.logger.info(f"✅ Successfully translated {result.num_samples} documents")
            return result.dataset
        except Exception as e:
            self.logger.error(f"❌ Document translation failed: {e}")
            # Return fallback dataset with error information
            from datasets import Dataset
            return Dataset.from_dict({
                'error': [str(e)],
                'note': ['Document translation failed - check YOLO model, fonts, and translation dependencies'],
                'input_languages_requested': [target_languages],
                'fallback': [True]
            })
