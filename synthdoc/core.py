"""
Core SynthDoc class - Main entry point for the library.

This module provides the main SynthDoc class that orchestrates all document generation workflows.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .models_manager import ensure_model
from .languages import LanguageSupport, Language
from .workflows import RawDocumentGenerator, VQAGenerator, DocumentTranslator
from .models import (
    RawDocumentGenerationConfig,
    VQAGenerationConfig,
    DocumentTranslationConfig,
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
            output_dir: Directory for output files (will prompt user if not provided)
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
        self.output_dir = self._setup_output_directory(output_dir)

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

        # Initialize workflows with unified output directory
        self.language_support = LanguageSupport()
        self.raw_doc_generator = RawDocumentGenerator(api_key, str(self.output_dir))
        # Backwards-compatibility alias (tests & older code reference self.doc_generator)
        self.doc_generator = self.raw_doc_generator
        self.vqa_generator = VQAGenerator(api_key, llm_model, str(self.output_dir))
        self.document_translator = DocumentTranslator(str(self.output_dir))

        # Log initialization status
        if api_key:
            self.logger.info(
                f"SynthDoc initialized successfully with model: {llm_model}"
            )
        else:
            self.logger.warning(
                "SynthDoc initialized without API key - limited functionality"
            )
            self.logger.info("Set API keys in .env file for full LLM features")

        # Check model status
        self._check_model_status()

    def _setup_output_directory(self, output_dir: Optional[Union[str, Path]]) -> Path:
        """Set up output directory with user prompt if not provided."""
        if output_dir:
            # User provided output directory
            output_path = Path(output_dir)
        else:
            # Ask user for output directory
            print("\n📁 SynthDoc Output Directory Setup")
            print("=" * 40)

            # Get default from config
            default_dir = self.config.default_output_dir

            # Prompt user for output directory
            user_input = input(
                f"Enter output directory (default: {default_dir}): "
            ).strip()

            if user_input:
                output_path = Path(user_input)
            else:
                output_path = Path(default_dir)

        # Create the main output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create simplified structure: images folder and metadata.jsonl
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        # Create metadata.jsonl file if it doesn't exist
        metadata_file = output_path / "metadata.jsonl"
        if not metadata_file.exists():
            metadata_file.touch()

        print(f"✅ Output directory created: {output_path}")
        print(f"📂 Images will be saved to: {images_dir}")
        print(f"📄 Metadata will be saved to: {metadata_file}")

        return output_path

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
                "en": Language.EN,
                "hi": Language.HI,
                "zh": Language.ZH,
                "es": Language.ES,
                "fr": Language.FR,
                "ar": Language.AR,
            }
            lang_enum = lang_mapping.get(language.lower(), Language.EN)

        # Create configuration
        config = RawDocumentGenerationConfig(
            language=lang_enum,
            num_pages=num_pages,
            prompt=prompt,
        )

        # Use workflow to generate documents
        result = self.raw_doc_generator.process(config)
        return result.dataset

    # Removed augment_layout method - LayoutAugmenter workflow not implemented

    # Removed augment_pdfs method - PDFAugmenter workflow not implemented

    def generate_vqa(
        self,
        source_documents: Optional[Union[List[Dict[str, Any]], Dataset]] = None,
        single_image: Optional[Union[str, Path]] = None,
        image_folder: Optional[Union[str, Path]] = None,
        pdf_file: Optional[Union[str, Path]] = None,
        pdf_folder: Optional[Union[str, Path]] = None,
        question_types: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None,
        hard_negative_ratio: float = 0.3,
        num_questions_per_doc: int = 3,
        processing_mode: str = "VLM",
        llm_model: str = "gemini-2.5-flash",
    ) -> Dataset:
        """
        Generate VQA datasets with hard negatives using flexible input types.

        Args:
            source_documents: Documents to generate questions about (list or Dataset)
            single_image: Single image file for VQA generation
            image_folder: Folder containing images for VQA generation
            pdf_file: Single PDF file for VQA generation
            pdf_folder: Folder containing PDFs for VQA generation
            question_types: Types of questions to generate (factual, reasoning, counting, etc.)
            difficulty_levels: Question complexity levels
            hard_negative_ratio: Ratio of hard negative examples
            num_questions_per_doc: Number of questions per document
            processing_mode: 'VLM' for vision+text or 'LLM' for text-only
            llm_model: LLM model to use (default: gemini-2.5-flash)

        Returns:
            HuggingFace Dataset with comprehensive VQA schema
        """
        self.logger.info("Generating VQA dataset")

        # Extract document paths from various input formats
        doc_paths = []

        # Handle source_documents input (for compatibility)
        if source_documents is not None:
            if isinstance(source_documents, Dataset):
                # Extract paths from HuggingFace Dataset
                for i, sample in enumerate(source_documents):
                    if "image" in sample and hasattr(sample["image"], "save"):
                        temp_path = self.output_dir / "temp" / f"vqa_doc_{i}.png"
                        temp_path.parent.mkdir(exist_ok=True)
                        sample["image"].save(temp_path)
                        doc_paths.append(str(temp_path))
                    elif "image_path" in sample:
                        doc_paths.append(sample["image_path"])
            else:
                # Handle list of documents
                for i, doc in enumerate(source_documents):
                    if "image_path" in doc:
                        doc_paths.append(doc["image_path"])
                    elif "image" in doc and hasattr(doc["image"], "save"):
                        temp_path = self.output_dir / "temp" / f"vqa_doc_{i}.png"
                        temp_path.parent.mkdir(exist_ok=True)
                        doc["image"].save(temp_path)
                        doc_paths.append(str(temp_path))

        # Create configuration
        config = VQAGenerationConfig(
            documents=doc_paths,
            single_image=single_image,
            image_folder=image_folder,
            pdf_file=pdf_file,
            pdf_folder=pdf_folder,
            question_types=question_types
            or [
                "factual",
                "reasoning",
                "comparative",
                "counting",
                "spatial",
                "descriptive",
            ],
            difficulty_levels=difficulty_levels,
            include_hard_negatives=hard_negative_ratio > 0,
            num_questions_per_doc=num_questions_per_doc,
            processing_mode=processing_mode,
            llm_model=llm_model,
        )

        # Use workflow to generate VQA
        result = self.vqa_generator.process(config)
        return result.dataset

    # Removed generate_handwriting method - HandwritingGenerator workflow not implemented

    # Removed create_handwriting method - HandwritingGenerator workflow not implemented

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
            yolo_model_path: Path to YOLO layout detection model (default: './model-doclayout-yolo.pt')
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
        self.logger.info(
            f"🌍 Starting document translation to languages: {target_languages}"
        )

        # Handle single image path input
        if isinstance(input_images, (str, Path)):
            input_images = [input_images]

        # Set default paths if not provided
        if yolo_model_path is None:
            # Use the model manager to ensure the YOLO model is downloaded
            self.logger.info("🔄 Ensuring YOLO model is available...")
            yolo_model_path = str(ensure_model("doclayout-yolo"))

        if font_path is None:
            # Use SynthDoc's built-in font path
            font_path = Path(__file__).parent / "fonts"
            if not font_path.exists():
                raise ValueError(
                    f"Font path not found: {font_path}. Please provide valid font_path."
                )

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
            self.logger.info(
                f"✅ Successfully translated {result.num_samples} documents"
            )
            return result.dataset
        except Exception as e:
            self.logger.error(f"❌ Document translation failed: {e}")
            # Return fallback dataset with error information
            from datasets import Dataset

            return Dataset.from_dict(
                {
                    "error": [str(e)],
                    "note": [
                        "Document translation failed - check YOLO model, fonts, and translation dependencies"
                    ],
                    "input_languages_requested": [target_languages],
                    "fallback": [True],
                }
            )

    def _check_model_status(self):
        """Check status of required models and provide helpful information."""
        try:
            from .models_manager import is_model_downloaded, list_available_models

            models_status = {}
            for model_name in list_available_models():
                models_status[model_name] = is_model_downloaded(model_name)

            missing_models = [
                name for name, downloaded in models_status.items() if not downloaded
            ]

            if missing_models:
                self.logger.info(
                    f"📦 Models available for auto-download: {', '.join(missing_models)}"
                )
                self.logger.info(
                    "💡 Models will be downloaded automatically when needed"
                )
            else:
                self.logger.info("✅ All models are ready")

        except Exception as e:
            # Don't fail initialization if model checking fails
            self.logger.debug(f"Could not check model status: {e}")
