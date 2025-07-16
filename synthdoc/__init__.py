"""
SynthDoc - A comprehensive library for generating synthetic documents.

This library provides multiple workflows for creating synthetic documents
for training and evaluating document understanding models.
"""

from typing import List

# Import the new workflow-based architecture
from .models import (
    RawDocumentGenerationConfig,
    VQAGenerationConfig,
    DocumentTranslationConfig,
    LayoutType,
    OutputFormat,
    WorkflowResult,
)

from .workflows import (
    BaseWorkflow,
    RawDocumentGenerator,
    VQAGenerator,
    DocumentTranslator,
)

# Import backward compatibility classes
from .core import SynthDoc, LanguageSupport
from .languages import Language

# Import model management utilities
from .models_manager import (
    download_model,
    ensure_model,
    is_model_downloaded,
    list_available_models,
    get_model_info,
    cleanup_models,
)

# Import utilities
from .utils import setup_logging, CostTracker
from .config import (
    load_env_config,
    get_api_key,
    get_llm_model,
    print_environment_status,
)

__version__ = "0.2.0"

# New workflow-based exports
__all__ = [
    # Core backward compatibility
    "SynthDoc",
    "LanguageSupport",
    "Language",
    # New workflow architecture
    "RawDocumentGenerationConfig",
    "VQAGenerationConfig",
    "DocumentTranslationConfig",
    "LayoutType",
    "OutputFormat",
    "WorkflowResult",
    # Workflows
    "BaseWorkflow",
    "RawDocumentGenerator",
    "VQAGenerator",
    "DocumentTranslator",
    # Utilities
    "setup_logging",
    "CostTracker",
    # Configuration
    "load_env_config",
    "get_api_key",
    "get_llm_model",
    "print_environment_status",
    # Model management
    "download_model",
    "ensure_model",
    "is_model_downloaded",
    "list_available_models",
    "get_model_info",
    "cleanup_models",
]


# Convenience functions for common workflows
def create_raw_documents(
    prompt: str = None,
    language: Language = Language.EN,
    num_pages: int = 1,
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN,
    include_graphs: bool = False,
    include_tables: bool = False,
    include_ai_images: bool = False,
    api_key: str = None,
    save_dir: str = "generated_docs",
) -> WorkflowResult:
    """
    Convenience function to create raw documents using the new workflow architecture.

    Args:
        prompt: Content prompt for document generation
        language: Target language
        num_pages: Number of pages to generate
        layout_type: Document layout type
        include_graphs: Whether to include graphs
        include_tables: Whether to include tables
        include_ai_images: Whether to include AI-generated images
        api_key: API key for LLM services (auto-loaded from .env if None)
        save_dir: Directory to save generated documents

    Returns:
        WorkflowResult with generated documents
    """
    # Auto-load API key from .env if not provided
    if not api_key:
        api_key = get_api_key("auto")
        if not api_key:
            raise ValueError(
                "No API key found. Set API key in .env file or pass api_key parameter"
            )

    # Create workflow
    generator = RawDocumentGenerator(groq_api_key=api_key, save_dir=save_dir)

    # Create configuration
    config = RawDocumentGenerationConfig(
        language=language,
        num_pages=num_pages,
        prompt=prompt,
        layout_type=layout_type,
        include_graphs=include_graphs,
        include_tables=include_tables,
        include_ai_images=include_ai_images,
    )

    return generator.process(config)


def create_vqa_dataset(
    documents: List[str],
    num_questions_per_doc: int = 5,
    include_hard_negatives: bool = True,
    question_types: List[str] = None,
    api_key: str = None,
    llm_model: str = None,
) -> WorkflowResult:
    """
    Convenience function to create VQA datasets.

    Args:
        documents: List of document paths or content
        num_questions_per_doc: Number of questions per document
        include_hard_negatives: Whether to include hard negative examples
        question_types: Types of questions to generate
        api_key: API key for LLM services (auto-loaded from .env if None)
        llm_model: LLM model to use (auto-detected from .env if None)

    Returns:
        WorkflowResult with VQA dataset
    """
    # Auto-load API key and model from .env if not provided
    if not api_key:
        api_key = get_api_key("auto")
        if not api_key:
            raise ValueError(
                "No API key found. Set API key in .env file or pass api_key parameter"
            )

    if not llm_model:
        llm_model = get_llm_model("auto")

    # Create workflow
    generator = VQAGenerator(api_key=api_key, llm_model=llm_model)

    # Create configuration
    config = VQAGenerationConfig(
        documents=documents,
        num_questions_per_doc=num_questions_per_doc,
        include_hard_negatives=include_hard_negatives,
        question_types=question_types or ["factual", "reasoning", "comparative"],
    )

    return generator.process(config)


def translate_documents(
    input_images: List[str],
    target_languages: List[str] = ["hi"],
    yolo_model_path: str = None,
    font_path: str = "./synthdoc/fonts/",
    save_dir: str = "translated_docs",
) -> WorkflowResult:
    """
    Convenience function to translate documents.

    Args:
        input_images: List of image paths to translate
        target_languages: Target languages for translation
        yolo_model_path: Path to YOLO model for layout detection (auto-downloaded if None)
        font_path: Path to fonts directory
        save_dir: Directory to save translated documents

    Returns:
        WorkflowResult with translated documents
    """
    # Auto-download YOLO model if not provided
    if yolo_model_path is None:
        yolo_model_path = str(ensure_model("doclayout-yolo"))

    # Create workflow
    translator = DocumentTranslator(save_dir=save_dir)

    # Create configuration
    config = DocumentTranslationConfig(
        input_images=input_images,
        target_languages=target_languages,
        yolo_model_path=yolo_model_path,
        font_path=font_path,
    )

    return translator.process(config)


# Quick examples for new users
def quick_example():
    """Run a quick example to demonstrate SynthDoc capabilities with .env loading."""
    print("🚀 SynthDoc Quick Example with .env Configuration")
    print("=" * 60)

    # Show environment status first
    print("\n🔧 Environment Configuration Status:")
    try:
        print_environment_status()
    except Exception as e:
        print(f"⚠️  Could not load environment status: {e}")

    print("\n" + "=" * 60)

    try:
        # Example 1: Backward compatibility with auto .env loading
        print("\n📚 Example 1: SynthDoc class with automatic .env loading")
        synth = SynthDoc()  # Automatically loads from .env
        print(f"✅ SynthDoc initialized with model: {synth.llm_model}")
        print(f"📁 Output directory: {synth.output_dir}")

        # Example 2: New workflow architecture
        print("\n📄 Example 2: Workflow functions with .env auto-loading")

        # Try LLM-powered features if API key available
        print("\n🤖 Testing LLM-powered features...")
        api_key = get_api_key("auto")
        if api_key:
            print("🔑 API key found - testing document generation...")
            try:
                docs = synth.generate_raw_docs(
                    language="en",
                    num_pages=1,
                    prompt="Write a brief AI research summary",
                )
                print(f"✅ Generated {len(docs)} document(s) using LLM")
            except Exception as e:
                print(f"⚠️  LLM generation failed: {e}")
        else:
            print("⚠️  No API key found - skipping LLM features")
            print("💡 Set API keys in .env file for full functionality")

        print("\n🎉 Quick example completed successfully!")

    except Exception as e:
        print(f"⚠️ Example error: {e}")
        print("💡 This is normal if you don't have all dependencies installed.")

    print("\n💡 Setup Instructions:")
    print("1. Copy env.template to .env")
    print("2. Add your API keys to .env file")
    print(
        "3. Run: from synthdoc import print_environment_status; print_environment_status()"
    )


def check_environment():
    """Check and display current environment configuration."""
    print("🔧 SynthDoc Environment Check")
    print("=" * 40)
    print_environment_status()

    print("\n💡 Next Steps:")
    config = load_env_config()
    if not any([config.openai_api_key, config.anthropic_api_key, config.groq_api_key]):
        print("1. Copy env.template to .env")
        print("2. Add at least one API key")
        print(
            "3. Test with: python -c 'from synthdoc import quick_example; quick_example()'"
        )
    else:
        print("✅ Environment looks good!")
        print("🚀 Try: python -c 'from synthdoc import quick_example; quick_example()'")


# Add to exports
__all__.extend(["check_environment", "translate_documents"])


if __name__ == "__main__":
    quick_example()
