"""
SynthDoc - A comprehensive library for generating synthetic documents.

This library provides multiple workflows for creating synthetic documents
for training and evaluating document understanding models.
"""

from typing import List

# Import the new workflow-based architecture
from .models import (
    RawDocumentGenerationConfig,
    LayoutAugmentationConfig,
    PDFAugmentationConfig,
    VQAGenerationConfig,
    HandwritingGenerationConfig,
    AugmentationType,
    LayoutType,
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

# Import backward compatibility classes
from .core import SynthDoc, LanguageSupport
from .languages import Language

# Import utilities
from .utils import setup_logging, CostTracker
from .config import load_env_config, get_api_key, get_llm_model, print_environment_status

__version__ = "0.2.0"

# New workflow-based exports
__all__ = [
    # Core backward compatibility
    "SynthDoc",
    "LanguageSupport",
    "Language",
    
    # New workflow architecture
    "RawDocumentGenerationConfig",
    "LayoutAugmentationConfig",
    "PDFAugmentationConfig", 
    "VQAGenerationConfig",
    "HandwritingGenerationConfig",
    "AugmentationType",
    "LayoutType",
    "OutputFormat",
    "WorkflowResult",
    
    # Workflows
    "BaseWorkflow",
    "RawDocumentGenerator",
    "LayoutAugmenter",
    "PDFAugmenter",
    "VQAGenerator",
    "HandwritingGenerator",
    
    # Utilities
    "setup_logging",
    "CostTracker",
    
    # Configuration
    "load_env_config",
    "get_api_key", 
    "get_llm_model",
    "print_environment_status",
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
    save_dir: str = "generated_docs"
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
            raise ValueError("No API key found. Set API key in .env file or pass api_key parameter")
    
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
        include_ai_images=include_ai_images
    )
    
    return generator.process(config)


def create_vqa_dataset(
    documents: List[str],
    num_questions_per_doc: int = 5,
    include_hard_negatives: bool = True,
    question_types: List[str] = None,
    api_key: str = None,
    llm_model: str = None
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
            raise ValueError("No API key found. Set API key in .env file or pass api_key parameter")
    
    if not llm_model:
        llm_model = get_llm_model("auto")
    
    # Create workflow
    generator = VQAGenerator(api_key=api_key, llm_model=llm_model)
    
    # Create configuration
    config = VQAGenerationConfig(
        documents=documents,
        num_questions_per_doc=num_questions_per_doc,
        include_hard_negatives=include_hard_negatives,
        question_types=question_types or ["factual", "reasoning", "comparative"]
    )
    
    return generator.process(config)


def create_handwriting_samples(
    text_content: str = None,
    language: Language = Language.EN,
    handwriting_style: str = "default",
    paper_template: str = "lined",
    num_samples: int = 1,
    save_dir: str = "handwriting_output"
) -> WorkflowResult:
    """
    Convenience function to create handwriting samples.
    
    Args:
        text_content: Text content to render as handwriting
        language: Target language
        handwriting_style: Style of handwriting ("default", "cursive", "print")
        paper_template: Paper background style ("lined", "grid", "blank")
        num_samples: Number of samples to generate
        save_dir: Directory to save handwriting samples
        
    Returns:
        WorkflowResult with handwriting samples
    """
    # Create workflow
    generator = HandwritingGenerator(save_dir=save_dir)
    
    # Create configuration
    config = HandwritingGenerationConfig(
        text_content=text_content,
        language=language,
        handwriting_style=handwriting_style,
        paper_template=paper_template,
        num_samples=num_samples
    )
    
    return generator.process(config)


def augment_layouts(
    documents: List[str],
    languages: List[Language] = None,
    fonts: List[str] = None,
    augmentations: List[AugmentationType] = None,
    save_dir: str = "layout_output"
) -> WorkflowResult:
    """
    Convenience function to create layout variations.
    
    Args:
        documents: List of document paths or content
        languages: Target languages for content
        fonts: Font families to apply
        augmentations: Visual augmentations to apply
        save_dir: Directory to save layout variations
        
    Returns:
        WorkflowResult with layout variations
    """
    # Create workflow
    augmenter = LayoutAugmenter(save_dir=save_dir)
    
    # Create configuration
    config = LayoutAugmentationConfig(
        documents=documents,
        languages=languages or [Language.EN],
        fonts=fonts,
        augmentations=augmentations
    )
    
    return augmenter.process(config)


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
        
        # Create handwriting samples (no API key needed)
        print("🖋️  Creating handwriting samples...")
        handwriting_result = create_handwriting_samples(
            text_content="Hello, this is a handwriting sample!",
            handwriting_style="cursive",
            num_samples=2
        )
        print(f"✅ Generated {handwriting_result.num_samples} handwriting samples")
        
        # Create layout variations (no API key needed)
        print("📄 Creating layout variations...")
        layout_result = augment_layouts(
            documents=["Sample document content for layout testing"],
            fonts=["Arial", "Times New Roman"]
        )
        print(f"✅ Generated {layout_result.num_samples} layout variations")
        
        # Try LLM-powered features if API key available
        print("\n🤖 Testing LLM-powered features...")
        api_key = get_api_key("auto")
        if api_key:
            print("🔑 API key found - testing document generation...")
            try:
                docs = synth.generate_raw_docs(
                    language="en", 
                    num_pages=1, 
                    prompt="Write a brief AI research summary"
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
    print("3. Run: from synthdoc import print_environment_status; print_environment_status()")


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
        print("3. Test with: python -c 'from synthdoc import quick_example; quick_example()'")
    else:
        print("✅ Environment looks good!")
        print("🚀 Try: python -c 'from synthdoc import quick_example; quick_example()'")


# Add to exports
__all__.extend(["check_environment"])


if __name__ == "__main__":
    quick_example()
