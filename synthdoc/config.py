"""
Configuration settings for SynthDoc.

This module contains default configuration values that can be overridden.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class DocumentConfig:
    """Document generation configuration."""

    default_language: str = "en"
    max_pages_per_document: int = 10
    default_prompt: str = "Generate diverse document content"

    # Image settings
    image_dpi: int = 300
    image_format: str = "PNG"
    image_quality: int = 95

    # Output settings
    output_format: str = "huggingface"
    save_intermediate: bool = True


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""

    default_augmentations: List[str] = field(
        default_factory=lambda: ["rotation", "scaling"]
    )
    intensity_range: tuple = (0.3, 0.7)

    # Specific augmentation settings
    rotation_range: tuple = (-15, 15)  # degrees
    scale_range: tuple = (0.8, 1.2)
    noise_range: tuple = (0.0, 0.1)
    brightness_range: tuple = (0.8, 1.2)


@dataclass
class VQAConfig:
    """VQA generation configuration."""

    default_question_types: List[str] = field(
        default_factory=lambda: ["factual", "reasoning"]
    )
    default_difficulty_levels: List[str] = field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    hard_negative_ratio: float = 0.3
    questions_per_document: int = 5


@dataclass
class DatasetManagementConfig:
    """Dataset management configuration."""

    # Default paths
    default_dataset_root: str = "./datasets"

    # Dataset creation defaults
    default_metadata_format: str = "jsonl"  # jsonl, csv, parquet
    default_splits: List[str] = field(
        default_factory=lambda: ["train", "test", "validation"]
    )
    copy_images: bool = True
    auto_flush_threshold: int = 100

    # Upload defaults
    default_private: bool = False
    auto_create_card: bool = True


@dataclass
class SynthDocConfig:
    """Main configuration class."""

    document: DocumentConfig = field(default_factory=DocumentConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    vqa: VQAConfig = field(default_factory=VQAConfig)
    dataset: DatasetManagementConfig = field(default_factory=DatasetManagementConfig)

    # Logging
    log_level: str = "INFO"

    # Performance
    max_workers: int = 4
    batch_size: int = 32

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Azure OpenAI
    azure_api_key: Optional[str] = None
    azure_api_base: Optional[str] = None
    azure_api_version: Optional[str] = None
    
    # HuggingFace
    huggingface_token: Optional[str] = None
    
    # Default settings
    default_llm_model: str = "gpt-4o-mini"
    default_output_dir: str = "./synthdoc_output"
    debug_mode: bool = False
    
    # Advanced settings
    max_tokens: int = 2048
    llm_temperature: float = 0.7
    request_timeout: int = 60
    enable_cost_tracking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "document": {
                "default_language": self.document.default_language,
                "max_pages_per_document": self.document.max_pages_per_document,
                "image_dpi": self.document.image_dpi,
                "image_format": self.document.image_format,
                "output_format": self.document.output_format,
            },
            "augmentation": {
                "default_augmentations": self.augmentation.default_augmentations,
                "intensity_range": self.augmentation.intensity_range,
                "rotation_range": self.augmentation.rotation_range,
                "scale_range": self.augmentation.scale_range,
            },
            "vqa": {
                "default_question_types": self.vqa.default_question_types,
                "hard_negative_ratio": self.vqa.hard_negative_ratio,
                "questions_per_document": self.vqa.questions_per_document,
            },
            "dataset": {
                "default_dataset_root": self.dataset.default_dataset_root,
                "default_metadata_format": self.dataset.default_metadata_format,
                "default_splits": self.dataset.default_splits,
                "copy_images": self.dataset.copy_images,
                "auto_flush_threshold": self.dataset.auto_flush_threshold,
            },
            "log_level": self.log_level,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "groq_api_key": self.groq_api_key,
            "cohere_api_key": self.cohere_api_key,
            "azure_api_key": self.azure_api_key,
            "azure_api_base": self.azure_api_base,
            "azure_api_version": self.azure_api_version,
            "huggingface_token": self.huggingface_token,
            "default_llm_model": self.default_llm_model,
            "default_output_dir": self.default_output_dir,
            "debug_mode": self.debug_mode,
            "max_tokens": self.max_tokens,
            "llm_temperature": self.llm_temperature,
            "request_timeout": self.request_timeout,
            "enable_cost_tracking": self.enable_cost_tracking,
        }


# Default configuration instance
DEFAULT_CONFIG = SynthDocConfig()


def load_env_config(env_file: Optional[Union[str, Path]] = None) -> SynthDocConfig:
    """
    Load configuration from environment variables and .env file.
    
    Args:
        env_file: Path to .env file. If None, searches for .env in current and parent directories.
        
    Returns:
        SynthDocConfig: Configuration object with loaded values.
    """
    if DOTENV_AVAILABLE:
        # Search for .env file if not specified
        if env_file is None:
            # Check current directory and parent directories
            search_paths = [
                Path.cwd() / ".env",
                Path.cwd().parent / ".env",
                Path(__file__).parent.parent / ".env"  # SynthDoc root
            ]
            
            for path in search_paths:
                if path.exists():
                    env_file = path
                    break
        
        # Load .env file if found
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
            print(f"✅ Loaded environment from: {env_file}")
        elif env_file:
            print(f"⚠️  .env file not found: {env_file}")
    else:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    
    # Load configuration from environment variables
    config = SynthDocConfig(
        # API Keys
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        
        # Azure OpenAI
        azure_api_key=os.getenv("AZURE_API_KEY"),
        azure_api_base=os.getenv("AZURE_API_BASE"),
        azure_api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
        
        # HuggingFace
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
        
        # Default settings
        default_llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        default_output_dir=os.getenv("DEFAULT_OUTPUT_DIR", "./synthdoc_output"),
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        
        # Advanced settings
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        enable_cost_tracking=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
    )
    
    return config


def get_api_key(provider: str = "auto") -> Optional[str]:
    """
    Get API key for specified provider with automatic fallback.
    
    Args:
        provider: API provider ("openai", "anthropic", "groq", "cohere", "auto")
        
    Returns:
        str: API key if found, None otherwise
    """
    config = load_env_config()
    
    if provider == "auto":
        # Try providers in order of preference
        providers = [
            ("groq", config.groq_api_key),
            ("openai", config.openai_api_key),
            ("anthropic", config.anthropic_api_key),
            ("cohere", config.cohere_api_key)
        ]
        
        for provider_name, api_key in providers:
            if api_key:
                print(f"🔑 Using {provider_name.upper()} API key")
                return api_key
        
        print("⚠️  No API keys found. Please set API keys in .env file")
        return None
    
    else:
        # Get specific provider key
        api_key = getattr(config, f"{provider}_api_key", None)
        if api_key:
            print(f"🔑 Using {provider.upper()} API key")
        else:
            print(f"⚠️  {provider.upper()} API key not found")
        return api_key


def get_llm_model(provider: str = "auto") -> str:
    """
    Get appropriate LLM model for provider.
    
    Args:
        provider: API provider
        
    Returns:
        str: Model name
    """
    config = load_env_config()
    
    if provider == "auto":
        # Determine best model based on available API key
        if config.groq_api_key:
            return "groq/llama-3.1-8b-instant"
        elif config.openai_api_key:
            return "gpt-4o-mini"
        elif config.anthropic_api_key:
            return "claude-3-5-sonnet-20241022"
        else:
            return config.default_llm_model
    
    # Provider-specific models
    model_mapping = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "groq": "groq/llama-3.1-8b-instant",
        "cohere": "command-r-plus"
    }
    
    return model_mapping.get(provider, config.default_llm_model)


def setup_environment() -> Dict[str, Any]:
    """
    Set up complete SynthDoc environment configuration.
    
    Returns:
        dict: Environment setup summary
    """
    config = load_env_config()
    
    # Check which API keys are available
    available_providers = []
    if config.openai_api_key:
        available_providers.append("OpenAI")
    if config.anthropic_api_key:
        available_providers.append("Anthropic")
    if config.groq_api_key:
        available_providers.append("Groq")
    if config.cohere_api_key:
        available_providers.append("Cohere")
    
    # Check Azure configuration
    azure_configured = bool(config.azure_api_key and config.azure_api_base)
    
    # Setup summary
    setup_info = {
        "config": config,
        "providers_available": available_providers,
        "azure_configured": azure_configured,
        "dotenv_available": DOTENV_AVAILABLE,
        "default_model": config.default_llm_model,
        "output_directory": config.default_output_dir,
        "cost_tracking_enabled": config.enable_cost_tracking
    }
    
    return setup_info


def print_environment_status():
    """Print detailed environment configuration status."""
    print("🔧 SynthDoc Environment Configuration")
    print("=" * 50)
    
    setup_info = setup_environment()
    config = setup_info["config"]
    
    # API Provider Status
    print("\n🔑 API Provider Status:")
    providers = ["openai", "anthropic", "groq", "cohere"]
    for provider in providers:
        api_key = getattr(config, f"{provider}_api_key")
        status = "✅ CONFIGURED" if api_key else "❌ NOT SET"
        print(f"  {provider.capitalize()}: {status}")
    
    # Azure Status
    print(f"\n☁️  Azure OpenAI: {'✅ CONFIGURED' if setup_info['azure_configured'] else '❌ NOT SET'}")
    
    # HuggingFace Status
    print(f"🤗 HuggingFace: {'✅ CONFIGURED' if config.huggingface_token else '❌ NOT SET'}")
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    print(f"  Default Model: {config.default_llm_model}")
    print(f"  Output Directory: {config.default_output_dir}")
    print(f"  Debug Mode: {config.debug_mode}")
    print(f"  Cost Tracking: {config.enable_cost_tracking}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Temperature: {config.llm_temperature}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if not setup_info["providers_available"]:
        print("  ⚠️  Set at least one API key for LLM features")
        print("  📄 Copy env.template to .env and add your keys")
    
    if not config.huggingface_token:
        print("  ⚠️  Set HUGGINGFACE_TOKEN for dataset uploads")
    
    if not DOTENV_AVAILABLE:
        print("  ⚠️  Install python-dotenv: pip install python-dotenv")


# Legacy compatibility - maintain existing config functionality
class Config:
    """Legacy configuration class for backward compatibility."""
    
    @staticmethod
    def get_default_fonts():
        """Get default fonts for different languages."""
        return {
            "en": ["Arial", "Times New Roman", "Calibri"],
            "hi": ["Noto Sans Devanagari", "Arial Unicode MS"],
            "zh": ["SimSun", "Microsoft YaHei", "Arial Unicode MS"],
            "ar": ["Arial Unicode MS", "Tahoma"],
            "ja": ["MS Gothic", "Arial Unicode MS"]
        }
    
    @staticmethod
    def get_supported_languages():
        """Get list of supported languages."""
        return ["en", "hi", "zh", "ar", "ja", "ko", "th", "vi", "bn", "ta", "te", "ml", "kn", "gu", "or", "pa", "mr"]
    
    @staticmethod
    def get_augmentation_types():
        """Get available augmentation types."""
        return ["rotation", "scaling", "noise", "blur", "color_shift", "cropping"]


# Global configuration instance
_global_config = None

def get_global_config() -> SynthDocConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_env_config()
    return _global_config
