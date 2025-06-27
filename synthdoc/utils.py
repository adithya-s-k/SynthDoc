"""
Utility functions for SynthDoc.

This module provides common utilities like logging setup, file operations,
and helper functions used across the library.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import litellm


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for SynthDoc.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("synthdoc")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def validate_language_code(code: str) -> bool:
    """
    Validate if a language code is supported.

    Args:
        code: Language code to validate

    Returns:
        True if supported, False otherwise
    """
    from .languages import LanguageSupport

    return code in LanguageSupport.get_supported_languages()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_extension(path: Union[str, Path]) -> str:
    """
    Get file extension from path.

    Args:
        path: File path

    Returns:
        File extension (without dot)
    """
    return Path(path).suffix.lstrip(".")


def validate_file_exists(path: Union[str, Path]) -> bool:
    """
    Check if file exists.

    Args:
        path: File path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # TODO: Implement configuration loading
    default_config = {
        "output_format": "huggingface",
        "default_language": "en",
        "max_pages_per_document": 10,
        "default_augmentations": ["rotation", "scaling"],
        "quality_settings": {"image_dpi": 300, "image_format": "PNG"},
    }

    return default_config


def format_dataset_for_huggingface(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format data for HuggingFace datasets.

    Args:
        data: List of document data

    Returns:
        HuggingFace compatible dataset format
    """
    # TODO: Implement HuggingFace formatting
    formatted = {"images": [], "annotations": [], "metadata": []}

    for item in data:
        formatted["images"].append(item.get("image"))
        formatted["annotations"].append(item.get("annotations", {}))
        formatted["metadata"].append(item.get("metadata", {}))

    return formatted


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(self.__class__.__name__)

    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        self.logger.info(
            f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"
        )

    def finish(self):
        """Mark as complete."""
        self.logger.info(f"{self.description}: Complete ({self.total}/{self.total})")


def merge_configs(
    base_config: Dict[str, Any], user_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge user configuration with base configuration.

    Args:
        base_config: Base configuration
        user_config: User overrides

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in user_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.api_calls_count = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def get_dynamic_pricing(self, model: str) -> Dict[str, float]:
        """Get dynamic pricing from LiteLLM."""
        try:
            # LiteLLM has built-in pricing
            return litellm.get_model_cost_map(model)
        except:
            # Fallback to known pricing
            fallback_pricing = {
                "groq/llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
                "groq/llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
                "groq/mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024}
            }
            return fallback_pricing.get(model, {"input": 0.0, "output": 0.0})
    
    def track_usage(self, response, model: str = "groq/llama3-8b-8192") -> float:
        """Track API usage and costs."""
        self.api_calls_count += 1
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
            
            # Use dynamic pricing
            pricing = self.get_dynamic_pricing(model)
            input_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
            output_cost = (completion_tokens / 1000) * pricing.get("output", 0)
            cost = input_cost + output_cost
            
            self.total_cost += cost
            return cost
        return 0.0
    
     
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost": self.total_cost,
            "api_calls": self.api_calls_count,
            "tokens_used": self.token_usage['total_tokens'],
            "token_breakdown": self.token_usage
        }