"""
Model management utilities for SynthDoc.

This module handles downloading and caching of pre-trained models from HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


logger = logging.getLogger(__name__)


# Model configurations
MODELS_CONFIG = {
    "doclayout-yolo": {
        "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
        "filename": "models/Layout/YOLO/doclayout_yolo_ft.pt",
        "local_filename": "doclayout_yolo_ft.pt",
        "description": "Document layout analysis YOLO model",
        "size_mb": 52,  # Approximate size
        "url": "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt",
    }
}


def get_models_dir() -> Path:
    """
    Get the directory where models are stored.

    Returns:
        Path to the downloaded_models directory
    """
    # Get the synthdoc package directory
    synthdoc_dir = Path(__file__).parent
    models_dir = synthdoc_dir / "downloaded_models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_model_path(model_name: str) -> Path:
    """
    Get the local path for a specific model.

    Args:
        model_name: Name of the model (key in MODELS_CONFIG)

    Returns:
        Path to the model file

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODELS_CONFIG:
        available_models = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    config = MODELS_CONFIG[model_name]
    models_dir = get_models_dir()
    return models_dir / config["local_filename"]


def is_model_downloaded(model_name: str) -> bool:
    """
    Check if a model is already downloaded locally.

    Args:
        model_name: Name of the model

    Returns:
        True if model exists locally, False otherwise
    """
    try:
        model_path = get_model_path(model_name)
        return model_path.exists() and model_path.stat().st_size > 0
    except ValueError:
        return False


def download_model_hf_hub(model_name: str, force_download: bool = False) -> Path:
    """
    Download a model using HuggingFace Hub.

    Args:
        model_name: Name of the model to download
        force_download: Whether to re-download if model already exists

    Returns:
        Path to the downloaded model

    Raises:
        ImportError: If huggingface_hub is not available
        ValueError: If model_name is not supported
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for model downloading. "
            "Install it with: pip install huggingface_hub"
        )

    if model_name not in MODELS_CONFIG:
        available_models = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    config = MODELS_CONFIG[model_name]
    local_path = get_model_path(model_name)

    # Check if model already exists
    if not force_download and is_model_downloaded(model_name):
        logger.info(f"Model '{model_name}' already exists at {local_path}")
        return local_path

    logger.info(f"📥 Downloading {config['description']} (~{config['size_mb']}MB)...")
    logger.info(f"Source: {config['repo_id']}/{config['filename']}")

    try:
        # Download to models directory
        models_dir = get_models_dir()

        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            cache_dir=str(models_dir / "cache"),
            force_download=force_download,
            local_dir=None,  # Use cache dir instead of local_dir
        )

        # Copy to our desired location with the correct name
        import shutil

        shutil.copy2(downloaded_path, local_path)

        logger.info(f"✅ Model downloaded successfully to {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"❌ Failed to download model '{model_name}': {e}")
        raise


def download_model_direct(model_name: str, force_download: bool = False) -> Path:
    """
    Download a model using direct HTTP requests (fallback method).

    Args:
        model_name: Name of the model to download
        force_download: Whether to re-download if model already exists

    Returns:
        Path to the downloaded model

    Raises:
        ImportError: If requests is not available
        ValueError: If model_name is not supported
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "requests is required for direct model downloading. "
            "Install it with: pip install requests"
        )

    if model_name not in MODELS_CONFIG:
        available_models = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    config = MODELS_CONFIG[model_name]
    local_path = get_model_path(model_name)

    # Check if model already exists
    if not force_download and is_model_downloaded(model_name):
        logger.info(f"Model '{model_name}' already exists at {local_path}")
        return local_path

    logger.info(f"📥 Downloading {config['description']} (~{config['size_mb']}MB)...")
    logger.info(f"Source: {config['url']}")

    try:
        # Create models directory
        temp_path = local_path.with_suffix(".tmp")

        # Download with progress
        response = requests.get(config["url"], stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\r📥 Downloading... {progress:.1f}%", end="", flush=True
                        )

        print()  # New line after progress

        # Move from temp to final location
        temp_path.rename(local_path)

        logger.info(f"✅ Model downloaded successfully to {local_path}")
        return local_path

    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"❌ Failed to download model '{model_name}': {e}")
        raise


def download_model(
    model_name: str, force_download: bool = False, use_hf_hub: bool = True
) -> Path:
    """
    Download a model with automatic fallback between HuggingFace Hub and direct download.

    Args:
        model_name: Name of the model to download
        force_download: Whether to re-download if model already exists
        use_hf_hub: Whether to try HuggingFace Hub first (fallback to direct if fails)

    Returns:
        Path to the downloaded model

    Raises:
        Exception: If all download methods fail
    """
    if model_name not in MODELS_CONFIG:
        available_models = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    # Check if model already exists
    if not force_download and is_model_downloaded(model_name):
        return get_model_path(model_name)

    errors = []

    # Try HuggingFace Hub first (if requested and available)
    if use_hf_hub and HF_HUB_AVAILABLE:
        try:
            return download_model_hf_hub(model_name, force_download)
        except Exception as e:
            logger.warning(f"HuggingFace Hub download failed: {e}")
            errors.append(f"HF Hub: {e}")

    # Fallback to direct download
    if REQUESTS_AVAILABLE:
        try:
            return download_model_direct(model_name, force_download)
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            errors.append(f"Direct: {e}")

    # If all methods failed
    error_msg = "All download methods failed:\n" + "\n".join(
        f"- {error}" for error in errors
    )
    raise Exception(error_msg)


def ensure_model(model_name: str, auto_download: bool = True) -> Path:
    """
    Ensure a model is available locally, downloading if necessary.

    Args:
        model_name: Name of the model
        auto_download: Whether to automatically download if model is missing

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model is not available and auto_download is False
        Exception: If download fails
    """
    if is_model_downloaded(model_name):
        return get_model_path(model_name)

    if not auto_download:
        raise FileNotFoundError(
            f"Model '{model_name}' not found locally. "
            f"Set auto_download=True or run download_model('{model_name}')"
        )

    logger.info(f"Model '{model_name}' not found locally. Downloading...")
    return download_model(model_name)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models and their configurations.

    Returns:
        Dictionary mapping model names to their configurations
    """
    return MODELS_CONFIG.copy()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information including download status

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODELS_CONFIG:
        available_models = ", ".join(MODELS_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    config = MODELS_CONFIG[model_name].copy()
    config["local_path"] = str(get_model_path(model_name))
    config["downloaded"] = is_model_downloaded(model_name)

    if config["downloaded"]:
        model_path = get_model_path(model_name)
        config["local_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)

    return config


def cleanup_models(model_name: Optional[str] = None) -> None:
    """
    Clean up downloaded models.

    Args:
        model_name: Specific model to remove, or None to remove all models
    """
    if model_name:
        if model_name not in MODELS_CONFIG:
            available_models = ", ".join(MODELS_CONFIG.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )

        model_path = get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Removed model '{model_name}' from {model_path}")
        else:
            logger.info(f"Model '{model_name}' was not downloaded")
    else:
        # Remove all models
        models_dir = get_models_dir()
        for model_file in models_dir.glob("*.pt"):
            model_file.unlink()
            logger.info(f"Removed {model_file}")

        # Also remove cache if it exists
        cache_dir = models_dir / "cache"
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            logger.info(f"Removed cache directory {cache_dir}")
