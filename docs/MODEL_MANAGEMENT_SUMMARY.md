# SynthDoc Model Management Update

This update implements automatic model downloading for SynthDoc, eliminating the need to manually store large model files in the repository.

## What Changed

### 🚀 New Features

1. **Automatic Model Download**: Models are automatically downloaded from HuggingFace when first needed
2. **Model Management System**: Complete system for managing pre-trained models
3. **CLI Commands**: New CLI commands for model management
4. **Smart Caching**: Models are cached locally and reused

### 📦 New Files

- `synthdoc/models_manager.py` - Core model management functionality
- `synthdoc/downloaded_models/` - Directory where models are stored (auto-created)
- `examples/model_management_example.py` - Example usage
- `test_models.py` - Test script for model management

### 🔧 Modified Files

- `synthdoc/core.py` - Updated to use model manager for YOLO model
- `synthdoc/__init__.py` - Added model management exports
- `synthdoc/workflows/document_translator/workflow.py` - Updated to auto-download YOLO model
- `synthdoc/models.py` - Made yolo_model_path optional with auto-download
- `synthdoc/cli.py` - Added model management CLI commands
- `.gitignore` - Added model files to gitignore

## How It Works

### 1. Model Configuration

Models are configured in `MODELS_CONFIG` in `models_manager.py`:

```python
MODELS_CONFIG = {
    "doclayout-yolo": {
        "repo_id": "opendatalab/PDF-Extract-Kit-1.0",
        "filename": "models/Layout/YOLO/doclayout_yolo_ft.pt",
        "local_filename": "doclayout_yolo_ft.pt",
        "description": "Document layout analysis YOLO model",
        "size_mb": 52,
        "url": "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt",
    }
}
```

### 2. Automatic Download

When a model is needed:

```python
from synthdoc import SynthDoc

# This will automatically download the YOLO model if needed
synth = SynthDoc()
result = synth.translate_documents(
    input_images=["document.png"],
    target_languages=["hi"]
)
```

### 3. Manual Management

```python
from synthdoc import download_model, list_available_models, get_model_info

# List available models
models = list_available_models()

# Get model info
info = get_model_info("doclayout-yolo")

# Download a specific model
download_model("doclayout-yolo")
```

### 4. CLI Commands

```bash
# List available models
synthdoc list-models

# Download a specific model
synthdoc download-models doclayout-yolo

# Download all models
synthdoc download-models

# Get model info
synthdoc model-info doclayout-yolo

# Clean up models
synthdoc clean-models
```

## Download Methods

The system supports two download methods with automatic fallback:

1. **HuggingFace Hub** (preferred): Uses `huggingface_hub` library
2. **Direct HTTP** (fallback): Uses `requests` for direct download

## Storage Location

Models are stored in: `synthdoc/downloaded_models/`

This directory is:
- Auto-created when needed
- Added to `.gitignore` to prevent committing large files
- Persistent across installations

## Benefits

### ✅ For Users
- No need to manually download large model files
- Models download automatically when needed
- Clear progress indicators during download
- Easy model management with CLI tools

### ✅ For Repository
- Reduced repository size (removed ~50MB model file)
- No large file storage in Git history
- Cleaner repository structure
- Easier for contributors to clone

### ✅ For Development
- Models stay up-to-date with latest versions
- Easy to add new models
- Supports multiple download sources
- Robust error handling with fallbacks

## Usage Examples

### Basic Usage (Automatic)

```python
from synthdoc import SynthDoc

# Model downloads automatically on first use
synth = SynthDoc()
result = synth.translate_documents(
    input_images=["document.png"],
    target_languages=["hi"]
)
```

### Manual Model Management

```python
from synthdoc import download_model, get_model_info, is_model_downloaded

# Check if model is downloaded
if not is_model_downloaded("doclayout-yolo"):
    print("Downloading model...")
    download_model("doclayout-yolo")

# Get model information
info = get_model_info("doclayout-yolo")
print(f"Model status: {'Downloaded' if info['downloaded'] else 'Not downloaded'}")
```

### CLI Usage

```bash
# See what models are available
synthdoc list-models

# Download specific model
synthdoc download-models doclayout-yolo

# Check model details
synthdoc model-info doclayout-yolo
```

## Migration Notes

### For Existing Users

- **No breaking changes**: Existing code continues to work
- **Automatic migration**: Models download automatically when needed
- **Optional manual pre-download**: Use CLI or API to download models in advance

### For Developers

- Remove any hardcoded model paths pointing to `./model-doclayout-yolo.pt`
- Models are now accessed via the model manager
- Add new models to `MODELS_CONFIG` in `models_manager.py`

## Testing

Run the test suite:

```bash
# Test model management system
python test_models.py

# Test actual downloading
python -c "from synthdoc.models_manager import download_model; download_model('doclayout-yolo')"

# Run example
python examples/model_management_example.py
```

## Future Enhancements

- Support for model versioning
- Automatic updates for newer model versions
- Support for custom model repositories
- Model compression/optimization
- Distributed model serving

## Summary

This update modernizes SynthDoc's model management by:

1. **Removing large files from the repository**
2. **Implementing automatic model downloading**
3. **Providing comprehensive model management tools**
4. **Maintaining backward compatibility**
5. **Adding robust error handling and fallbacks**

The system is designed to be user-friendly, developer-friendly, and maintainable for the long term.
