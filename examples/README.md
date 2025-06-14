# SynthDoc Examples

This directory contains example scripts demonstrating how to use SynthDoc.

## Files

- `basic_usage.py` - Basic usage examples covering all main features
- `multilingual_demo.py` - Demonstrations focused on multilingual support
- `advanced_augmentation.py` - Advanced augmentation techniques
- `vqa_pipeline.py` - Complete VQA dataset generation pipeline

## Running Examples

```bash
# Install SynthDoc first
cd ..
poetry install

# Run basic usage example
python examples/basic_usage.py

# Or with the CLI
synthdoc generate --lang hi --pages 5 --output ./output
synthdoc languages  # Show all supported languages
```

## Expected Output

Examples will create output in the specified directories with:
- Generated document images
- Annotation files
- HuggingFace dataset files
- Metadata and configuration files
