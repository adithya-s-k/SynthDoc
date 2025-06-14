# SynthDoc Quickstart Guide

This guide will help you get started with SynthDoc, including setting up LLM integration for advanced document generation.

## Installation

### Option 1: Basic Installation
```bash
pip install synthdoc
```

### Option 2: With LLM Support (Recommended)
```bash
pip install synthdoc[llm]
```

This installs SynthDoc with LiteLLM for advanced document generation using various LLM providers.

## Basic Usage

### 1. Python API

```python
from synthdoc import SynthDoc

# Initialize
synth = SynthDoc(output_dir="./output")

# Generate documents in English
docs = synth.generate_raw_docs(
    language="en", 
    num_pages=5,
    prompt="Generate technical documentation"
)

# Generate multilingual documents
for lang in ["hi", "zh", "es"]:
    docs = synth.generate_raw_docs(
        language=lang,
        num_pages=3
    )

# Apply layout augmentation
dataset = synth.augment_layout(
    documents=docs,
    languages=["en", "hi"],
    augmentations=["rotation", "scaling", "noise"]
)
```

### 2. Command Line Interface

```bash
# Generate documents
synthdoc generate --lang hi --pages 10 --output ./output

# Show supported languages
synthdoc languages

# Apply layout augmentation
synthdoc layout ./input_docs --output ./output --lang en hi

# Get help
synthdoc --help
```

## Supported Languages

The library supports 22 languages across different script systems:

| Category | Languages | Scripts |
|----------|-----------|---------|
| **Base** | English | Latin |
| **Indic** | Hindi, Kannada, Tamil, Telugu, Marathi, Punjabi, Bengali, Odia, Malayalam, Gujarati, Sanskrit | Devanagari, Kannada, Tamil, Telugu, Gurmukhi, Bengali, Odia, Malayalam, Gujarati |
| **Other** | Japanese, Korean, Chinese, German, French, Italian, Russian, Arabic, Spanish, Thai | Kanji/Kana, Hangul, Simplified Chinese, Latin, Cyrillic, Arabic, Thai |

## Core Features

### 1. Raw Document Generation
- LLM-powered content generation
- Multi-language support
- Custom prompts
- Built-in augmentations

### 2. Layout Augmentation
- Font variations
- Visual transformations
- HuggingFace dataset output
- Annotation preservation

### 3. VQA Generation
- Question-answer pairs
- Hard negatives
- Multiple difficulty levels
- Retrieval training support

### 4. Handwriting Synthesis
- Multiple writing styles
- Paper templates
- Natural variations

## Configuration

Create a config file to customize behavior:

```python
from synthdoc.config import SynthDocConfig, DocumentConfig

config = SynthDocConfig()
config.document.default_language = "hi"
config.document.image_dpi = 600
config.augmentation.default_augmentations = ["rotation", "noise", "brightness"]

synth = SynthDoc(config=config)
```

## Output Formats

SynthDoc generates HuggingFace-compatible datasets with:

```
{
    "image": Document image (PNG/JPEG),
    "image_width": Image width in pixels,
    "pdf_name": Source document identifier,
    "page_number": Page number (0-indexed),
    "markdown": Document content in Markdown,
    "html": Document content in HTML,
    "layout": Layout annotation data,
    "lines": Text line detection annotations,
    "images": Embedded image annotations,
    "equations": Mathematical equation annotations,
    "tables": Table structure annotations,
    "metadata": Additional document metadata
}
```

## Examples

See the `examples/` directory for complete examples:
- `basic_usage.py` - Basic functionality demo
- `multilingual_demo.py` - Multilingual document generation
- `advanced_augmentation.py` - Complex augmentation pipelines
