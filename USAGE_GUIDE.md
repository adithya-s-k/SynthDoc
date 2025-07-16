# SynthDoc Workflow Usage Guide

This guide shows how to use the 3 core workflows in SynthDoc.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env with your API keys
```

## 1. Raw Document Generator

Generate synthetic documents from scratch using LLM.

```python
from synthdoc.models import RawDocumentGenerationConfig
from synthdoc.workflows import RawDocumentGenerator
from synthdoc.languages import Language

# Basic configuration
config = RawDocumentGenerationConfig(
    language=Language.EN,
    num_pages=2,
    prompt="Write about machine learning"
)

# Generate documents
generator = RawDocumentGenerator(save_dir="my_documents")
result = generator.process(config)

print(f"Generated {result.num_samples} pages")
print(f"Output: {result.metadata['output_structure']['output_dir']}")
```

**Output Structure:**
```
my_documents/
├── images/
│   ├── document_123_page_1.png
│   └── document_123_page_2.png
└── metadata.jsonl
```

## 2. Document Translation

Translate existing documents to different languages while preserving layout.

```python
from synthdoc.models import DocumentTranslationConfig
from synthdoc.workflows import DocumentTranslator

# Basic configuration
config = DocumentTranslationConfig(
    input_images=["document1.pdf", "document2.png"],
    target_languages=["hi", "es"]
)

# Translate documents
translator = DocumentTranslator(save_dir="translated_docs")
result = translator.process(config)

print(f"Translated {result.num_samples} documents")
```

**Output Structure:**
```
translated_docs/
├── images/
│   ├── translated_hi_0_document1.jpg
│   ├── translated_es_0_document1.jpg
│   └── translated_hi_1_document2.jpg
└── metadata.jsonl
```

## 3. VQA Generation

Generate visual question-answer pairs from images or PDFs.

```python
from synthdoc.models import VQAGenerationConfig
from synthdoc.workflows import VQAGenerator

# Basic configuration
config = VQAGenerationConfig(
    documents=["image1.png", "document.pdf"],
    num_questions_per_doc=5
)

# Generate VQA pairs
generator = VQAGenerator(api_key="your_gemini_api_key", save_dir="vqa_output")
result = generator.process(config)

print(f"Generated {result.num_samples} question-answer pairs")
```

**Output Structure:**
```
vqa_output/
├── images/
│   ├── image1.png
│   └── document_page_1.png
└── metadata.jsonl
```

## Common Options

### Languages
```python
from synthdoc.languages import Language

# Raw document generation
config = RawDocumentGenerationConfig(
    language=Language.HI,  # Hindi
    # Language.EN, Language.FR, Language.ES, etc.
)

# Document translation
config = DocumentTranslationConfig(
    target_languages=["hi", "bn", "ta"]  # Hindi, Bengali, Tamil
)
```

### Input Sources
```python
# Single files
config = VQAGenerationConfig(
    documents=["document.pdf"]
)

# Multiple files
config = VQAGenerationConfig(
    documents=["image1.png", "image2.jpg", "document.pdf"]
)

# Folders
config = VQAGenerationConfig(
    documents=["./images_folder", "./pdfs_folder"]
)
```

### Output Directories
```python
# Custom output directory
generator = RawDocumentGenerator(save_dir="custom_output")
translator = DocumentTranslator(save_dir="custom_output")
vqa_gen = VQAGenerator(save_dir="custom_output")
```

## Loading Results

All workflows create datasets that can be loaded:

```python
from datasets import load_dataset

# Load from output directory
dataset = load_dataset("imagefolder", data_dir="my_documents")

# Access the data
for item in dataset["train"]:
    print(f"Image: {item['image']}")
    print(f"Metadata: {item['metadata']}")
```

## Notes

- All workflows use default model paths and fonts automatically
- API keys should be set in your `.env` file
- Results are saved in standardized `images/` + `metadata.jsonl` format
- Output is compatible with HuggingFace datasets 