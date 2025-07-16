# SynthDoc

[![PyPI version](https://badge.fury.io/py/synthdoc.svg)](https://badge.fury.io/py/synthdoc)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive library for generating synthetic documents designed for training and evaluating models in document understanding tasks. SynthDoc supports multiple languages, fonts, and document layouts to create diverse training datasets for OCR models, document layout analysis, visual question answering, and retrieval systems.

## Key Features

-   🌐 **Multi-language Document Generation**: Create documents in various languages using LLMs.
-   ❓ **VQA Dataset Generation**: Produce rich Visual Question Answering datasets with hard negatives for robust model training.
-   🔄 **Document Translation**: Translate the text in documents to other languages while preserving the original layout.
-   🧩 **HuggingFace Integration**: Outputs datasets directly in HuggingFace `Dataset` format for seamless integration with your ML pipelines.
-   ⚙️ **Flexible Configuration**: Easily configure LLM providers (OpenAI, Groq, Anthropic, Ollama) and API keys using a `.env` file.
-   🚀 **Extensible Workflows**: A modular architecture that makes it easy to add new document generation and augmentation workflows.

## Available Workflows

### 1. Raw Document Generation

Generate synthetic documents from scratch using Large Language Models (LLMs) to create diverse content across multiple languages.

**Purpose**: Create original document content for training data augmentation and model robustness testing.

**Process**:
1.  An LLM generates contextually appropriate content in the specified language.
2.  The content is rendered into document images with proper formatting.
3.  The output is a standardized HuggingFace dataset.

**Output Schema**:
-   `image`: The generated document page as a PIL Image.
-   `image_path`: Path to the saved image file.
-   `page_number`: The page number.
-   `language`: The language of the generated content.
-   `prompt`: The prompt used for content generation.
-   And other metadata...

---

### 2. VQA (Visual Question Answering) Generation

Generate question-answer pairs for visual question answering tasks, including hard negatives for training robust retrieval models.

**Purpose**: Create comprehensive VQA datasets for training and evaluating visual document understanding models.

**Process**:
1.  **General VQA**: Generate diverse question-answer pairs about document content, layout, and visual elements.
2.  **Hard Negative VQA**: Create challenging negative examples that are semantically similar but factually incorrect.
3.  **Similarity Scoring**: Generate similarity scores for retrieval training.

**Output Schema (extends raw document schema)**:
-   `questions`: List of generated questions.
-   `answers`: Corresponding ground truth answers.
-   `hard_negatives`: Challenging incorrect answers.
-   And other VQA-related metadata...

---

### 3. Document Translation

Translate the text within a document image to one or more target languages while preserving the visual layout.

**Purpose**: Adapt existing document datasets for multi-lingual model training.

**Process**:
1.  **Layout Detection**: A YOLO-based model detects text blocks in the source image.
2.  **OCR**: Text is extracted from each detected block.
3.  **Translation**: The extracted text is translated to the target language(s).
4.  **Rendering**: The translated text is rendered back onto a copy of the original image in the same location, using appropriate fonts.

**Output**: A HuggingFace dataset containing the translated document images.

## Installation

### Basic Installation
```bash
pip install synthdoc
```

### With LLM Support (Recommended)
To enable content generation with LLMs, install with the `llm` extra:
```bash
pip install synthdoc[llm]
```

### For Development
```bash
git clone https://github.com/adithya-s-k/SynthDoc.git
cd SynthDoc
pip install -e .[llm]
```

## Quick Start

### 1. Configure Environment (Recommended)

SynthDoc uses [LiteLLM](https://github.com/BerriAI/litellm) to connect to 100+ LLM providers.

1.  **Copy the environment template**:
    ```bash
    cp env.template .env
    ```

2.  **Edit your `.env` file** and add your API keys. The `DEFAULT_LLM_MODEL` will be used for generation.
    ```env
    # .env file
    OPENAI_API_KEY=your_openai_key
    GROQ_API_KEY=your_groq_key
    DEFAULT_LLM_MODEL=gpt-4o-mini
    ```

### 2. Use SynthDoc

Create a `main.py` file:
```python
from synthdoc import SynthDoc

# SynthDoc automatically loads from .env file
# It will prompt you to specify an output directory on first run.
synth = SynthDoc()

# --- Workflow 1: Generate Raw Documents ---
print("Generating raw documents...")
raw_dataset = synth.generate_raw_docs(
    language="en", 
    num_pages=2,
    prompt="Generate a short technical report about climate change."
)
print(f"Generated {len(raw_dataset)} raw documents.")
print(raw_dataset[0])


# --- Workflow 2: Generate VQA ---
print("\nGenerating VQA dataset...")
vqa_dataset = synth.generate_vqa(
    source_documents=raw_dataset,
    num_questions_per_doc=3
)
print(f"Generated {len(vqa_dataset)} VQA samples.")
print(vqa_dataset[0]['questions'])


# --- Workflow 3: Translate Documents ---
print("\nTranslating documents...")
# This workflow requires a local YOLO model, which will be auto-downloaded.
translated_dataset = synth.translate_documents(
    input_dataset=raw_dataset,
    target_languages=["es", "fr"] # Translate to Spanish and French
)
print(f"Translated documents into {len(translated_dataset)} language versions.")
print(translated_dataset)
```

### Manual Configuration (Not Recommended)

You can override the `.env` configuration by passing parameters directly.

```python
# Override model and API key
synth_manual = SynthDoc(llm_model="groq/llama-3-8b-8192", api_key="your-groq-key")

# Use local Ollama models (no API key needed)
synth_ollama = SynthDoc(llm_model="ollama/llama2")
```

## Roadmap

-   [x] Core document generation pipeline
-   [x] Multi-language content generation via LLMs
-   [x] VQA generation module
-   [x] Document translation workflow
-   [ ] **Layout Augmentation**: Programmatically alter document layouts.
-   [ ] **PDF Augmentation**: Recombine elements from a corpus of PDFs to create new documents.
-   [ ] **Handwriting Synthesis**: Generate documents with realistic handwritten fonts.
-   [ ] **Comprehensive Testing**: Increase test coverage across all workflows.
-   [ ] **Detailed Tutorials**: Create in-depth documentation and examples.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.