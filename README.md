# SynthDoc

A comprehensive library for generating synthetic documents designed for training and evaluating models in document understanding tasks. SynthDoc supports multiple languages, fonts, and document layouts to create diverse training datasets for OCR models, document layout analysis, visual question answering, and retrieval systems.

## Features

✅ **Multi-language Support**: Generate documents in various languages with appropriate fonts and scripts (Fully Implemented)
✅ **Flexible Document Generation**: Create documents from scratch using LLMs or augment existing documents (Fully Implemented)
✅ **Layout Analysis**: Extract and manipulate document layouts for training layout detection models (Fully Implemented)
✅ **VQA Dataset Creation**: Generate visual question-answering datasets with hard negatives for retrieval training (Fully Implemented)
✅ **Handwriting Support**: Create handwritten documents using custom fonts and templates with paper styles (Fully Implemented)
✅ **PDF Document Recombination**: Extract and recombine elements from existing PDF documents (Fully Implemented)
✅ **HuggingFace Integration**: Output datasets in HuggingFace format for seamless integration with ML pipelines (Fully Implemented)

## Workflows

### 1. Raw Document Generation

Generate synthetic documents from scratch using Large Language Models (LLMs) to create diverse content across multiple languages.

**Purpose**: Create original document content for training data augmentation and model robustness testing.

**Input Arguments**:
- `language`: Target language for content generation (e.g., 'en', 'es', 'fr', 'zh')
- `num_pages`: Number of pages to generate (integer)
- `prompt`: Optional custom prompt for specific content generation. If not provided, content is randomized
- `augmentations`: Optional list of augmentation techniques to apply (enum list)

**Process**:
1. LLM generates contextually appropriate content in the specified language
2. Content is structured into document format with proper formatting
3. Optional augmentations are applied (rotation, noise, etc.)

**Output**: Documents with rich textual content ready for layout application

---

### 2. Layout Augmentation

Apply various layout transformations and styling to existing documents or generated content to create visually diverse training data.

**Purpose**: Enhance document visual diversity for layout detection and document understanding model training.

**Input Arguments**:
- `documents`: List of PDF files or images to process
- `languages`: List of target languages for text content
- `fonts`: List of font families to apply
- `augmentations`: Optional list of visual augmentations (rotation, scaling, cropping, color changes, etc.)
- `layout_templates`: Optional predefined layout templates

**Process**:
1. Extract content from input documents
2. Apply different fonts and language-specific formatting
3. Transform layouts using specified augmentation techniques
4. Generate multiple variations per input document

**Output**: HuggingFace dataset with the following schema:
```
- image: Document image (PNG/JPEG)
- image_width: Image width in pixels
- pdf_name: Source document identifier
- page_number: Page number (0-indexed)
- markdown: Document content in Markdown format
- html: Document content in HTML format
- layout: Layout annotation data (bounding boxes, element types)
- lines: Text line detection annotations
- images: Embedded image annotations
- equations: Mathematical equation annotations
- tables: Table structure annotations
- page_size: Document page dimensions
- content_list: Structured content elements
- base_layout_detection: Layout detection ground truth
- pdf_info: Document metadata
```

---

### 3. PDF Augmentation

Create new randomized documents by extracting and recombining elements from a corpus of existing documents.

**Purpose**: Generate novel document layouts by mixing and matching elements from existing documents, creating diverse training scenarios.

**Input Arguments**:
- `corpus`: Collection of source images or PDF files
- `extraction_elements`: Types of elements to extract (text blocks, images, tables, headers, etc.)
- `combination_strategy`: Method for combining extracted elements
- `output_layout_types`: Target layout styles for generated documents

**Process**:
1. Analyze corpus documents to extract reusable elements
2. Categorize elements by type (headers, paragraphs, images, tables, etc.)
3. Intelligently combine elements to create coherent new documents
4. Ensure proper spacing, alignment, and visual hierarchy

**Output**: Same HuggingFace dataset schema as Layout Augmentation workflow

---

### 4. VQA (Visual Question Answering) Generation

Generate question-answer pairs for visual question answering tasks, including hard negatives for training robust retrieval models.

**Purpose**: Create comprehensive VQA datasets for training and evaluating visual document understanding models.

**Input Arguments**:
- `source_documents`: Documents to generate questions about
- `question_types`: Types of questions to generate (factual, reasoning, comparative, etc.)
- `difficulty_levels`: Question complexity levels
- `hard_negative_ratio`: Ratio of hard negative examples to include

**Process**:
1. **General VQA**: Generate diverse question-answer pairs about document content, layout, and visual elements
2. **Hard Negative VQA**: Create challenging negative examples that are semantically similar but factually incorrect
3. **Similarity Scoring**: Generate similarity scores for retrieval training

**Output**: Extended dataset with additional fields:
```
- questions: List of generated questions
- answers: Corresponding ground truth answers
- hard_negatives: Challenging incorrect answers
- question_types: Question category classifications
- difficulty_scores: Question complexity ratings
- similarity_scores: Answer similarity metrics
```

---

### 5. Handwriting Font-Based Generation

Generate documents with handwritten appearance using custom handwriting fonts and templates.

**Purpose**: Create handwritten document datasets for training OCR models on handwritten text and mixed content documents.

**Input Arguments**:
- `handwriting_template`: Handwriting style template or font
- `language`: Target language for content
- `writing_style`: Cursive, print, or mixed handwriting styles
- `paper_templates`: Background paper styles (lined, grid, blank, etc.)

**Process**:
1. Apply handwriting fonts to generated or existing content
2. Simulate natural handwriting variations (spacing, alignment, pressure)
3. Add realistic paper backgrounds and artifacts (lined paper, grid paper, or blank)
4. Include common handwriting imperfections for robustness

**Paper Template Options**:
- `lined`: Traditional lined notebook paper with horizontal lines and margin
- `grid`: Graph paper with grid lines for technical writing
- `blank`: Clean paper with subtle texture and minimal guidelines

**Output**: HuggingFace dataset optimized for handwritten document understanding tasks

## Installation

### Basic Installation
```bash
pip install synthdoc
```

### With LLM Support (Recommended)
```bash
# Install with LiteLLM for advanced document and VQA generation
pip install synthdoc[llm]

# Or install manually
pip install synthdoc litellm
```

### For Development
```bash
git clone <repository-url>
cd SynthDoc
pip install -e .[llm]
```

## Quick Start

### Setup with .env Configuration (Recommended)

1. **Install SynthDoc**:
```bash
pip install synthdoc[llm]  # Includes LLM support
```

2. **Configure environment**:
```bash
# Copy the environment template
cp env.template .env

# Edit .env file and add your API keys
# OPENAI_API_KEY=your_openai_key_here
# GROQ_API_KEY=your_groq_key_here
# etc.
```

3. **Use SynthDoc with automatic configuration**:
```python
from synthdoc import SynthDoc

# SynthDoc automatically loads from .env file
synth = SynthDoc()  # API keys and models auto-detected

# Generate raw documents using LLM
documents = synth.generate_raw_docs(
    language="en", 
    num_pages=3,
    prompt="Generate a technical report about AI"
)

# Apply layout augmentation
dataset = synth.augment_layout(
    documents=documents,
    fonts=["Arial", "Times New Roman"],
    augmentations=["rotation", "scaling"]
)

# Generate VQA pairs using LLM
vqa_dataset = synth.generate_vqa(
    source_documents=documents,
    question_types=["factual", "reasoning", "comparative"],
    difficulty_levels=["easy", "medium", "hard"]
)

# Generate handwritten documents with different paper styles
handwriting_lined = synth.generate_handwriting(
    content="Practice handwriting on lined paper",
    writing_style="cursive",
    paper_template="lined"
)

handwriting_grid = synth.generate_handwriting(
    content="Technical notes on grid paper",
    writing_style="print", 
    paper_template="grid"
)

handwriting_blank = synth.generate_handwriting(
    content="Creative writing on blank paper",
    writing_style="mixed",
    paper_template="blank"
)
```

### Environment Status Check

```python
from synthdoc import print_environment_status

# Check your configuration
print_environment_status()
```

### Using Different LLM Providers

SynthDoc uses [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM access. Configure providers in `.env`:

```bash
# .env file configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  
GROQ_API_KEY=your_groq_key
DEFAULT_LLM_MODEL=gpt-4o-mini
```

```python
# Automatic provider selection (uses .env)
synth = SynthDoc()  # Auto-detects best available provider

# Manual provider override
synth_openai = SynthDoc(llm_model="gpt-4o-mini")
synth_claude = SynthDoc(llm_model="claude-3-5-sonnet-20241022")
synth_groq = SynthDoc(llm_model="groq/llama-3-8b-8192")

# Local Ollama models (no API key needed)
synth_ollama = SynthDoc(llm_model="ollama/llama2")
```

### Manual API Key Override (Not Recommended)

```python
# You can still manually pass API keys if needed
synth = SynthDoc(llm_model="gpt-4o-mini", api_key="your-key-here")
```

All workflows return `WorkflowResult` with:
- `dataset`: HuggingFace format dictionary
- `metadata`: Additional information
- `num_samples`: Number of generated samples

### Available Augmentations

```python
# Initialize without API keys in .env file
synth = SynthDoc()  # Will use fallback mode if no API keys found

# Still generates documents but with template content
documents = synth.generate_raw_docs(language="en", num_pages=2)

# Disable automatic .env loading if needed
synth = SynthDoc(load_dotenv=False)
```

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [x] Implement core document generation pipeline
    - [x] Integrate LLMs for content generation
    - [x] Document rendering (text to images)
    - [ ] Add docling and MinerU support
- [x] Add multi-language font support
- [x] Develop layout augmentation engine
- [x] Create VQA generation module
- [x] Implement handwriting synthesis
- [x] Add comprehensive testing suite
- [x] Create documentation and tutorials