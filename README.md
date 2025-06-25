# SynthDoc

A comprehensive library for generating synthetic documents designed for training and evaluating models in document understanding tasks. SynthDoc supports multiple languages, fonts, and document layouts to create diverse training datasets for OCR models, document layout analysis, visual question answering, and retrieval systems.

## Features

✅ **Multi-language Support**: Generate documents in various languages with appropriate fonts and scripts
✅ **Flexible Document Generation**: Create documents from scratch using LLMs or augment existing documents
✅ **Layout Analysis**: Extract and manipulate document layouts for training layout detection models
✅ **VQA Dataset Creation**: Generate visual question-answering datasets with hard negatives for retrieval training
✅ **Handwriting Support**: Create handwritten documents using custom fonts and templates
✅ **HuggingFace Integration**: Output datasets in HuggingFace format for seamless integration with ML pipelines

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
3. Add realistic paper backgrounds and artifacts
4. Include common handwriting imperfections for robustness

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

### Minimal Workflow Usage

Each workflow has a simple configuration class and returns HuggingFace dataset format:

```python
from synthdoc.models import RawDocumentGenerationConfig, AugmentationType
from synthdoc.workflows import RawDocumentGenerator

# 1. Raw Document Generation
config = RawDocumentGenerationConfig(
    language="en",
    num_pages=3,
    augmentations=[AugmentationType.ROTATION, AugmentationType.NOISE]
)

generator = RawDocumentGenerator()
result = generator.process(config)

print(f"Generated {result.num_samples} samples")
print(f"Dataset: {result.dataset}")  # HuggingFace format
```

### All Workflows

```python
from synthdoc.models import *
from synthdoc.workflows import *

# 1. Raw Document Generation
raw_config = RawDocumentGenerationConfig(language="en", num_pages=2)
raw_result = RawDocumentGenerator().process(raw_config)

# 2. Layout Augmentation
layout_config = LayoutAugmentationConfig(
    documents=["doc1.pdf", "doc2.pdf"],
    languages=["en"],
    augmentations=[AugmentationType.SCALING]
)
layout_result = LayoutAugmenter().process(layout_config)

# 3. PDF Augmentation
pdf_config = PDFAugmentationConfig(
    pdf_files=["input.pdf"],
    augmentations=[AugmentationType.BLUR]
)
pdf_result = PDFAugmenter().process(pdf_config)

# 4. VQA Generation
vqa_config = VQAGenerationConfig(
    documents=["doc.pdf"],
    num_questions_per_doc=5,
    include_hard_negatives=True
)
vqa_result = VQAGenerator().process(vqa_config)

# 5. Handwriting Generation
handwriting_config = HandwritingGenerationConfig(
    text_content="Sample text",
    handwriting_style="cursive",
    num_samples=10
)
handwriting_result = HandwritingGenerator().process(handwriting_config)
```

All workflows return `WorkflowResult` with:
- `dataset`: HuggingFace format dictionary
- `metadata`: Additional information
- `num_samples`: Number of generated samples

### Available Augmentations

```python
from synthdoc.models import AugmentationType

# Available augmentation types:
AugmentationType.ROTATION
AugmentationType.SCALING  
AugmentationType.NOISE
AugmentationType.BLUR
AugmentationType.COLOR_SHIFT
AugmentationType.CROPPING
```

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Implement core document generation pipeline
    - [ ] Integrate LLMs for content generation
    - [ ] Add docling and MinerU support
- [ ] Add multi-language font support
- [ ] Develop layout augmentation engine
- [ ] Create VQA generation module
- [ ] Implement handwriting synthesis
- [ ] Add comprehensive testing suite
- [ ] Create documentation and tutorials