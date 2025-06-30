# SynthDoc Usage Guide

This guide covers all the implemented features of SynthDoc with practical examples and best practices.

## Installation

### Basic Installation
```bash
pip install synthdoc
```

### With Full Features
```bash
# Install with LLM support and image processing
pip install synthdoc[llm]
pip install pillow opencv-python numpy
```

### Development Installation
```bash
git clone <repository-url>
cd SynthDoc
pip install -e .[llm]
```

## Quick Start

### Basic Document Generation

```python
from synthdoc import SynthDoc

# Initialize SynthDoc
synth = SynthDoc(output_dir="./output")

# Generate documents
docs = synth.generate_raw_docs(
    language="en",
    num_pages=3,
    prompt="Generate technical documentation about AI"
)

print(f"Generated {len(docs)} documents")
for doc in docs:
    print(f"Document {doc['id']}: {doc['image_path']}")
```

### With LLM Integration

```python
import os
from synthdoc import SynthDoc

# Set up LLM integration
synth = SynthDoc(
    output_dir="./output",
    llm_model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Generate high-quality content
docs = synth.generate_raw_docs(
    language="en",
    num_pages=2,
    prompt="Create a technical report on machine learning algorithms"
)
```

## Core Features

### 1. Document Generation

#### Multi-Language Support

```python
from synthdoc import SynthDoc, LanguageSupport

synth = SynthDoc()

# Check supported languages
languages = synth.get_supported_languages()
print(f"Supported: {len(languages)} languages")

# Generate in different languages
for lang in ["en", "hi", "zh", "ar"]:
    docs = synth.generate_raw_docs(
        language=lang,
        num_pages=1,
        prompt=f"Technical document in {lang}"
    )
    print(f"Generated {lang}: {docs[0]['image_path']}")
```

#### Language Information

```python
# Get detailed language info
lang_info = synth.get_language_info("hi")
print(f"Language: {lang_info['name']}")
print(f"Script: {lang_info['script']}")
print(f"RTL: {lang_info['rtl']}")
print(f"Fonts: {lang_info['fonts']}")
```

### 2. Layout Augmentation

#### Basic Augmentation

```python
# Apply layout transformations
augmented = synth.augment_layout(
    documents=docs,
    languages=["en", "hi"],
    fonts=["Arial", "Times New Roman", "Helvetica"],
    augmentations=["rotation", "scaling", "brightness"]
)

print(f"Generated {len(augmented['images'])} variations")
```

#### Advanced Image Processing

```python
from synthdoc.augmentations import Augmentor

augmentor = Augmentor()

# Available augmentations
print("Available:", augmentor.get_available_augmentations())

# Apply specific augmentations
image = docs[0]["image"]
rotated = augmentor.apply_rotation(image, intensity=0.7)
noisy = augmentor.apply_noise(image, intensity=0.3)
blurred = augmentor.apply_blur(image, intensity=0.5)
```

#### Augmentation Parameters

```python
# Customize augmentation intensity
augmentations = {
    "rotation": 0.8,    # High rotation variation
    "brightness": 0.3,  # Subtle brightness changes
    "noise": 0.4,       # Moderate noise
    "scaling": 0.6,     # Medium scaling variation
}

for aug_type, intensity in augmentations.items():
    if aug_type == "rotation":
        aug_image = augmentor.apply_rotation(image, intensity)
    elif aug_type == "brightness":
        aug_image = augmentor.apply_brightness(image, intensity)
    # ... etc
```

### 3. VQA Generation

#### Basic VQA Dataset

```python
# Generate VQA pairs
vqa_dataset = synth.generate_vqa(
    source_documents=docs,
    question_types=["factual", "reasoning", "comparative"],
    difficulty_levels=["easy", "medium", "hard"],
    hard_negative_ratio=0.3
)

print(f"Generated {len(vqa_dataset['questions'])} Q&A pairs")

# Access VQA data
for i in range(3):
    print(f"Q: {vqa_dataset['questions'][i]}")
    print(f"A: {vqa_dataset['answers'][i]}")
    print(f"Type: {vqa_dataset['question_types'][i]}")
    print(f"Hard Negatives: {vqa_dataset['hard_negatives'][i]}")
    print("---")
```

#### Question Types

- **factual**: Direct questions about document content
- **reasoning**: Questions requiring logical inference
- **comparative**: Questions comparing document elements
- **structural**: Questions about document layout

#### Difficulty Levels

- **easy**: Simple, direct questions
- **medium**: Requires understanding context
- **hard**: Complex reasoning or inference

### 4. Handwriting Generation

#### Basic Handwriting

```python
# Generate handwritten documents
handwritten = synth.generate_handwriting(
    content="Sample handwritten text",
    language="en",
    writing_style="print",
    paper_template="lined"
)

print(f"Handwritten image: {handwritten['image_path']}")
```

#### Writing Styles and Paper Templates

```python
# Different writing styles
styles = ["print", "cursive", "mixed"]
papers = ["blank", "lined", "grid"]

for style in styles:
    for paper in papers:
        handwritten = synth.generate_handwriting(
            content=f"Sample {style} on {paper} paper",
            language="en",
            writing_style=style,
            paper_template=paper
        )
```

#### Multi-Language Handwriting

```python
# Handwriting in different languages
samples = [
    ("en", "English handwriting sample"),
    ("hi", "हिंदी हस्तलेखन नमूना"),
    ("zh", "中文手写示例")
]

for lang, content in samples:
    handwritten = synth.generate_handwriting(
        content=content,
        language=lang,
        writing_style="print",
        paper_template="lined"
    )
```

## Dataset Management

### Creating Datasets

```python
from synthdoc.dataset_manager import create_dataset_manager
from synthdoc.models import DatasetItem, SplitType, create_document_metadata

# Create dataset manager
dataset_manager = create_dataset_manager(
    dataset_root="./datasets",
    dataset_name="my_synthetic_docs",
    copy_images=True
)

# Add documents to dataset
for i, doc in enumerate(docs):
    if doc.get("image_path"):
        metadata = create_document_metadata(
            file_name=f"doc_{i}.png",
            text=doc["content"],
            document_type="technical",
            source="synthdoc"
        )
        
        item = DatasetItem(
            image_path=doc["image_path"],
            metadata=metadata,
            split=SplitType.TRAIN if i < 8 else SplitType.TEST
        )
        
        filename = dataset_manager.add_item(item)
        print(f"Added {filename}")

# Get dataset statistics
stats = dataset_manager.get_stats()
print(f"Total items: {stats.total_items}")
print(f"Splits: {dict(stats.split_counts)}")
```

### Workflow Integration

```python
from synthdoc.workflows import create_document_workflow, create_vqa_workflow

# Document understanding workflow
doc_workflow = create_document_workflow(
    dataset_root="./workflows",
    workflow_name="document_understanding"
)

doc_workflow.initialize_dataset("my_doc_dataset")

# Add samples
for doc in docs:
    doc_workflow.add_document_sample(
        image_path=doc["image_path"],
        text=doc["content"],
        document_type="technical_report",
        split=SplitType.TRAIN
    )

# Finalize and get HuggingFace dataset
hf_dataset = doc_workflow.finalize_dataset()
```

## Command Line Interface

### Basic Commands

```bash
# Generate documents
synthdoc generate --lang en --pages 5 --output ./output

# Show supported languages
synthdoc languages

# Apply layout augmentation
synthdoc layout ./input_docs --output ./output --lang en hi

# Generate VQA dataset
synthdoc vqa ./docs --output ./vqa_dataset --type factual reasoning
```

### Advanced CLI Usage

```bash
# With custom LLM model
synthdoc generate --lang hi --pages 3 --model gpt-4 --api-key $OPENAI_API_KEY

# With specific augmentations
synthdoc layout ./docs --augment rotation scaling brightness --font Arial "Times New Roman"

# VQA with hard negatives
synthdoc vqa ./docs --hard-negative-ratio 0.4 --difficulty easy medium hard
```

## Configuration

### Custom Configuration

```python
from synthdoc.config import SynthDocConfig, DocumentConfig

# Create custom config
config = SynthDocConfig()
config.document.default_language = "hi"
config.document.image_dpi = 600
config.augmentation.default_augmentations = ["rotation", "noise", "brightness"]

# Use with SynthDoc
synth = SynthDoc(config=config)
```

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: Configure model preferences
export SYNTHDOC_DEFAULT_MODEL="gpt-3.5-turbo"
export SYNTHDOC_OUTPUT_DIR="./my_output"
```

## Best Practices

### Performance Optimization

```python
# Batch processing for large datasets
docs = []
for i in range(0, 100, 10):  # Process in batches of 10
    batch = synth.generate_raw_docs(
        language="en",
        num_pages=10,
        prompt=f"Batch {i//10} documents"
    )
    docs.extend(batch)
    print(f"Processed batch {i//10}")
```

### Memory Management

```python
# For large image processing tasks
from synthdoc.augmentations import Augmentor

augmentor = Augmentor()

# Process images in batches to avoid memory issues
batch_size = 10
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    augmented_batch = augmentor.batch_augment_images(
        batch,
        augmentations=["rotation", "brightness"],
        intensity=0.5
    )
    # Process or save augmented_batch
```

### Quality Control

```python
# Validate generated content
def validate_document(doc):
    """Validate generated document quality."""
    if not doc.get("content"):
        return False, "No content"
    
    if len(doc["content"]) < 100:
        return False, "Content too short"
    
    if not doc.get("image_path"):
        return False, "No image generated"
    
    return True, "Valid"

# Apply validation
valid_docs = []
for doc in docs:
    is_valid, reason = validate_document(doc)
    if is_valid:
        valid_docs.append(doc)
    else:
        print(f"Invalid doc {doc['id']}: {reason}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Install missing dependencies
   pip install pillow opencv-python numpy
   ```

2. **LLM Connection Issues**
   ```python
   # Check API key
   import os
   print("API Key set:", bool(os.getenv("OPENAI_API_KEY")))
   
   # Use fallback mode
   synth = SynthDoc()  # Works without API key
   ```

3. **Image Processing Errors**
   ```python
   # Check PIL availability
   try:
       from PIL import Image
       print("PIL available")
   except ImportError:
       print("Install PIL: pip install pillow")
   ```

4. **Font Issues**
   ```python
   # Check available fonts
   from synthdoc.fonts import FontManager
   
   font_manager = FontManager()
   fonts = font_manager.list_all_fonts()
   print(f"Available fonts: {len(fonts)}")
   ```

### Debugging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug logging
synth = SynthDoc(output_dir="./debug_output", log_level="DEBUG")
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Basic functionality demonstration
- `integration_example.py` - Comprehensive feature showcase
- `multilingual_demo.py` - Multi-language document generation

## Testing

Run the test suite to verify your installation:

```bash
python test_implementation.py
```

This will test all core functionality and provide a detailed report.

## API Reference

For detailed API documentation, see the docstrings in the source code or generate documentation with:

```bash
pip install sphinx
sphinx-build -b html docs/ docs/_build/
``` 