# SynthDoc Dataset Management

SynthDoc now includes comprehensive, type-safe dataset management capabilities for creating and managing datasets in HuggingFace ImageFolder format. This system is designed to work seamlessly with incremental workflows where data is generated continuously.

## Features

- **Type-Safe**: Full Pydantic type safety for all dataset operations
- **Incremental**: Add samples one by one or in batches as they're generated
- **HuggingFace Compatible**: Automatic ImageFolder format with metadata.jsonl
- **Multiple Formats**: Support for JSONL, CSV, and Parquet metadata formats
- **Task-Specific**: Specialized workflows for different ML tasks
- **Validation**: Built-in dataset validation and consistency checking
- **Hub Integration**: Easy upload to HuggingFace Hub

## Quick Start

### Basic Usage

```python
from synthdoc import create_image_captioning_workflow, SplitType

# Create a workflow for image captioning
workflow = create_image_captioning_workflow(
    dataset_root="./datasets",
    workflow_name="my_captioning_project"
)

# Initialize a new dataset
workflow.initialize_dataset("caption_dataset_v1")

# Add samples incrementally
workflow.add_captioned_image(
    image_path="path/to/image.jpg",
    caption="A beautiful landscape with mountains",
    split=SplitType.TRAIN,
    source="my_generator"
)

# Finalize and get HuggingFace dataset
dataset = workflow.finalize_dataset()

# Upload to Hub
url = workflow.upload_to_hub("username/my-caption-dataset")
```

### Available Workflows

SynthDoc provides specialized workflows for different tasks:

```python
from synthdoc import (
    create_image_captioning_workflow,
    create_vqa_workflow,
    create_ocr_workflow,
    create_document_workflow,
    create_detection_workflow,
    create_classification_workflow
)

# Image Captioning
captioning_workflow = create_image_captioning_workflow("./datasets")

# Visual Question Answering
vqa_workflow = create_vqa_workflow("./datasets")

# OCR
ocr_workflow = create_ocr_workflow("./datasets")

# Document Understanding
document_workflow = create_document_workflow("./datasets")

# Object Detection
detection_workflow = create_detection_workflow("./datasets")

# Classification
classification_workflow = create_classification_workflow("./datasets")
```

## Task-Specific Examples

### Image Captioning

```python
workflow = create_image_captioning_workflow("./datasets")
workflow.initialize_dataset("my_captions")

workflow.add_captioned_image(
    image_path="image1.jpg",
    caption="A red car driving down the street",
    split=SplitType.TRAIN
)

dataset = workflow.finalize_dataset()
```

### Visual Question Answering

```python
workflow = create_vqa_workflow("./datasets")
workflow.initialize_dataset("my_vqa")

workflow.add_vqa_sample(
    image_path="document.png",
    question="What is the title of this document?",
    answer="Annual Report 2023",
    question_type="factual",
    difficulty="easy",
    split=SplitType.TRAIN
)

dataset = workflow.finalize_dataset()
```

### OCR with Annotations

```python
workflow = create_ocr_workflow("./datasets")
workflow.initialize_dataset("my_ocr")

workflow.add_ocr_sample(
    image_path="receipt.png",
    text="GROCERY STORE\\nTotal: $45.67",
    words=[
        {"text": "GROCERY", "bbox": [10, 20, 80, 35]},
        {"text": "STORE", "bbox": [90, 20, 140, 35]},
        {"text": "Total:", "bbox": [10, 50, 50, 65]},
        {"text": "$45.67", "bbox": [60, 50, 110, 65]},
    ],
    language="en",
    split=SplitType.TRAIN
)

dataset = workflow.finalize_dataset()
```

### Document Understanding

```python
workflow = create_document_workflow("./datasets")
workflow.initialize_dataset("my_documents")

workflow.add_document_sample(
    image_path="invoice.png",
    text="Invoice #12345\\nDate: 2023-12-01\\nAmount: $1,234.56",
    layout={
        "header": {"bbox": [0, 0, 800, 100]},
        "body": {"bbox": [0, 100, 800, 600]}
    },
    entities=[
        {"text": "12345", "label": "invoice_number", "bbox": [100, 20, 150, 35]},
        {"text": "2023-12-01", "label": "date", "bbox": [100, 50, 180, 65]}
    ],
    document_type="invoice",
    split=SplitType.TRAIN
)

dataset = workflow.finalize_dataset()
```

## Low-Level API

For more control, use the `DatasetManager` directly:

```python
from synthdoc import (
    create_dataset_manager,
    DatasetItem,
    create_image_captioning_metadata,
    SplitType,
    MetadataFormat
)

# Create manager with custom configuration
manager = create_dataset_manager(
    dataset_root="./datasets",
    dataset_name="custom_dataset",
    metadata_format=MetadataFormat.JSONL,
    copy_images=True,
    batch_size=100
)

# Create metadata
metadata = create_image_captioning_metadata(
    file_name="",  # Will be auto-generated
    caption="A sample caption",
    source="my_generator"
)

# Create dataset item
item = DatasetItem(
    image_path="path/to/image.jpg",
    metadata=metadata,
    split=SplitType.TRAIN
)

# Add to dataset
filename = manager.add_item(item)

# Load as HuggingFace dataset
dataset = manager.load_dataset()
```

## Configuration Options

### Metadata Formats

Choose from different metadata formats:

```python
from synthdoc import MetadataFormat

# JSONL (default) - one JSON object per line
manager = create_dataset_manager(
    dataset_root="./datasets",
    metadata_format=MetadataFormat.JSONL
)

# CSV - comma-separated values
manager = create_dataset_manager(
    dataset_root="./datasets",
    metadata_format=MetadataFormat.CSV
)

# Parquet - columnar format
manager = create_dataset_manager(
    dataset_root="./datasets",
    metadata_format=MetadataFormat.PARQUET
)
```

### Dataset Splits

Configure custom splits:

```python
from synthdoc import SplitType

custom_splits = [SplitType.TRAIN, SplitType.TEST, SplitType.VALIDATION]

manager = create_dataset_manager(
    dataset_root="./datasets",
    splits=custom_splits
)
```

## Dataset Structure

The created datasets follow HuggingFace ImageFolder format:

```
my_dataset/
├── train/
│   ├── 000001.png
│   ├── 000002.png
│   ├── 000003.png
│   └── metadata.jsonl
├── test/
│   ├── 000004.png
│   ├── 000005.png
│   └── metadata.jsonl
└── validation/
    ├── 000006.png
    ├── 000007.png
    └── metadata.jsonl
```

### Metadata Format Examples

**Image Captioning (metadata.jsonl):**
```jsonl
{"file_name": "000001.png", "text": "A beautiful sunset over mountains", "created_at": "2023-12-01T10:00:00", "source": "synthdoc"}
{"file_name": "000002.png", "text": "A busy city street", "created_at": "2023-12-01T10:01:00", "source": "synthdoc"}
```

**VQA (metadata.jsonl):**
```jsonl
{"file_name": "000001.png", "question": "What is shown in this image?", "answer": "A document", "question_type": "factual", "difficulty": "easy"}
{"file_name": "000002.png", "question": "How many tables are visible?", "answer": "3", "question_type": "counting", "difficulty": "medium"}
```

**Object Detection (metadata.jsonl):**
```jsonl
{"file_name": "000001.png", "objects": {"bbox": [[10, 20, 100, 50], [200, 300, 150, 80]], "categories": [0, 1]}}
{"file_name": "000002.png", "objects": {"bbox": [[50, 60, 120, 90]], "categories": [2]}}
```

## Uploading to HuggingFace Hub

### Using Workflow Upload

```python
# Upload with workflow
url = workflow.upload_to_hub(
    repo_id="username/my-dataset",
    private=False,
    commit_message="Initial dataset upload"
)
```

### Using Manager Upload

```python
from synthdoc import HubUploadConfig

upload_config = HubUploadConfig(
    repo_id="username/my-dataset",
    private=False,
    token="your_hf_token",  # Optional if logged in
    commit_message="Upload synthetic dataset",
    create_pr=False
)

url = manager.push_to_hub(upload_config)
```

## Validation

Datasets are automatically validated for consistency:

```python
# Get validation results
validation = manager.validate_dataset()

if validation.is_valid:
    print("Dataset is valid!")
else:
    print("Issues found:")
    for error in validation.errors:
        print(f"  Error: {error}")
    for warning in validation.warnings:
        print(f"  Warning: {warning}")
```

## Dataset Statistics

Track your dataset creation progress:

```python
# Get current statistics
stats = manager.get_stats()
print(f"Total items: {stats.total_items}")
print(f"Split breakdown: {stats.split_counts}")

# Get detailed summary
summary = manager.export_summary()
print(f"Dataset: {summary.dataset_name}")
print(f"Created: {summary.created_at}")
print(f"Valid: {summary.validation.is_valid}")
```

## Integration with Existing Workflows

The dataset management can be easily integrated into existing SynthDoc workflows:

```python
# Your existing generation code
def generate_documents():
    # Initialize dataset workflow
    workflow = create_image_captioning_workflow("./datasets")
    workflow.initialize_dataset("generated_docs")
    
    # Your document generation loop
    for i in range(100):
        # Generate document (your existing code)
        image_path = generate_document_image(i)
        caption = generate_caption(image_path)
        
        # Add to dataset
        workflow.add_captioned_image(
            image_path=image_path,
            caption=caption,
            split=SplitType.TRAIN
        )
    
    # Finalize and upload
    dataset = workflow.finalize_dataset()
    return dataset
```

## Error Handling

The system includes comprehensive error handling:

```python
try:
    workflow.add_captioned_image(
        image_path="nonexistent.jpg",
        caption="Test caption",
        split=SplitType.TRAIN
    )
except ValidationError as e:
    print(f"Validation error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Use appropriate workflows**: Choose the workflow that matches your task type
2. **Batch processing**: Use the auto-flush functionality for large datasets
3. **Validation**: Always validate your dataset before uploading
4. **Metadata consistency**: Use the type-safe metadata creation functions
5. **Split ratios**: Consider your use case when distributing data across splits
6. **Documentation**: Create dataset cards for better discoverability

## Advanced Usage

### Custom Metadata

```python
from synthdoc import BaseItemMetadata

# Create custom metadata
custom_metadata = BaseItemMetadata(
    file_name="custom.jpg",
    dataset_type=DatasetType.CUSTOM,
    custom={
        "my_custom_field": "custom_value",
        "another_field": 42
    }
)

item = DatasetItem(
    image_path="path/to/custom.jpg",
    metadata=custom_metadata,
    split=SplitType.TRAIN
)

manager.add_item(item)
```

### Batch Operations

```python
# Prepare multiple items
items = []
for i in range(100):
    metadata = create_image_captioning_metadata(
        file_name="",
        caption=f"Caption {i}",
        source="batch_generator"
    )
    items.append(DatasetItem(
        image_path=f"image_{i}.jpg",
        metadata=metadata,
        split=SplitType.TRAIN
    ))

# Add all at once
filenames = manager.add_batch(items)
```

This dataset management system makes it easy to create high-quality, properly formatted datasets that are ready for training and can be easily shared via HuggingFace Hub.
