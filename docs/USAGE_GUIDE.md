# SynthDoc Workflow Usage Guide

This guide shows how to use SynthDoc workflows for document generation and processing.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env with your API keys
```

## Working Examples

For up-to-date and working examples, please refer to the `examples/` directory in the repository:

- `examples/workflow_examples.py` - Complete workflow examples
- `examples/basic_usage.py` - Basic usage patterns
- `examples/integration_example.py` - Integration examples
- `examples/vqa_pipeline.py` - VQA generation examples

## Output Structure

All workflows create standardized output:

```
output_directory/
├── images/
│   ├── generated_image_1.png
│   ├── generated_image_2.png
│   └── ...
└── metadata.jsonl
```

## Loading Results

Generated datasets can be loaded with HuggingFace datasets:

```python
from datasets import load_dataset

# Load from output directory
dataset = load_dataset("imagefolder", data_dir="output_directory")

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
- For detailed usage examples, see the `examples/` directory 