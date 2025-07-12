#!/usr/bin/env python3
"""
Simple test for document translation workflow using SynthDoc API.
"""

from synthdoc import SynthDoc

# Initialize SynthDoc (automatically loads from .env file)
synth = SynthDoc()

# Translate documents using the simple API
dataset = synth.translate_documents(
    input_images=["image.jpeg"],
    target_languages=["hi"],
    # Optional parameters (will use sensible defaults):
    # yolo_model_path="model-doclayout-yolo.pt",
    # font_path="synthdoc/fonts/",
    # confidence_threshold=0.4,
    # image_size=1024
)

print(f"✅ Created {len(dataset)} translated samples")
print(f"📦 Dataset features: {list(dataset.features.keys())}")

# Check if this is an error dataset
if len(dataset) > 0 and 'error' in dataset.features:
    # This is an error fallback dataset
    sample = dataset[0]
    print(f"❌ Translation failed:")
    print(f"   - Error: {sample['error']}")
    print(f"   - Note: {sample['note']}")
elif len(dataset) > 0:
    # This is a successful translation dataset
    sample = dataset[0]
    print(f"📄 Sample info:")
    print(f"   - ID: {sample['id']}")
    print(f"   - Target language: {sample['target_language']}")
    print(f"   - Font used: {sample['font_used']}")
    print(f"   - Regions detected: {sample['num_regions']}")
    print(f"   - Image size: {sample['image'].size}")
else:
    print("ℹ️  No samples in dataset")