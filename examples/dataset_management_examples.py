"""
Examples demonstrating SynthDoc's incremental dataset management features.

This file shows how to integrate the dataset manager into various workflows
and demonstrates type-safe dataset creation.
"""

from pathlib import Path
from synthdoc import (
    create_image_captioning_workflow,
    create_vqa_workflow,
    create_ocr_workflow,
    create_document_workflow,
    create_dataset_manager,
    SplitType,
    DatasetType,
    MetadataFormat,
    HubUploadConfig,
    create_image_captioning_metadata,
    create_vqa_metadata,
    DatasetItem,
)


def example_image_captioning_workflow():
    """Example: Creating an image captioning dataset incrementally."""

    # Initialize workflow
    workflow = create_image_captioning_workflow(
        dataset_root="./datasets",
        workflow_name="image_captioning_demo",
        auto_flush_threshold=50,  # Flush every 50 samples
    )

    # Initialize dataset
    workflow.initialize_dataset("my_captioning_dataset")

    # Add samples incrementally (simulating a workflow)
    sample_data = [
        ("image1.jpg", "A beautiful sunset over the mountains"),
        ("image2.jpg", "A cat sitting on a windowsill"),
        ("image3.jpg", "A busy street scene with cars and pedestrians"),
    ]

    for image_path, caption in sample_data:
        workflow.add_captioned_image(
            image_path=image_path,
            caption=caption,
            split=SplitType.TRAIN,
            source="synthetic_generation",
        )

    # Add some test samples
    test_data = [
        ("test1.jpg", "A mountain landscape"),
        ("test2.jpg", "An urban scene"),
    ]

    for image_path, caption in test_data:
        workflow.add_captioned_image(
            image_path=image_path,
            caption=caption,
            split=SplitType.TEST,
            source="synthetic_generation",
        )

    # Finalize and get the dataset
    dataset = workflow.finalize_dataset()

    print(f"Created dataset with {len(dataset['train'])} training samples")
    print(f"and {len(dataset['test'])} test samples")

    # Optional: Upload to Hub
    # url = workflow.upload_to_hub("username/my-captioning-dataset")
    # print(f"Uploaded to: {url}")

    return dataset


def example_vqa_workflow():
    """Example: Creating a VQA dataset incrementally."""

    workflow = create_vqa_workflow(dataset_root="./datasets", workflow_name="vqa_demo")

    workflow.initialize_dataset("my_vqa_dataset")

    # Add VQA samples
    vqa_samples = [
        (
            "doc1.png",
            "What is the title of this document?",
            "Annual Report 2023",
            "factual",
            "easy",
        ),
        ("doc2.png", "How many tables are in this page?", "3", "counting", "medium"),
        (
            "doc3.png",
            "What can you infer about the company's performance?",
            "The company shows strong growth",
            "reasoning",
            "hard",
        ),
    ]

    for image_path, question, answer, q_type, difficulty in vqa_samples:
        workflow.add_vqa_sample(
            image_path=image_path,
            question=question,
            answer=answer,
            question_type=q_type,
            difficulty=difficulty,
            split=SplitType.TRAIN,
        )

    dataset = workflow.finalize_dataset()
    return dataset


def example_ocr_workflow():
    """Example: Creating an OCR dataset incrementally."""

    workflow = create_ocr_workflow(dataset_root="./datasets", workflow_name="ocr_demo")

    workflow.initialize_dataset("my_ocr_dataset")

    # Add OCR samples with detailed annotations
    ocr_samples = [
        {
            "image": "receipt1.png",
            "text": "GROCERY STORE\nTotal: $45.67\nThank you!",
            "words": [
                {"text": "GROCERY", "bbox": [10, 20, 80, 35]},
                {"text": "STORE", "bbox": [90, 20, 140, 35]},
                {"text": "Total:", "bbox": [10, 50, 50, 65]},
                {"text": "$45.67", "bbox": [60, 50, 110, 65]},
                {"text": "Thank", "bbox": [10, 80, 50, 95]},
                {"text": "you!", "bbox": [60, 80, 90, 95]},
            ],
            "language": "en",
        }
    ]

    for sample in ocr_samples:
        workflow.add_ocr_sample(
            image_path=sample["image"],
            text=sample["text"],
            words=sample["words"],
            language=sample["language"],
            split=SplitType.TRAIN,
        )

    dataset = workflow.finalize_dataset()
    return dataset


def example_document_understanding_workflow():
    """Example: Creating a document understanding dataset."""

    workflow = create_document_workflow(
        dataset_root="./datasets", workflow_name="document_understanding_demo"
    )

    workflow.initialize_dataset("my_document_dataset")

    # Add document samples with layout and entity information
    document_samples = [
        {
            "image": "invoice1.png",
            "text": "Invoice #12345\nDate: 2023-12-01\nAmount: $1,234.56",
            "layout": {
                "header": {"bbox": [0, 0, 800, 100]},
                "body": {"bbox": [0, 100, 800, 600]},
                "footer": {"bbox": [0, 600, 800, 700]},
            },
            "entities": [
                {
                    "text": "12345",
                    "label": "invoice_number",
                    "bbox": [100, 20, 150, 35],
                },
                {"text": "2023-12-01", "label": "date", "bbox": [100, 50, 180, 65]},
                {"text": "$1,234.56", "label": "amount", "bbox": [100, 80, 170, 95]},
            ],
            "document_type": "invoice",
        }
    ]

    for sample in document_samples:
        workflow.add_document_sample(
            image_path=sample["image"],
            text=sample["text"],
            layout=sample["layout"],
            entities=sample["entities"],
            document_type=sample["document_type"],
            split=SplitType.TRAIN,
        )

    dataset = workflow.finalize_dataset()
    return dataset


def example_low_level_dataset_manager():
    """Example: Using the low-level DatasetManager directly."""

    # Create dataset manager with custom configuration
    manager = create_dataset_manager(
        dataset_root="./datasets",
        dataset_name="custom_dataset",
        metadata_format=MetadataFormat.JSONL,
        copy_images=True,
        batch_size=100,
    )

    # Create custom metadata
    items = []
    for i in range(5):
        metadata = create_image_captioning_metadata(
            file_name=f"image_{i:03d}.jpg",
            caption=f"This is a caption for image {i}",
            source="custom_generator",
        )

        item = DatasetItem(
            image_path=f"path/to/image_{i:03d}.jpg",
            metadata=metadata,
            split=SplitType.TRAIN,
        )
        items.append(item)

    # Add batch of items
    filenames = manager.add_batch(items)
    print(f"Added {len(filenames)} items to dataset")

    # Get statistics
    stats = manager.get_stats()
    print(f"Dataset stats: {stats}")

    # Validate dataset
    validation = manager.validate_dataset()
    print(f"Dataset valid: {validation.is_valid}")

    # Load as HuggingFace dataset
    dataset = manager.load_dataset()

    # Upload to Hub
    upload_config = HubUploadConfig(
        repo_id="username/my-custom-dataset",
        private=False,
        commit_message="Initial upload of custom dataset",
    )
    # url = manager.push_to_hub(upload_config)

    return dataset


def example_mixed_dataset_types():
    """Example: Creating a dataset with mixed task types."""

    manager = create_dataset_manager(
        dataset_root="./datasets", dataset_name="mixed_tasks_dataset"
    )

    # Mix different types of metadata in one dataset
    items = []

    # Add image captioning samples
    for i in range(3):
        metadata = create_image_captioning_metadata(
            file_name=f"caption_{i:03d}.jpg",
            caption=f"Caption for image {i}",
            source="captioning_workflow",
        )
        items.append(
            DatasetItem(
                image_path=f"path/to/caption_{i:03d}.jpg",
                metadata=metadata,
                split=SplitType.TRAIN,
            )
        )

    # Add VQA samples
    for i in range(3):
        metadata = create_vqa_metadata(
            file_name=f"vqa_{i:03d}.jpg",
            question=f"What is shown in image {i}?",
            answer=f"Answer for image {i}",
            question_type="factual",
            source="vqa_workflow",
        )
        items.append(
            DatasetItem(
                image_path=f"path/to/vqa_{i:03d}.jpg",
                metadata=metadata,
                split=SplitType.TRAIN,
            )
        )

    # Add all items
    manager.add_batch(items)

    # Load and return dataset
    dataset = manager.load_dataset()
    return dataset


def main():
    """Run all examples."""
    print("=== Image Captioning Workflow ===")
    # dataset1 = example_image_captioning_workflow()

    print("\n=== VQA Workflow ===")
    # dataset2 = example_vqa_workflow()

    print("\n=== OCR Workflow ===")
    # dataset3 = example_ocr_workflow()

    print("\n=== Document Understanding Workflow ===")
    # dataset4 = example_document_understanding_workflow()

    print("\n=== Low-level Dataset Manager ===")
    # dataset5 = example_low_level_dataset_manager()

    print("\n=== Mixed Dataset Types ===")
    # dataset6 = example_mixed_dataset_types()

    print("\nAll examples completed successfully!")
    print("Note: Examples are commented out to avoid file system operations.")
    print("Uncomment the lines above to run actual dataset creation.")


if __name__ == "__main__":
    main()
