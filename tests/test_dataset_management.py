"""
Tests for SynthDoc dataset management functionality.

This test suite verifies the type safety, functionality, and integration
of the dataset management features.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
import json

from synthdoc import (
    create_dataset_manager,
    create_image_captioning_workflow,
    create_vqa_workflow,
    create_ocr_workflow,
    DatasetItem,
    SplitType,
    DatasetType,
    MetadataFormat,
    HubUploadConfig,
    create_image_captioning_metadata,
    create_vqa_metadata,
    create_ocr_metadata,
    ValidationError,
)


class TestDatasetManager:
    """Test the core DatasetManager functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_image_path = self.temp_dir / "test_image.png"

        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(self.test_image_path)

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_dataset_manager_creation(self):
        """Test creating a dataset manager."""
        manager = create_dataset_manager(
            dataset_root=self.temp_dir, dataset_name="test_dataset"
        )

        assert manager.dataset_path.exists()
        assert manager.config.dataset_name == "test_dataset"
        assert len(manager.config.splits) == 3  # train, test, validation

    def test_add_single_item(self):
        """Test adding a single item to the dataset."""
        manager = create_dataset_manager(
            dataset_root=self.temp_dir, dataset_name="test_dataset"
        )

        metadata = create_image_captioning_metadata(
            file_name="", caption="Test caption", source="test"
        )

        item = DatasetItem(
            image_path=self.test_image_path, metadata=metadata, split=SplitType.TRAIN
        )

        filename = manager.add_item(item)

        assert filename.endswith(".png")
        assert (manager.dataset_path / "train" / filename).exists()

        # Check metadata file
        metadata_file = manager.dataset_path / "train" / "metadata.jsonl"
        assert metadata_file.exists()

        with open(metadata_file, "r") as f:
            metadata_entry = json.loads(f.read().strip())
            assert metadata_entry["file_name"] == filename
            assert metadata_entry["image_captioning"]["text"] == "Test caption"

    def test_batch_operations(self):
        """Test adding multiple items in a batch."""
        manager = create_dataset_manager(
            dataset_root=self.temp_dir, dataset_name="test_dataset"
        )

        items = []
        for i in range(5):
            metadata = create_image_captioning_metadata(
                file_name="", caption=f"Caption {i}", source="test"
            )

            items.append(
                DatasetItem(
                    image_path=self.test_image_path,
                    metadata=metadata,
                    split=SplitType.TRAIN,
                )
            )

        filenames = manager.add_batch(items)

        assert len(filenames) == 5
        stats = manager.get_stats()
        assert stats.split_counts["train"] == 5

    def test_validation(self):
        """Test dataset validation."""
        manager = create_dataset_manager(
            dataset_root=self.temp_dir, dataset_name="test_dataset"
        )

        # Add a valid item
        metadata = create_image_captioning_metadata(
            file_name="", caption="Test caption", source="test"
        )

        item = DatasetItem(
            image_path=self.test_image_path, metadata=metadata, split=SplitType.TRAIN
        )

        manager.add_item(item)

        # Validate
        validation = manager.validate_dataset()
        assert validation.is_valid
        assert len(validation.errors) == 0

    def test_different_metadata_formats(self):
        """Test support for different metadata formats."""
        for format_type in [MetadataFormat.JSONL, MetadataFormat.CSV]:
            manager = create_dataset_manager(
                dataset_root=self.temp_dir / format_type.value,
                dataset_name="test_dataset",
                metadata_format=format_type,
            )

            metadata = create_image_captioning_metadata(
                file_name="", caption="Test caption", source="test"
            )

            item = DatasetItem(
                image_path=self.test_image_path,
                metadata=metadata,
                split=SplitType.TRAIN,
            )

            filename = manager.add_item(item)

            # Check that the correct metadata file format was created
            expected_file = (
                manager.dataset_path / "train" / f"metadata.{format_type.value}"
            )
            assert expected_file.exists()


class TestWorkflows:
    """Test the workflow classes."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_image_path = self.temp_dir / "test_image.png"

        # Create a test image
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(self.test_image_path)

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_image_captioning_workflow(self):
        """Test the image captioning workflow."""
        workflow = create_image_captioning_workflow(
            dataset_root=self.temp_dir, workflow_name="test_captioning"
        )

        workflow.initialize_dataset("captioning_test")

        workflow.add_captioned_image(
            image_path=self.test_image_path,
            caption="A test image",
            split=SplitType.TRAIN,
            source="test",
        )

        workflow.add_captioned_image(
            image_path=self.test_image_path,
            caption="Another test image",
            split=SplitType.TEST,
            source="test",
        )

        # Check stats before finalization
        assert len(workflow.current_batch) == 0  # Should auto-flush

        dataset = workflow.finalize_dataset()

        assert "train" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) >= 1
        assert len(dataset["test"]) >= 1

    def test_vqa_workflow(self):
        """Test the VQA workflow."""
        workflow = create_vqa_workflow(
            dataset_root=self.temp_dir, workflow_name="test_vqa"
        )

        workflow.initialize_dataset("vqa_test")

        workflow.add_vqa_sample(
            image_path=self.test_image_path,
            question="What color is this image?",
            answer="Blue",
            question_type="factual",
            difficulty="easy",
            split=SplitType.TRAIN,
            source="test",
        )

        dataset = workflow.finalize_dataset()

        assert "train" in dataset
        assert len(dataset["train"]) >= 1

        # Check that VQA metadata is correctly structured
        sample = dataset["train"][0]
        assert "question" in sample
        assert "answer" in sample

    def test_ocr_workflow(self):
        """Test the OCR workflow."""
        workflow = create_ocr_workflow(
            dataset_root=self.temp_dir, workflow_name="test_ocr"
        )

        workflow.initialize_dataset("ocr_test")

        words = [
            {"text": "Hello", "bbox": [10, 10, 50, 25]},
            {"text": "World", "bbox": [60, 10, 100, 25]},
        ]

        workflow.add_ocr_sample(
            image_path=self.test_image_path,
            text="Hello World",
            words=words,
            language="en",
            split=SplitType.TRAIN,
            source="test",
        )

        dataset = workflow.finalize_dataset()

        assert "train" in dataset
        assert len(dataset["train"]) >= 1


class TestMetadataCreation:
    """Test metadata creation functions."""

    def test_image_captioning_metadata(self):
        """Test creating image captioning metadata."""
        metadata = create_image_captioning_metadata(
            file_name="test.jpg", caption="A test caption", source="test_source"
        )

        assert metadata.file_name == "test.jpg"
        assert metadata.dataset_type == DatasetType.IMAGE_CAPTIONING
        assert metadata.image_captioning.text == "A test caption"
        assert metadata.source == "test_source"

    def test_vqa_metadata(self):
        """Test creating VQA metadata."""
        metadata = create_vqa_metadata(
            file_name="test.jpg",
            question="What is this?",
            answer="A test",
            question_type="factual",
            difficulty="easy",
            source="test_source",
        )

        assert metadata.file_name == "test.jpg"
        assert metadata.dataset_type == DatasetType.VQA
        assert metadata.vqa.question == "What is this?"
        assert metadata.vqa.answer == "A test"
        assert metadata.vqa.question_type == "factual"
        assert metadata.vqa.difficulty == "easy"

    def test_ocr_metadata(self):
        """Test creating OCR metadata."""
        words = [{"text": "test", "bbox": [0, 0, 20, 15]}]

        metadata = create_ocr_metadata(
            file_name="test.jpg",
            text="test",
            words=words,
            language="en",
            source="test_source",
        )

        assert metadata.file_name == "test.jpg"
        assert metadata.dataset_type == DatasetType.OCR
        assert metadata.ocr.text == "test"
        assert metadata.ocr.words == words
        assert metadata.ocr.language == "en"


class TestTypeValidation:
    """Test Pydantic type validation."""

    def test_invalid_split_type(self):
        """Test that invalid split types are rejected."""
        with pytest.raises(ValueError):
            DatasetItem(
                image_path="test.jpg",
                metadata={"file_name": "test.jpg"},
                split="invalid_split",  # Should be SplitType enum
            )

    def test_invalid_bbox_format(self):
        """Test that invalid bounding box formats are rejected."""
        from synthdoc.models import ObjectDetectionMetadata

        with pytest.raises(ValidationError):
            ObjectDetectionMetadata(
                bbox=[[10, 20, 30]],  # Missing height
                categories=[0],
            )

    def test_bbox_category_mismatch(self):
        """Test that bbox and category count mismatches are caught."""
        from synthdoc.models import ObjectDetectionMetadata

        with pytest.raises(ValidationError):
            ObjectDetectionMetadata(
                bbox=[[10, 20, 30, 40], [50, 60, 70, 80]],  # 2 boxes
                categories=[0],  # 1 category
            )


class TestHubIntegration:
    """Test HuggingFace Hub integration."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_image_path = self.temp_dir / "test_image.png"

        # Create a test image
        img = Image.new("RGB", (100, 100), color="green")
        img.save(self.test_image_path)

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("synthdoc.dataset_manager.load_dataset")
    def test_hub_upload_config(self, mock_load_dataset):
        """Test HuggingFace Hub upload configuration."""
        # Mock the dataset loading
        mock_dataset = Mock()
        mock_dataset.push_to_hub = Mock(return_value="success")
        mock_load_dataset.return_value = mock_dataset

        manager = create_dataset_manager(
            dataset_root=self.temp_dir, dataset_name="test_dataset"
        )

        # Add a sample item
        metadata = create_image_captioning_metadata(
            file_name="", caption="Test caption", source="test"
        )

        item = DatasetItem(
            image_path=self.test_image_path, metadata=metadata, split=SplitType.TRAIN
        )

        manager.add_item(item)

        # Test upload configuration
        upload_config = HubUploadConfig(
            repo_id="test/test-dataset", private=True, commit_message="Test upload"
        )

        url = manager.push_to_hub(upload_config)

        # Verify the upload was called with correct parameters
        mock_dataset.push_to_hub.assert_called_once()
        call_args = mock_dataset.push_to_hub.call_args
        assert call_args[1]["repo_id"] == "test/test-dataset"
        assert call_args[1]["private"] == True
        assert call_args[1]["commit_message"] == "Test upload"


def test_full_workflow_integration():
    """Integration test for complete workflow."""
    temp_dir = Path(tempfile.mkdtemp())
    test_image_path = temp_dir / "test_image.png"

    try:
        # Create a test image
        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(test_image_path)

        # Create workflow
        workflow = create_image_captioning_workflow(
            dataset_root=temp_dir,
            workflow_name="integration_test",
            auto_flush_threshold=2,  # Small threshold for testing
        )

        workflow.initialize_dataset("integration_dataset")

        # Add multiple samples to test batching
        for i in range(5):
            workflow.add_captioned_image(
                image_path=test_image_path,
                caption=f"Caption {i}",
                split=SplitType.TRAIN if i < 3 else SplitType.TEST,
                source="integration_test",
            )

        # Finalize dataset
        dataset = workflow.finalize_dataset()

        # Verify dataset structure
        assert "train" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) == 3
        assert len(dataset["test"]) == 2

        # Verify dataset path exists
        assert workflow.manager.dataset_path.exists()

        # Verify metadata files exist
        train_metadata = workflow.manager.dataset_path / "train" / "metadata.jsonl"
        test_metadata = workflow.manager.dataset_path / "test" / "metadata.jsonl"
        assert train_metadata.exists()
        assert test_metadata.exists()

        # Verify validation passes
        validation = workflow.manager.validate_dataset()
        assert validation.is_valid

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
