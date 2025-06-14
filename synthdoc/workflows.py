"""
Workflow integration for SynthDoc dataset management.

This module provides high-level interfaces for integrating the dataset manager
into existing workflows, with specialized support for different task types.
"""

from typing import Dict, List, Optional, Union, Any, Type, Callable
from pathlib import Path
from datetime import datetime
import logging

from pydantic import BaseModel, Field
from datasets import DatasetDict

from .dataset_manager import DatasetManager, create_dataset_manager
from .models import (
    DatasetItem,
    DatasetConfig,
    SplitType,
    DatasetType,
    MetadataFormat,
    HubUploadConfig,
    create_image_captioning_metadata,
    create_object_detection_metadata,
    create_vqa_metadata,
    create_classification_metadata,
    create_ocr_metadata,
    create_document_metadata,
    BaseItemMetadata,
)


class WorkflowConfig(BaseModel):
    """Configuration for workflow-based dataset building."""

    workflow_name: str = Field(..., description="Name of the workflow")
    dataset_root: Path = Field(..., description="Root directory for datasets")
    dataset_type: DatasetType = Field(..., description="Primary type of dataset")
    auto_flush_threshold: int = Field(default=100, description="Auto-flush batch size")
    metadata_format: MetadataFormat = Field(
        default=MetadataFormat.JSONL, description="Metadata file format"
    )
    copy_images: bool = Field(default=True, description="Whether to copy images")

    class Config:
        arbitrary_types_allowed = True


class WorkflowDatasetBuilder(BaseModel):
    """
    High-level dataset builder for workflow integration.

    This class provides a simplified interface for incrementally building datasets
    during workflow execution, with automatic batching and type safety.
    """

    config: WorkflowConfig
    manager: Optional[DatasetManager] = None
    current_batch: List[DatasetItem] = Field(default_factory=list)
    logger: Optional[logging.Logger] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: WorkflowConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the workflow builder."""
        logger = logging.getLogger(f"WorkflowBuilder_{self.config.workflow_name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def initialize_dataset(self, dataset_name: Optional[str] = None) -> DatasetManager:
        """Initialize a new dataset for the workflow."""
        if dataset_name is None:
            dataset_name = f"{self.config.workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.manager = create_dataset_manager(
            dataset_root=self.config.dataset_root,
            dataset_name=dataset_name,
            metadata_format=self.config.metadata_format,
            copy_images=self.config.copy_images,
            batch_size=self.config.auto_flush_threshold,
        )

        self.logger.info(f"Initialized dataset: {dataset_name}")
        return self.manager

    def add_sample(
        self,
        image_path: Union[str, Path],
        metadata: Union[BaseItemMetadata, Dict[str, Any]],
        split: SplitType = SplitType.TRAIN,
        custom_filename: Optional[str] = None,
    ) -> None:
        """Add a single sample to the current batch."""
        if not self.manager:
            raise RuntimeError(
                "Dataset not initialized. Call initialize_dataset() first."
            )

        item = DatasetItem(
            image_path=image_path,
            metadata=metadata,
            split=split,
            custom_filename=custom_filename,
        )

        self.current_batch.append(item)

        # Auto-flush when batch is full
        if len(self.current_batch) >= self.config.auto_flush_threshold:
            self.flush_batch()

    def flush_batch(self) -> None:
        """Flush the current batch to the dataset."""
        if not self.current_batch or not self.manager:
            return

        filenames = self.manager.add_batch(self.current_batch)
        self.logger.info(f"Flushed batch of {len(filenames)} samples")
        self.current_batch.clear()

    def finalize_dataset(self, create_card: bool = True) -> DatasetDict:
        """Finalize the dataset and return it."""
        if not self.manager:
            raise RuntimeError("Dataset not initialized")

        # Flush any remaining samples
        self.flush_batch()

        # Create dataset card if requested
        if create_card:
            self.manager.create_dataset_card()

        # Load and return the dataset
        return self.manager.load_dataset()

    def upload_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """Upload the finalized dataset to HuggingFace Hub."""
        if not self.manager:
            raise RuntimeError("Dataset not initialized")

        upload_config = HubUploadConfig(
            repo_id=repo_id, private=private, token=token, commit_message=commit_message
        )

        return self.manager.push_to_hub(upload_config)


class ImageCaptioningWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for image captioning tasks."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        workflow_name: str = "image_captioning",
        **kwargs,
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.IMAGE_CAPTIONING,
            **kwargs,
        )
        super().__init__(config=config)

    def add_captioned_image(
        self,
        image_path: Union[str, Path],
        caption: str,
        split: SplitType = SplitType.TRAIN,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add an image with caption to the dataset."""
        metadata = create_image_captioning_metadata(
            file_name="",  # Will be set by the manager
            caption=caption,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


class VQAWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for VQA tasks."""

    def __init__(
        self, dataset_root: Union[str, Path], workflow_name: str = "vqa", **kwargs
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.VQA,
            **kwargs,
        )
        super().__init__(config=config)

    def add_vqa_sample(
        self,
        image_path: Union[str, Path],
        question: str,
        answer: str,
        split: SplitType = SplitType.TRAIN,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add a VQA sample to the dataset."""
        metadata = create_vqa_metadata(
            file_name="",  # Will be set by the manager
            question=question,
            answer=answer,
            question_type=question_type,
            difficulty=difficulty,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


class OCRWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for OCR tasks."""

    def __init__(
        self, dataset_root: Union[str, Path], workflow_name: str = "ocr", **kwargs
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.OCR,
            **kwargs,
        )
        super().__init__(config=config)

    def add_ocr_sample(
        self,
        image_path: Union[str, Path],
        text: str,
        split: SplitType = SplitType.TRAIN,
        words: Optional[List[Dict[str, Any]]] = None,
        lines: Optional[List[Dict[str, Any]]] = None,
        language: Optional[str] = None,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add an OCR sample to the dataset."""
        metadata = create_ocr_metadata(
            file_name="",  # Will be set by the manager
            text=text,
            words=words,
            lines=lines,
            language=language,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


class DocumentUnderstandingWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for document understanding tasks."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        workflow_name: str = "document_understanding",
        **kwargs,
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.DOCUMENT_UNDERSTANDING,
            **kwargs,
        )
        super().__init__(config=config)

    def add_document_sample(
        self,
        image_path: Union[str, Path],
        split: SplitType = SplitType.TRAIN,
        text: Optional[str] = None,
        layout: Optional[Dict[str, Any]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        document_type: Optional[str] = None,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add a document understanding sample to the dataset."""
        metadata = create_document_metadata(
            file_name="",  # Will be set by the manager
            text=text,
            layout=layout,
            entities=entities,
            document_type=document_type,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


class ObjectDetectionWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for object detection tasks."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        workflow_name: str = "object_detection",
        **kwargs,
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.OBJECT_DETECTION,
            **kwargs,
        )
        super().__init__(config=config)

    def add_detection_sample(
        self,
        image_path: Union[str, Path],
        bboxes: List[List[float]],
        categories: List[int],
        split: SplitType = SplitType.TRAIN,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add an object detection sample to the dataset."""
        metadata = create_object_detection_metadata(
            file_name="",  # Will be set by the manager
            bboxes=bboxes,
            categories=categories,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


class ClassificationWorkflow(WorkflowDatasetBuilder):
    """Specialized workflow builder for classification tasks."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        workflow_name: str = "classification",
        **kwargs,
    ):
        config = WorkflowConfig(
            workflow_name=workflow_name,
            dataset_root=Path(dataset_root),
            dataset_type=DatasetType.CLASSIFICATION,
            **kwargs,
        )
        super().__init__(config=config)

    def add_classified_image(
        self,
        image_path: Union[str, Path],
        label: Union[str, int],
        split: SplitType = SplitType.TRAIN,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        **metadata_kwargs,
    ) -> None:
        """Add a classified image to the dataset."""
        metadata = create_classification_metadata(
            file_name="",  # Will be set by the manager
            label=label,
            confidence=confidence,
            source=source,
            **metadata_kwargs,
        )
        self.add_sample(image_path, metadata, split)


# Factory functions for easy workflow creation


def create_image_captioning_workflow(
    dataset_root: Union[str, Path], workflow_name: str = "image_captioning", **kwargs
) -> ImageCaptioningWorkflow:
    """Create an image captioning workflow."""
    return ImageCaptioningWorkflow(
        dataset_root=dataset_root, workflow_name=workflow_name, **kwargs
    )


def create_vqa_workflow(
    dataset_root: Union[str, Path], workflow_name: str = "vqa", **kwargs
) -> VQAWorkflow:
    """Create a VQA workflow."""
    return VQAWorkflow(dataset_root=dataset_root, workflow_name=workflow_name, **kwargs)


def create_ocr_workflow(
    dataset_root: Union[str, Path], workflow_name: str = "ocr", **kwargs
) -> OCRWorkflow:
    """Create an OCR workflow."""
    return OCRWorkflow(dataset_root=dataset_root, workflow_name=workflow_name, **kwargs)


def create_document_workflow(
    dataset_root: Union[str, Path],
    workflow_name: str = "document_understanding",
    **kwargs,
) -> DocumentUnderstandingWorkflow:
    """Create a document understanding workflow."""
    return DocumentUnderstandingWorkflow(
        dataset_root=dataset_root, workflow_name=workflow_name, **kwargs
    )


def create_detection_workflow(
    dataset_root: Union[str, Path], workflow_name: str = "object_detection", **kwargs
) -> ObjectDetectionWorkflow:
    """Create an object detection workflow."""
    return ObjectDetectionWorkflow(
        dataset_root=dataset_root, workflow_name=workflow_name, **kwargs
    )


def create_classification_workflow(
    dataset_root: Union[str, Path], workflow_name: str = "classification", **kwargs
) -> ClassificationWorkflow:
    """Create a classification workflow."""
    return ClassificationWorkflow(
        dataset_root=dataset_root, workflow_name=workflow_name, **kwargs
    )
