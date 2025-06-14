"""
Incremental Dataset Manager for SynthDoc with full Pydantic type safety.

This module provides utilities for managing incremental dataset creation with
the HuggingFace ImageFolder structure, supporting continuous accumulation of
data and final dataset loading/uploading.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from datetime import datetime
import uuid

from datasets import load_dataset, DatasetDict
from pydantic import BaseModel, ValidationError
import pandas as pd

from .models import (
    DatasetConfig,
    DatasetItem,
    DatasetStats,
    ValidationResult,
    DatasetSummary,
    HubUploadConfig,
    BaseItemMetadata,
    MultiImageMetadata,
    SplitType,
    MetadataFormat,
    DatasetType,
)


class DatasetManager(BaseModel):
    """
    Type-safe dataset manager for incremental dataset creation.

    Features:
    - Full Pydantic type safety
    - Maintains proper HuggingFace ImageFolder structure
    - Handles metadata.jsonl files incrementally
    - Supports various metadata formats
    - Provides dataset loading and validation
    - Enables easy upload to HuggingFace Hub
    """

    config: DatasetConfig
    dataset_path: Path
    metadata_files: Dict[str, Path] = {}
    item_counters: Dict[str, int] = {}
    logger: Optional[logging.Logger] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: DatasetConfig, **kwargs):
        """Initialize the dataset manager with configuration."""
        # Resolve dataset path
        dataset_name = (
            config.dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        dataset_path = config.dataset_root / dataset_name

        super().__init__(config=config, dataset_path=dataset_path, **kwargs)

        # Setup logging
        self.logger = self._setup_logging()

        # Create directory structure
        if self.config.auto_create_splits:
            self._create_structure()

        self.logger.info(f"Initialized dataset manager: {self.dataset_path}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the dataset manager."""
        logger = logging.getLogger(f"DatasetManager_{self.config.dataset_name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_structure(self) -> None:
        """Create the dataset directory structure."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        for split in self.config.splits:
            split_path = self.dataset_path / split.value
            split_path.mkdir(exist_ok=True)

            # Initialize metadata file and counter
            metadata_filename = f"metadata.{self.config.metadata_format.value}"
            metadata_path = split_path / metadata_filename
            self.metadata_files[split.value] = metadata_path
            self.item_counters[split.value] = self._count_existing_items(split.value)

            self.logger.info(f"Created split directory: {split_path}")

    def _count_existing_items(self, split: str) -> int:
        """Count existing items in a split to maintain proper numbering."""
        metadata_path = self.metadata_files.get(split)

        if not metadata_path or not metadata_path.exists():
            return 0

        try:
            if self.config.metadata_format == MetadataFormat.JSONL:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            elif self.config.metadata_format == MetadataFormat.CSV:
                df = pd.read_csv(metadata_path)
                return len(df)
            elif self.config.metadata_format == MetadataFormat.PARQUET:
                df = pd.read_parquet(metadata_path)
                return len(df)
        except Exception as e:
            self.logger.warning(f"Error counting existing items in {split}: {e}")
            return 0

        return 0

    def add_item(self, item: DatasetItem) -> str:
        """
        Add a single item to the dataset with full type validation.

        Args:
            item: DatasetItem with image path, metadata, and split information

        Returns:
            The filename used for the image in the dataset
        """
        # Validate split
        if item.split.value not in [s.value for s in self.config.splits]:
            raise ValueError(
                f"Invalid split '{item.split}'. Available splits: {self.config.splits}"
            )

        # Generate filename
        if item.custom_filename:
            filename = item.custom_filename
        else:
            ext = Path(item.image_path).suffix
            filename = f"{self.item_counters[item.split.value]:06d}{ext}"

        split_path = self.dataset_path / item.split.value
        target_image_path = split_path / filename

        # Handle image copying/linking
        self._handle_image_copy(Path(item.image_path), target_image_path)

        # Prepare and validate metadata
        metadata_entry = self._prepare_metadata_entry(filename, item.metadata)

        # Append to metadata file
        self._append_metadata(item.split.value, metadata_entry)

        # Update counter
        self.item_counters[item.split.value] += 1

        self.logger.debug(f"Added item to {item.split.value}: {filename}")
        return filename

    def _handle_image_copy(self, source_path: Path, target_path: Path) -> None:
        """Handle image copying or symlinking based on configuration."""
        if self.config.copy_images:
            shutil.copy2(source_path, target_path)
        else:
            # Create a symbolic link or fall back to copying
            if not target_path.exists():
                try:
                    target_path.symlink_to(source_path.absolute())
                except OSError:
                    # Fall back to copying if symlink fails
                    shutil.copy2(source_path, target_path)

    def _prepare_metadata_entry(
        self,
        filename: str,
        metadata: Union[BaseItemMetadata, MultiImageMetadata, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepare metadata entry with proper validation."""
        if isinstance(metadata, BaseModel):
            # Convert Pydantic model to dict
            metadata_dict = metadata.model_dump(exclude_none=True)
            if isinstance(metadata, BaseItemMetadata):
                metadata_dict["file_name"] = filename
            return metadata_dict
        elif isinstance(metadata, dict):
            # Ensure file_name is set for dict metadata
            metadata_copy = metadata.copy()
            if "file_name" not in metadata_copy:
                metadata_copy["file_name"] = filename
            return metadata_copy
        else:
            raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def _append_metadata(self, split: str, metadata_entry: Dict[str, Any]) -> None:
        """Append metadata entry to the appropriate file."""
        metadata_path = self.metadata_files[split]

        if self.config.metadata_format == MetadataFormat.JSONL:
            with open(metadata_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(metadata_entry, ensure_ascii=False, default=str) + "\n"
                )
        elif self.config.metadata_format == MetadataFormat.CSV:
            # Convert to DataFrame and append
            df = pd.DataFrame([metadata_entry])
            if metadata_path.exists():
                df.to_csv(metadata_path, mode="a", header=False, index=False)
            else:
                df.to_csv(metadata_path, index=False)
        elif self.config.metadata_format == MetadataFormat.PARQUET:
            # For parquet, we need to read existing data and append
            if metadata_path.exists():
                existing_df = pd.read_parquet(metadata_path)
                new_df = pd.DataFrame([metadata_entry])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = pd.DataFrame([metadata_entry])
            combined_df.to_parquet(metadata_path, index=False)

    def add_batch(self, items: List[DatasetItem]) -> List[str]:
        """
        Add a batch of items to the dataset with type validation.

        Args:
            items: List of DatasetItem objects

        Returns:
            List of filenames used in the dataset
        """
        filenames = []

        for item in items:
            try:
                filename = self.add_item(item)
                filenames.append(filename)
            except ValidationError as e:
                self.logger.warning(f"Skipping invalid item: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Error adding item: {e}")
                continue

        self.logger.info(f"Added batch of {len(filenames)} items")
        return filenames

    def get_stats(self) -> DatasetStats:
        """Get current statistics for all splits."""
        split_counts = {
            split: self.item_counters.get(split, 0)
            for split in [s.value for s in self.config.splits]
        }
        total_items = sum(split_counts.values())

        return DatasetStats(split_counts=split_counts, total_items=total_items)

    def validate_dataset(self) -> ValidationResult:
        """
        Validate the dataset structure and consistency.

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        stats = {}

        for split in self.config.splits:
            split_value = split.value
            split_path = self.dataset_path / split_value
            metadata_path = self.metadata_files.get(split_value)

            # Check if directories exist
            if not split_path.exists():
                errors.append(f"Split directory missing: {split_value}")
                continue

            # Check metadata file
            if not metadata_path or not metadata_path.exists():
                warnings.append(f"No metadata file for split: {split_value}")
                continue

            # Validate metadata entries
            try:
                entries = self._read_metadata_entries(split_value)
                missing_images = []

                for entry in entries:
                    if isinstance(entry, dict) and "file_name" not in entry:
                        errors.append(
                            f"Metadata entry missing 'file_name' in {split_value}"
                        )
                        continue

                    file_name = (
                        entry.get("file_name")
                        if isinstance(entry, dict)
                        else getattr(entry, "file_name", None)
                    )
                    if file_name:
                        image_path = split_path / file_name
                        if not image_path.exists():
                            missing_images.append(file_name)

                if missing_images:
                    warnings.append(
                        f"Missing images in {split_value}: {len(missing_images)} files"
                    )

                stats[split_value] = {
                    "metadata_entries": len(entries),
                    "missing_images": len(missing_images),
                }

            except Exception as e:
                errors.append(f"Error reading metadata for {split_value}: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, stats=stats
        )

    def _read_metadata_entries(self, split: str) -> List[Dict[str, Any]]:
        """Read metadata entries from file."""
        metadata_path = self.metadata_files[split]

        if self.config.metadata_format == MetadataFormat.JSONL:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif self.config.metadata_format == MetadataFormat.CSV:
            df = pd.read_csv(metadata_path)
            return df.to_dict("records")
        elif self.config.metadata_format == MetadataFormat.PARQUET:
            df = pd.read_parquet(metadata_path)
            return df.to_dict("records")

        return []

    def load_dataset(self, **kwargs) -> DatasetDict:
        """
        Load the dataset using HuggingFace datasets library.

        Args:
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            DatasetDict containing all splits
        """
        self.logger.info(f"Loading dataset from: {self.dataset_path}")

        # Validate before loading
        validation = self.validate_dataset()
        if not validation.is_valid:
            self.logger.warning("Dataset validation found errors:")
            for error in validation.errors:
                self.logger.warning(f"  - {error}")

        try:
            dataset = load_dataset(
                "imagefolder", data_dir=str(self.dataset_path), **kwargs
            )

            self.logger.info("Dataset loaded successfully")
            self.logger.info(f"Dataset splits: {list(dataset.keys())}")
            for split, data in dataset.items():
                self.logger.info(f"  {split}: {len(data)} samples")

            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def push_to_hub(self, upload_config: HubUploadConfig) -> str:
        """
        Push the dataset to HuggingFace Hub with type-safe configuration.

        Args:
            upload_config: HubUploadConfig with all upload parameters

        Returns:
            The repository URL
        """
        # Load dataset first
        dataset = self.load_dataset()

        # Default commit message
        commit_message = upload_config.commit_message
        if commit_message is None:
            stats = self.get_stats()
            commit_message = f"Upload dataset with {stats.total_items} items across {len(stats.split_counts)} splits"

        self.logger.info(f"Pushing dataset to Hub: {upload_config.repo_id}")

        try:
            result = dataset.push_to_hub(
                repo_id=upload_config.repo_id,
                private=upload_config.private,
                token=upload_config.token,
                commit_message=commit_message,
                create_pr=upload_config.create_pr,
            )

            self.logger.info(f"Successfully pushed dataset to: {upload_config.repo_id}")
            return f"https://huggingface.co/datasets/{upload_config.repo_id}"

        except Exception as e:
            self.logger.error(f"Error pushing to Hub: {e}")
            raise

    def export_summary(self) -> DatasetSummary:
        """Export a comprehensive summary of the dataset."""
        stats = self.get_stats()
        validation = self.validate_dataset()

        return DatasetSummary(
            dataset_name=self.config.dataset_name or self.dataset_path.name,
            dataset_path=self.dataset_path,
            created_at=datetime.now(),
            splits=[s.value for s in self.config.splits],
            stats=stats,
            validation=validation,
        )

    def create_dataset_card(self, output_path: Optional[Path] = None) -> Path:
        """
        Create a comprehensive dataset card (README.md) for the dataset.

        Args:
            output_path: Path to save the README.md file

        Returns:
            Path to the created dataset card
        """
        if output_path is None:
            output_path = self.dataset_path / "README.md"

        summary = self.export_summary()

        readme_content = f"""# {summary.dataset_name}

## Dataset Summary

This dataset was created using SynthDoc's Incremental Dataset Manager and follows the HuggingFace ImageFolder format.

- **Created:** {summary.created_at.isoformat()}
- **Total Items:** {summary.stats.total_items:,}
- **Splits:** {", ".join(summary.splits)}
- **Metadata Format:** {self.config.metadata_format.value}

## Dataset Structure

```
{summary.dataset_name}/
"""

        for split in summary.splits:
            readme_content += f"""├── {split}/
│   ├── metadata.{self.config.metadata_format.value}
│   └── images...
"""

        readme_content += """```

## Split Statistics

"""

        for split, count in summary.stats.split_counts.items():
            readme_content += f"- **{split}:** {count:,} items\n"

        readme_content += f"""

## Usage

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("imagefolder", data_dir="{self.dataset_path}")

# Load specific split
train_dataset = load_dataset("imagefolder", data_dir="{self.dataset_path}", split="train")

# Access data
print(dataset["train"][0])  # First item from train split
```

## Dataset Creation

This dataset was created using SynthDoc's type-safe dataset management system with the following configuration:

- **Metadata Format:** {self.config.metadata_format.value}
- **Image Copying:** {"Enabled" if self.config.copy_images else "Symlinks"}
- **Batch Size:** {self.config.batch_size}

## Validation Results

"""

        if summary.validation.is_valid:
            readme_content += "✅ Dataset validation passed successfully.\n\n"
        else:
            readme_content += "⚠️ Dataset validation found issues:\n\n"
            for error in summary.validation.errors:
                readme_content += f"- **Error:** {error}\n"
            for warning in summary.validation.warnings:
                readme_content += f"- **Warning:** {warning}\n"
            readme_content += "\n"

        readme_content += (
            """## Citation

If you use this dataset, please cite:

```bibtex
@dataset{synthdoc_dataset,
    title={Synthetic Document Dataset},
    author={SynthDoc},
    year={"""
            + str(summary.created_at.year)
            + """},
    url={https://github.com/your-repo/synthdoc}
}
```
"""
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        self.logger.info(f"Dataset card created: {output_path}")
        return output_path


# Factory functions for easier dataset creation


def create_dataset_manager(
    dataset_root: Union[str, Path],
    dataset_name: Optional[str] = None,
    splits: Optional[List[SplitType]] = None,
    metadata_format: MetadataFormat = MetadataFormat.JSONL,
    copy_images: bool = True,
    batch_size: int = 100,
) -> DatasetManager:
    """
    Factory function to create a DatasetManager with sensible defaults.

    Args:
        dataset_root: Root directory for datasets
        dataset_name: Name of the dataset (auto-generated if None)
        splits: List of splits to create
        metadata_format: Format for metadata files
        copy_images: Whether to copy images or create symlinks
        batch_size: Batch size for processing

    Returns:
        Configured DatasetManager instance
    """
    config = DatasetConfig(
        dataset_root=Path(dataset_root),
        dataset_name=dataset_name,
        splits=splits or [SplitType.TRAIN, SplitType.TEST, SplitType.VALIDATION],
        metadata_format=metadata_format,
        copy_images=copy_images,
        batch_size=batch_size,
    )

    return DatasetManager(config=config)
