"""
Integration guide for adding dataset management to existing SynthDoc workflows.

This file demonstrates how to modify existing SynthDoc generators to automatically
create and manage datasets in HuggingFace ImageFolder format.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

from synthdoc import (
    SynthDoc,
    create_image_captioning_workflow,
    create_vqa_workflow,
    create_ocr_workflow,
    SplitType,
)


class DatasetEnabledSynthDoc(SynthDoc):
    """
    Extended SynthDoc class with built-in dataset management.

    This class wraps the original SynthDoc functionality and automatically
    manages dataset creation for all generated content.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        auto_create_datasets: bool = True,
        **kwargs,
    ):
        """
        Initialize DatasetEnabledSynthDoc.

        Args:
            output_dir: Directory for temporary files
            dataset_root: Root directory for dataset creation
            auto_create_datasets: Whether to automatically create datasets
            **kwargs: Arguments passed to parent SynthDoc class
        """
        super().__init__(output_dir=output_dir, **kwargs)

        self.dataset_root = dataset_root or Path("./datasets")
        self.auto_create_datasets = auto_create_datasets

        # Initialize workflow managers
        self.workflows = {}

        if auto_create_datasets:
            self._initialize_workflows()

    def _initialize_workflows(self):
        """Initialize dataset workflows for different task types."""
        # Image captioning workflow
        self.workflows["captioning"] = create_image_captioning_workflow(
            dataset_root=self.dataset_root, workflow_name="synthdoc_captioning"
        )

        # VQA workflow
        self.workflows["vqa"] = create_vqa_workflow(
            dataset_root=self.dataset_root, workflow_name="synthdoc_vqa"
        )

        # OCR workflow
        self.workflows["ocr"] = create_ocr_workflow(
            dataset_root=self.dataset_root, workflow_name="synthdoc_ocr"
        )

    def generate_captioning_dataset(
        self,
        num_samples: int = 100,
        dataset_name: Optional[str] = None,
        split_ratios: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete image captioning dataset.

        Args:
            num_samples: Number of samples to generate
            dataset_name: Name for the dataset
            split_ratios: Ratios for train/test/validation splits

        Returns:
            Dictionary with dataset information and HuggingFace dataset
        """
        if split_ratios is None:
            split_ratios = {"train": 0.8, "test": 0.1, "validation": 0.1}

        # Initialize dataset
        workflow = self.workflows["captioning"]
        if dataset_name:
            workflow.initialize_dataset(dataset_name)
        else:
            workflow.initialize_dataset()

        # Generate samples
        for i in range(num_samples):
            # Determine split based on ratios
            split = self._determine_split(i, num_samples, split_ratios)

            # Generate document image and caption
            # This is a placeholder - you would integrate with actual SynthDoc generators
            image_path = self._generate_document_image(f"captioning_sample_{i}")
            caption = self._generate_caption_for_image(image_path)

            # Add to dataset
            workflow.add_captioned_image(
                image_path=image_path,
                caption=caption,
                split=split,
                source="synthdoc_generator",
            )

            if i % 10 == 0:
                self.logger.info(f"Generated {i}/{num_samples} captioning samples")

        # Finalize dataset
        dataset = workflow.finalize_dataset()

        return {
            "dataset": dataset,
            "path": workflow.manager.dataset_path,
            "stats": workflow.manager.get_stats(),
            "workflow": workflow,
        }

    def generate_vqa_dataset(
        self,
        num_samples: int = 100,
        questions_per_image: int = 3,
        dataset_name: Optional[str] = None,
        split_ratios: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete VQA dataset.

        Args:
            num_samples: Number of document images to generate
            questions_per_image: Number of questions per image
            dataset_name: Name for the dataset
            split_ratios: Ratios for train/test/validation splits

        Returns:
            Dictionary with dataset information and HuggingFace dataset
        """
        if split_ratios is None:
            split_ratios = {"train": 0.8, "test": 0.1, "validation": 0.1}

        # Initialize dataset
        workflow = self.workflows["vqa"]
        if dataset_name:
            workflow.initialize_dataset(dataset_name)
        else:
            workflow.initialize_dataset()

        # Generate samples
        sample_count = 0
        for i in range(num_samples):
            # Generate document image
            image_path = self._generate_document_image(f"vqa_doc_{i}")

            # Generate multiple QA pairs for this image
            for j in range(questions_per_image):
                split = self._determine_split(
                    sample_count, num_samples * questions_per_image, split_ratios
                )

                # Generate QA pair
                question, answer, q_type, difficulty = self._generate_qa_pair_for_image(
                    image_path, j
                )

                # Add to dataset
                workflow.add_vqa_sample(
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    question_type=q_type,
                    difficulty=difficulty,
                    split=split,
                    source="synthdoc_generator",
                )

                sample_count += 1

            if i % 10 == 0:
                self.logger.info(f"Generated VQA for {i}/{num_samples} documents")

        # Finalize dataset
        dataset = workflow.finalize_dataset()

        return {
            "dataset": dataset,
            "path": workflow.manager.dataset_path,
            "stats": workflow.manager.get_stats(),
            "workflow": workflow,
        }

    def generate_ocr_dataset(
        self,
        num_samples: int = 100,
        dataset_name: Optional[str] = None,
        include_annotations: bool = True,
        split_ratios: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete OCR dataset.

        Args:
            num_samples: Number of samples to generate
            dataset_name: Name for the dataset
            include_annotations: Whether to include word/line annotations
            split_ratios: Ratios for train/test/validation splits

        Returns:
            Dictionary with dataset information and HuggingFace dataset
        """
        if split_ratios is None:
            split_ratios = {"train": 0.8, "test": 0.1, "validation": 0.1}

        # Initialize dataset
        workflow = self.workflows["ocr"]
        if dataset_name:
            workflow.initialize_dataset(dataset_name)
        else:
            workflow.initialize_dataset()

        # Generate samples
        for i in range(num_samples):
            split = self._determine_split(i, num_samples, split_ratios)

            # Generate document image with known text content
            image_path, text_content = self._generate_document_with_text(
                f"ocr_sample_{i}"
            )

            # Generate word-level annotations if requested
            words = None
            lines = None
            if include_annotations:
                words, lines = self._generate_ocr_annotations(image_path, text_content)

            # Add to dataset
            workflow.add_ocr_sample(
                image_path=image_path,
                text=text_content,
                words=words,
                lines=lines,
                language="en",  # This could be dynamic based on generation
                split=split,
                source="synthdoc_generator",
            )

            if i % 10 == 0:
                self.logger.info(f"Generated {i}/{num_samples} OCR samples")

        # Finalize dataset
        dataset = workflow.finalize_dataset()

        return {
            "dataset": dataset,
            "path": workflow.manager.dataset_path,
            "stats": workflow.manager.get_stats(),
            "workflow": workflow,
        }

    def _determine_split(
        self, index: int, total: int, split_ratios: Dict[str, float]
    ) -> SplitType:
        """Determine which split an item should go to based on ratios."""
        ratio = index / total

        if ratio < split_ratios.get("train", 0.8):
            return SplitType.TRAIN
        elif ratio < split_ratios.get("train", 0.8) + split_ratios.get("test", 0.1):
            return SplitType.TEST
        else:
            return SplitType.VALIDATION

    def _generate_document_image(self, identifier: str) -> Path:
        """
        Generate a document image using SynthDoc generators.

        This is a placeholder method - integrate with actual SynthDoc functionality.
        """
        # Placeholder implementation
        # In reality, this would use SynthDoc's document generators
        output_path = self.output_dir / f"{identifier}.png"

        # TODO: Integrate with actual SynthDoc document generation
        # For now, create a placeholder
        if not output_path.exists():
            # Create a simple placeholder image
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (800, 600), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), f"Generated Document: {identifier}", fill="black")
            img.save(output_path)

        return output_path

    def _generate_caption_for_image(self, image_path: Path) -> str:
        """Generate a caption for a document image."""
        # Placeholder implementation
        # In reality, this would analyze the generated document
        return f"A synthetic document image containing text and layout elements"

    def _generate_qa_pair_for_image(
        self, image_path: Path, question_index: int
    ) -> tuple:
        """Generate a question-answer pair for a document image."""
        # Placeholder implementation
        # In reality, this would use SynthDoc's VQA generator
        questions = [
            (
                "What type of document is this?",
                "A synthetic document",
                "classification",
                "easy",
            ),
            ("How many paragraphs are visible?", "2", "counting", "medium"),
            (
                "What can you infer about the document's purpose?",
                "It appears to be a test document",
                "reasoning",
                "hard",
            ),
        ]

        return questions[question_index % len(questions)]

    def _generate_document_with_text(self, identifier: str) -> tuple:
        """Generate a document image with known text content."""
        # Placeholder implementation
        output_path = self.output_dir / f"{identifier}.png"
        text_content = f"Sample text content for {identifier}\nThis is a multi-line document\nwith various text elements."

        # TODO: Integrate with actual SynthDoc text generation
        if not output_path.exists():
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (800, 600), color="white")
            draw = ImageDraw.Draw(img)

            # Draw the text content
            y_offset = 50
            for line in text_content.split("\n"):
                draw.text((50, y_offset), line, fill="black")
                y_offset += 30

            img.save(output_path)

        return output_path, text_content

    def _generate_ocr_annotations(self, image_path: Path, text_content: str) -> tuple:
        """Generate word and line level annotations for OCR."""
        # Placeholder implementation
        # In reality, this would use actual layout analysis

        words = []
        lines = []

        y_offset = 50
        for line_idx, line in enumerate(text_content.split("\n")):
            line_bbox = [50, y_offset, 50 + len(line) * 10, y_offset + 25]
            lines.append({"text": line, "bbox": line_bbox})

            x_offset = 50
            for word in line.split():
                word_width = len(word) * 10
                words.append(
                    {
                        "text": word,
                        "bbox": [
                            x_offset,
                            y_offset,
                            x_offset + word_width,
                            y_offset + 25,
                        ],
                    }
                )
                x_offset += word_width + 10

            y_offset += 30

        return words, lines

    def upload_all_datasets_to_hub(
        self, repo_prefix: str, private: bool = False, token: Optional[str] = None
    ) -> Dict[str, str]:
        """Upload all created datasets to HuggingFace Hub."""
        urls = {}

        for task_type, workflow in self.workflows.items():
            if hasattr(workflow, "manager") and workflow.manager:
                repo_id = f"{repo_prefix}/{task_type}_dataset"
                try:
                    url = workflow.upload_to_hub(
                        repo_id=repo_id,
                        private=private,
                        token=token,
                        commit_message=f"Upload {task_type} dataset from SynthDoc",
                    )
                    urls[task_type] = url
                    self.logger.info(f"Uploaded {task_type} dataset to {url}")
                except Exception as e:
                    self.logger.error(f"Failed to upload {task_type} dataset: {e}")

        return urls


def example_usage():
    """Example of using the DatasetEnabledSynthDoc class."""

    # Initialize enhanced SynthDoc
    synthdoc = DatasetEnabledSynthDoc(
        output_dir=Path("./temp_output"),
        dataset_root=Path("./datasets"),
        auto_create_datasets=True,
    )

    print("=== Generating Image Captioning Dataset ===")
    captioning_result = synthdoc.generate_captioning_dataset(
        num_samples=50, dataset_name="my_captioning_dataset"
    )
    print(
        f"Created captioning dataset with {captioning_result['stats'].total_items} items"
    )

    print("\n=== Generating VQA Dataset ===")
    vqa_result = synthdoc.generate_vqa_dataset(
        num_samples=20, questions_per_image=3, dataset_name="my_vqa_dataset"
    )
    print(f"Created VQA dataset with {vqa_result['stats'].total_items} items")

    print("\n=== Generating OCR Dataset ===")
    ocr_result = synthdoc.generate_ocr_dataset(
        num_samples=30, dataset_name="my_ocr_dataset", include_annotations=True
    )
    print(f"Created OCR dataset with {ocr_result['stats'].total_items} items")

    # Upload all datasets to Hub (uncomment to actually upload)
    # urls = synthdoc.upload_all_datasets_to_hub("your-username")
    # print(f"Uploaded datasets: {urls}")

    return {"captioning": captioning_result, "vqa": vqa_result, "ocr": ocr_result}


if __name__ == "__main__":
    # Note: This example uses placeholder implementations
    # In practice, you would integrate with actual SynthDoc generators
    results = example_usage()
    print("\nDataset generation completed!")
    print("Note: This example uses placeholder document generation.")
    print("Integrate with actual SynthDoc generators for real datasets.")
