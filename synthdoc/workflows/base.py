from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from PIL import Image
from datasets import Dataset


class BaseWorkflow(ABC):
    """Base class for all SynthDoc workflows."""

    @abstractmethod
    def process(self, config):
        """Process the workflow with given configuration and return WorkflowResult with HuggingFace Dataset."""
        pass

    def _create_hf_dataset(
        self, samples: list, metadata: dict = None
    ) -> Dataset:
        """Helper method to create actual HuggingFace Dataset object."""
        if not samples:
            # Create empty dataset with minimal schema
            return Dataset.from_dict({})
        
        # Create Dataset from list of samples
        dataset = Dataset.from_list(samples)
        
        # Add metadata to dataset info if provided
        if metadata:
            dataset.info.description = f"SynthDoc generated dataset: {metadata.get('workflow', 'unknown')}"
            dataset.info.features = dataset.features
            
        return dataset

    def _create_comprehensive_hf_dataset(
        self,
        images: List[Image.Image],
        image_paths: List[str],
        pdf_names: List[str],
        page_numbers: List[int] = None,
        markdown_content: List[str] = None,
        html_content: List[str] = None,
        layout_annotations: List[List[Dict]] = None,
        line_annotations: List[List[Dict]] = None,
        embedded_images: List[List[Dict]] = None,
        equations: List[List[Dict]] = None,
        tables: List[List[Dict]] = None,
        content_lists: List[List[Dict]] = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dataset:
        """
        Create comprehensive HuggingFace dataset format matching documented schema.
        
        This creates the full schema documented in README:
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
        """
        num_samples = len(images)
        
        # Ensure all lists have the same length
        page_numbers = page_numbers or [0] * num_samples
        markdown_content = markdown_content or [""] * num_samples
        html_content = html_content or [""] * num_samples
        layout_annotations = layout_annotations or [[] for _ in range(num_samples)]
        line_annotations = line_annotations or [[] for _ in range(num_samples)]
        embedded_images = embedded_images or [[] for _ in range(num_samples)]
        equations = equations or [[] for _ in range(num_samples)]
        tables = tables or [[] for _ in range(num_samples)]
        content_lists = content_lists or [[] for _ in range(num_samples)]
        
        # Create samples list for Dataset
        samples = []
        for i in range(num_samples):
            sample = {
                # Core image data
                "image": images[i],
                "image_width": images[i].width,
                "image_height": images[i].height,
                "image_path": image_paths[i],
                
                # Document identification
                "pdf_name": pdf_names[i],
                "page_number": page_numbers[i],
                
                # Content in different formats
                "markdown": markdown_content[i],
                "html": html_content[i],
                
                # Layout and structure annotations
                "layout": layout_annotations[i],
                "lines": line_annotations[i],
                "images_embedded": embedded_images[i],
                "equations": equations[i],
                "tables": tables[i],
                
                # Page and document metadata
                "page_size": [images[i].width, images[i].height],
                "content_list": content_lists[i],
                
                # Layout detection ground truth
                "base_layout_detection": self._create_layout_detection_metadata(
                    layout_annotations[i], 
                    content_lists[i] if content_lists else []
                ),
                
                # Document metadata
                "pdf_info": self._create_pdf_info_metadata(
                    pdf_names[i], 
                    page_numbers[i], 
                    additional_metadata
                )
            }
            samples.append(sample)
        
        # Create and return actual HuggingFace Dataset
        if not samples:
            return Dataset.from_dict({})
        
        return Dataset.from_list(samples)

    def _create_layout_detection_metadata(self, layout_annotations: List[Dict], content_list: List[Dict]) -> Dict[str, Any]:
        """Create layout detection ground truth metadata."""
        element_types = []
        element_counts = {}
        
        for annotation in layout_annotations:
            element_type = annotation.get("type", "unknown")
            element_types.append(element_type)
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        return {
            "total_elements": len(layout_annotations),
            "element_types": element_types,
            "element_counts": element_counts,
            "content_elements": len(content_list),
            "layout_complexity": self._assess_layout_complexity(layout_annotations),
            "has_multi_column": self._detect_multi_column_layout(layout_annotations),
            "text_regions": len([a for a in layout_annotations if a.get("type") == "text"]),
            "image_regions": len([a for a in layout_annotations if a.get("type") == "image"]),
            "table_regions": len([a for a in layout_annotations if a.get("type") == "table"])
        }

    def _create_pdf_info_metadata(self, pdf_name: str, page_number: int, additional_metadata: Dict) -> Dict[str, Any]:
        """Create PDF information metadata."""
        metadata = {
            "source_document": pdf_name,
            "page_number": page_number,
            "generation_timestamp": None,  # Would be set by calling workflow
            "processing_pipeline": "synthdoc",
            "workflow_version": "0.2.0"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata

    def _assess_layout_complexity(self, layout_annotations: List[Dict]) -> str:
        """Assess the complexity of the document layout."""
        num_elements = len(layout_annotations)
        
        if num_elements <= 3:
            return "simple"
        elif num_elements <= 10:
            return "moderate"
        else:
            return "complex"

    def _detect_multi_column_layout(self, layout_annotations: List[Dict]) -> bool:
        """Detect if the layout has multiple columns."""
        # Simple heuristic: if there are text elements with significantly different x positions
        text_elements = [a for a in layout_annotations if a.get("type") == "text"]
        
        if len(text_elements) < 2:
            return False
        
        x_positions = []
        for element in text_elements:
            bbox = element.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x_positions.append(bbox[0])  # Left x coordinate
        
        if len(set(x_positions)) > 1:
            # Check if there are distinct column positions
            sorted_x = sorted(set(x_positions))
            if len(sorted_x) >= 2 and (sorted_x[1] - sorted_x[0]) > 100:  # Significant gap
                return True
        
        return False

    def _create_bounding_box_annotation(
        self, 
        x: float, 
        y: float, 
        width: float, 
        height: float, 
        element_type: str,
        element_id: int = None,
        confidence: float = 1.0,
        text_content: str = None
    ) -> Dict[str, Any]:
        """Create a standardized bounding box annotation."""
        return {
            "id": element_id,
            "type": element_type,
            "bbox": [x, y, width, height],
            "bbox_normalized": [x, y, x + width, y + height],  # [x1, y1, x2, y2] format
            "confidence": confidence,
            "text": text_content,
            "area": width * height
        }

    def _create_text_line_annotation(
        self,
        line_text: str,
        bbox: List[float],
        line_id: int = None,
        confidence: float = 1.0,
        language: str = None
    ) -> Dict[str, Any]:
        """Create a text line annotation."""
        return {
            "id": line_id,
            "text": line_text,
            "bbox": bbox,
            "confidence": confidence,
            "language": language,
            "word_count": len(line_text.split()),
            "character_count": len(line_text)
        }

    def _create_table_annotation(
        self,
        table_data: List[List[str]],
        bbox: List[float],
        table_id: int = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Create a table structure annotation."""
        return {
            "id": table_id,
            "bbox": bbox,
            "confidence": confidence,
            "rows": len(table_data) if table_data else 0,
            "columns": len(table_data[0]) if table_data and table_data[0] else 0,
            "data": table_data,
            "has_header": True if table_data and len(table_data) > 1 else False
        }

    def _create_image_annotation(
        self,
        image_bbox: List[float],
        image_id: int = None,
        image_type: str = "embedded",
        caption: str = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Create an embedded image annotation."""
        return {
            "id": image_id,
            "type": image_type,
            "bbox": image_bbox,
            "confidence": confidence,
            "caption": caption,
            "area": image_bbox[2] * image_bbox[3] if len(image_bbox) >= 4 else 0
        }

    def _create_equation_annotation(
        self,
        equation_text: str,
        bbox: List[float],
        equation_id: int = None,
        equation_type: str = "inline",
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Create a mathematical equation annotation."""
        return {
            "id": equation_id,
            "type": equation_type,  # "inline" or "display"
            "bbox": bbox,
            "confidence": confidence,
            "latex": equation_text,
            "mathml": None  # Could be generated from LaTeX
        } 