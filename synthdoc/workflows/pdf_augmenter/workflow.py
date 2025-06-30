import random
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

from ..base import BaseWorkflow
from ...models import PDFAugmentationConfig, WorkflowResult, AugmentationType
from ...utils import CostTracker

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PDFAugmenter(BaseWorkflow):
    """Augment existing PDF documents by extracting and recombining elements."""

    def __init__(self, save_dir: str = "pdf_augmentation_output"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.cost_tracker = CostTracker()

    def process(self, config: PDFAugmentationConfig) -> WorkflowResult:
        """Augment PDF documents by extracting and recombining elements."""
        print(f"📄 Starting PDF augmentation for {len(config.pdf_files)} files...")
        
        if not PYMUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            print("⚠️  PDF processing libraries not available. Install with: pip install PyMuPDF pdfplumber")
            return self._create_fallback_result(config)
        
        # Extract elements from all PDF files
        all_elements = self._extract_elements_from_corpus(config.pdf_files)
        print(f"📑 Extracted {len(all_elements)} elements from corpus")
        
        # Generate new documents by recombining elements
        generated_docs = self._generate_new_documents(all_elements, config)
        print(f"🔄 Generated {len(generated_docs)} new documents")
        
        # Apply augmentations if specified
        if config.augmentations:
            generated_docs = self._apply_augmentations(generated_docs, config.augmentations)
            print(f"🎨 Applied {len(config.augmentations)} augmentations")
        
        # Create output files
        output_files = self._save_generated_documents(generated_docs)
        
        # Create HuggingFace dataset format
        samples = self._create_dataset_samples(generated_docs, output_files, config)
        dataset = self._create_hf_dataset(
            samples,
            {
                "workflow": "pdf_augmentation",
                "config": config.dict(),
                "corpus_files": len(config.pdf_files)
            }
        )
        
        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "pdf_augmentation",
                "corpus_files": len(config.pdf_files),
                "elements_extracted": len(all_elements),
                "documents_generated": len(generated_docs),
                "augmentations_applied": config.augmentations or [],
                "preserve_text": config.preserve_text
            },
            num_samples=len(generated_docs),
            output_files=output_files
        )

    def _extract_elements_from_corpus(self, pdf_files: List[Path]) -> List[Dict[str, Any]]:
        """Extract reusable elements from corpus PDFs."""
        all_elements = []
        
        for pdf_path in pdf_files:
            try:
                if PYMUPDF_AVAILABLE:
                    elements = self._extract_with_pymupdf(pdf_path)
                elif PDFPLUMBER_AVAILABLE:
                    elements = self._extract_with_pdfplumber(pdf_path)
                else:
                    continue
                    
                all_elements.extend(elements)
                print(f"  📄 {pdf_path.name}: extracted {len(elements)} elements")
                
            except Exception as e:
                print(f"  ❌ Error processing {pdf_path.name}: {e}")
                continue
        
        return all_elements

    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract elements using PyMuPDF."""
        elements = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text blocks
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:  # Text block
                        text_content = self._extract_text_from_block(block)
                        if text_content.strip():  # Only add non-empty blocks
                            element = {
                                "type": "text_block",
                                "content": text_content,
                                "bbox": block["bbox"],
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "element_category": self._categorize_text_element(text_content)
                            }
                            elements.append(element)
            
            doc.close()
            
        except Exception as e:
            print(f"    ❌ PyMuPDF extraction error: {e}")
        
        return elements

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract elements using pdfplumber."""
        elements = []
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    
                    # Extract text with positions
                    text = page.extract_text()
                    if text and text.strip():
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        
                        for para in paragraphs:
                            element = {
                                "type": "text_block",
                                "content": para,
                                "bbox": [0, 0, page.width, 50],  # Simplified bbox
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "element_category": self._categorize_text_content(para)
                            }
                            elements.append(element)
                        
        except Exception as e:
            print(f"    ❌ pdfplumber extraction error: {e}")
        
        return elements

    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a PyMuPDF text block."""
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "") + " "
        return text.strip()

    def _categorize_text_element(self, text: str) -> str:
        """Categorize text element based on content."""
        text = text.strip()
        
        if len(text) < 50:
            return "headers"
        elif text.startswith(("•", "-", "1.", "2.", "3.", "*")):
            return "lists"
        elif len(text) > 100:
            return "paragraphs"
        else:
            return "other"

    def _categorize_text_content(self, text: str) -> str:
        """Categorize text based on content analysis."""
        return self._categorize_text_element(text)

    def _generate_new_documents(self, elements: List[Dict[str, Any]], config: PDFAugmentationConfig) -> List[Dict[str, Any]]:
        """Generate new documents by intelligently combining elements."""
        if not elements:
            return []
        
        # Categorize elements
        categorized = self._categorize_elements(elements)
        
        generated_docs = []
        num_docs_to_generate = min(10, max(3, len(config.pdf_files) * 2))
        
        for doc_idx in range(num_docs_to_generate):
            doc = self._create_document_from_elements(categorized, doc_idx, config.preserve_text)
            generated_docs.append(doc)
        
        return generated_docs

    def _categorize_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize extracted elements by type."""
        categorized = {
            "headers": [],
            "paragraphs": [],
            "lists": [],
            "other": []
        }
        
        for element in elements:
            category = element.get("element_category", "other")
            if category in categorized:
                categorized[category].append(element)
            else:
                categorized["other"].append(element)
        
        return categorized

    def _create_document_from_elements(self, categorized: Dict[str, List], doc_idx: int, preserve_text: bool) -> Dict[str, Any]:
        """Create a new document by combining elements."""
        doc_elements = []
        
        # Start with a header if available
        if categorized["headers"]:
            header = random.choice(categorized["headers"])
            doc_elements.append(header)
        
        # Add 2-4 paragraphs
        num_paragraphs = random.randint(2, 4)
        for _ in range(num_paragraphs):
            if categorized["paragraphs"]:
                para = random.choice(categorized["paragraphs"])
                doc_elements.append(para)
        
        # Optionally add lists
        if categorized["lists"] and random.random() < 0.3:
            list_elem = random.choice(categorized["lists"])
            doc_elements.append(list_elem)
        
        # Create document image
        doc_image = self._render_document_from_elements(doc_elements, doc_idx)
        
        return {
            "id": f"pdf_recombined_{doc_idx}",
            "elements": doc_elements,
            "image": doc_image,
            "preserve_text": preserve_text,
            "num_elements": len(doc_elements),
            "element_types": [elem["type"] for elem in doc_elements]
        }

    def _render_document_from_elements(self, elements: List[Dict], doc_idx: int) -> Image.Image:
        """Render document elements into a single image."""
        img_width, img_height = 595, 842  # A4 size
        image = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        y_offset = 50
        margin = 50
        
        for element in elements:
            content = str(element.get("content", ""))
            if content:
                # Simple text rendering
                lines = self._wrap_text(content, img_width - 2*margin, 80)
                
                for line in lines[:10]:  # Limit lines
                    if y_offset < img_height - 100:
                        draw.text((margin, y_offset), line, fill="black", font=font)
                        y_offset += 20
                
                y_offset += 15
        
        return image

    def _wrap_text(self, text: str, max_width: int, char_width: int = 8) -> List[str]:
        """Simple text wrapping."""
        max_chars = max_width // char_width
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def _apply_augmentations(self, documents: List[Dict], augmentations: List[AugmentationType]) -> List[Dict]:
        """Apply augmentations to generated documents."""
        augmented_docs = []
        
        for doc in documents:
            for aug_type in augmentations:
                augmented_doc = doc.copy()
                augmented_doc["image"] = self._apply_single_augmentation(doc["image"], aug_type)
                augmented_doc["id"] = f"{doc['id']}_{aug_type.value}"
                augmented_doc["augmentation"] = aug_type.value
                augmented_docs.append(augmented_doc)
        
        return documents + augmented_docs

    def _apply_single_augmentation(self, image: Image.Image, aug_type: AugmentationType) -> Image.Image:
        """Apply single augmentation."""
        try:
            if aug_type == AugmentationType.ROTATION:
                angle = random.uniform(-5, 5)
                return image.rotate(angle, expand=True, fillcolor='white')
            elif aug_type == AugmentationType.SCALING:
                scale = random.uniform(0.9, 1.1)
                new_size = (int(image.width * scale), int(image.height * scale))
                return image.resize(new_size)
            else:
                return image
        except:
            return image

    def _save_generated_documents(self, documents: List[Dict]) -> List[str]:
        """Save documents and return paths."""
        output_files = []
        
        for doc in documents:
            image_path = self.save_dir / f"recombined_doc_{doc['id']}.png"
            doc["image"].save(image_path)
            output_files.append(str(image_path))
        
        return output_files

    def _create_dataset_samples(self, documents: List[Dict], output_files: List[str], config: PDFAugmentationConfig) -> List[Dict[str, Any]]:
        """Create dataset samples for HuggingFace Dataset."""
        samples = []
        
        for i, doc in enumerate(documents):
            sample = {
                "id": f"pdf_recombined_{doc['id']}",
                "image": doc["image"],
                "image_path": output_files[i] if i < len(output_files) else "",
                "image_width": doc["image"].width,
                "image_height": doc["image"].height,
                "pdf_name": f"recombined_doc_{doc['id']}",
                "page_number": 0,
                "markdown": self._extract_text_from_doc_elements(doc["elements"]),
                "html": f"<div>{self._extract_text_from_doc_elements(doc['elements'])}</div>",
                "layout": self._create_layout_annotations(doc["elements"], doc["image"].width, doc["image"].height),
                "lines": [],  # Would need OCR for line detection
                "images_embedded": [],  # No image extraction in this implementation
                "equations": [],  # No equation detection
                "tables": [],  # No table extraction in this implementation
                "page_size": [doc["image"].width, doc["image"].height],
                "content_list": doc["elements"],
                "base_layout_detection": {
                    "num_elements": doc["num_elements"],
                    "element_types": doc["element_types"],
                    "layout_type": "recombined"
                },
                "pdf_info": {
                    "source": "pdf_augmentation_recombination",
                    "generation_method": "element_recombination"
                },
                "recombination_metadata": {
                    "source_elements": len(doc["elements"]),
                    "element_types": doc["element_types"],
                    "preserve_text": doc["preserve_text"]
                }
            }
            samples.append(sample)
        
        return samples

    def _extract_text_from_doc_elements(self, elements: List[Dict]) -> str:
        """Extract text from document elements."""
        text_parts = []
        for element in elements:
            content = element.get("content", "")
            if content:
                text_parts.append(str(content))
        return "\n\n".join(text_parts)

    def _create_layout_annotations(self, elements: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        """Create layout annotations."""
        annotations = []
        
        for i, element in enumerate(elements):
            annotation = {
                "id": i,
                "type": element["type"],
                "category": element.get("element_category", "other"),
                "confidence": 0.9
            }
            annotations.append(annotation)
        
        return annotations

    def _create_fallback_result(self, config: PDFAugmentationConfig) -> WorkflowResult:
        """Create fallback result when PDF libraries unavailable."""
        # Create empty dataset
        dataset = self._create_hf_dataset(
            [],
            {"note": "PDF processing requires PyMuPDF or pdfplumber"}
        )
        
        return WorkflowResult(
            dataset=dataset,
            metadata={"workflow_type": "pdf_augmentation", "status": "fallback_mode"},
            num_samples=0,
            output_files=[]
        ) 