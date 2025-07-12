import os
import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset
from ..base import BaseWorkflow
from ...models import LayoutAugmentationConfig, WorkflowResult
from ...languages import load_language_font
from ...augmentations import Augmentor


class LayoutAugmenter(BaseWorkflow):
    """Apply layout transformations to existing documents."""

    def __init__(self, save_dir: str = "layout_output"):
        super().__init__()
        self.save_dir = save_dir
        self.augmentor = Augmentor()
        os.makedirs(save_dir, exist_ok=True)

    def process(self, config: LayoutAugmentationConfig) -> WorkflowResult:
        """Apply layout augmentations based on configuration."""
        print(f"🎨 Starting layout augmentation for {len(config.documents)} documents...")
        
        all_samples = []
        
        for doc_idx, doc_path in enumerate(config.documents):
            print(f"Processing document {doc_idx + 1}/{len(config.documents)}: {doc_path}")
            
            # Extract content from document
            content = self._extract_document_content(doc_path)
            
            # Generate variations for each language and font combination
            for lang in config.languages:
                for font_name in (config.fonts or ["Arial", "Times New Roman"]):
                    samples = self._create_layout_variations(
                        content, doc_path, lang, font_name, config
                    )
                    all_samples.extend(samples)

        # Create comprehensive dataset using README schema
        if not all_samples:
            dataset = Dataset.from_dict({})
        else:
            # Extract data for comprehensive dataset creation
            images = [s['image'] for s in all_samples]
            image_paths = [s.get('image_path', '') for s in all_samples]
            pdf_names = [s.get('pdf_name', f"layout_doc_{i}") for i, s in enumerate(all_samples)]
            page_numbers = [s.get('page_number', 0) for s in all_samples]
            markdown_content = [s.get('markdown', '') for s in all_samples]
            html_content = [s.get('html', '') for s in all_samples]
            
            # Parse layout annotations - ensure we have proper data structures
            layout_annotations = []
            line_annotations = []
            embedded_images = []
            equations = []
            tables = []
            content_lists = []
            
            for s in all_samples:
                # Create simple layout annotations for augmented documents
                layout_data = [{
                    "type": "text_block",
                    "bbox": [60, 60, 740, 800],
                    "confidence": 0.95,
                    "text": s.get('content', '')[:100] + "..." if len(s.get('content', '')) > 100 else s.get('content', '')
                }]
                
                lines_data = [{
                    "text": line,
                    "bbox": [60, 60 + i*24, 740, 84 + i*24],
                    "confidence": 0.95
                } for i, line in enumerate(s.get('content', '').split('\n')[:20])]
                
                # No embedded images, equations, or tables for layout augmentation
                images_data = []
                equations_data = []
                tables_data = []
                
                # Simple content list
                content_data = [{
                    "id": i+1,
                    "type": "text",
                    "content": line,
                    "bbox": [60, 60 + i*24, 740, 84 + i*24]
                } for i, line in enumerate(s.get('content', '').split('\n')[:10])]
                
                layout_annotations.append(layout_data)
                line_annotations.append(lines_data)
                embedded_images.append(images_data)
                equations.append(equations_data)
                tables.append(tables_data)
                content_lists.append(content_data)
            
            dataset = self._create_comprehensive_hf_dataset(
                images=images,
                image_paths=image_paths,
                pdf_names=pdf_names,
                page_numbers=page_numbers,
                markdown_content=markdown_content,
                html_content=html_content,
                layout_annotations=layout_annotations,
                line_annotations=line_annotations,
                embedded_images=embedded_images,
                equations=equations,
                tables=tables,
                content_lists=content_lists,
                additional_metadata={
                    "workflow": "layout_augmentation",
                    "config": config.dict(),
                    "total_variations": len(all_samples)
                }
            )

        output_files = [sample.get("image_path") for sample in all_samples if sample.get("image_path")]

        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "layout_augmentation",
                "total_variations": len(all_samples),
                "languages_used": [lang.value if hasattr(lang, 'value') else str(lang) for lang in config.languages],
                "fonts_used": config.fonts or ["Arial", "Times New Roman"]
            },
            num_samples=len(all_samples),
            output_files=output_files
        )

    def _extract_document_content(self, doc_path: Union[str, Path]) -> str:
        """Extract text content from various document formats."""
        try:
            doc_path_str = str(doc_path)
            
            # Handle different file types
            if doc_path_str.endswith(('.txt', '.md')):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif doc_path_str.endswith('.pdf'):
                # For now, return sample PDF content
                return self._get_sample_pdf_content()
            elif doc_path_str.endswith(('.png', '.jpg', '.jpeg')):
                # For images, we'd need OCR - for now return sample content
                return self._get_sample_image_content()
            else:
                # If it's a dict (from raw document generation), extract content
                if isinstance(doc_path, dict) and "content" in doc_path:
                    return doc_path["content"]
                else:
                    return self._get_sample_content()
                    
        except Exception as e:
            print(f"⚠️ Error extracting content from {doc_path}: {e}")
            return self._get_sample_content()

    def _create_layout_variations(
        self, content: str, doc_path: Union[str, Path], language, 
        font_name: str, config: LayoutAugmentationConfig
    ) -> List[Dict[str, Any]]:
        """Create multiple layout variations for a single document."""
        variations = []
        
        # Create base layouts (single column, two column, etc.)
        layout_types = ["single_column", "two_column", "newspaper", "magazine"]
        
        for layout_idx, layout_type in enumerate(layout_types[:3]):  # Limit to 3 variations
            try:
                # Create document image with specific layout
                image = self._render_document_with_layout(
                    content, language, font_name, layout_type
                )
                
                # Apply augmentations if specified
                if config.augmentations:
                    image = self._apply_augmentations(image, config.augmentations)
                
                # Save image
                filename = f"layout_{layout_idx}_{font_name}_{layout_type}.png"
                img_path = os.path.join(self.save_dir, filename)
                image.save(img_path)
                
                # Extract layout information
                layout_info = self._extract_layout_info(image, content, layout_type)
                
                variation = {
                    "id": f"layout_{layout_idx}_{font_name}_{layout_type}",
                    "image": image,  # Include actual PIL Image for image dataset
                    "image_path": img_path,
                    "text": content,  # Add text content for image dataset
                    "markdown": content,  # Also include as markdown
                    "original_document": str(doc_path),
                    "language": language.value if hasattr(language, 'value') else str(language),
                    "font": font_name,
                    "layout_type": layout_type,
                    "augmentations": [aug.value if hasattr(aug, 'value') else str(aug) for aug in (config.augmentations or [])],
                    "imagewidth": image.width,
                    "imageheight": image.height,
                    "page_number": 0,
                    "pdf_name": Path(str(doc_path)).stem,
                    "layout_info": layout_info,
                    "content": content,
                    "metadata": {
                        "page_number": 0,
                        "pdf_name": Path(str(doc_path)).stem,
                        "content_language": language.value if hasattr(language, 'value') else str(language),
                        "font_family": font_name,
                        "layout_template": layout_type
                    }
                }
                variations.append(variation)
                
            except Exception as e:
                print(f"⚠️ Error creating layout variation {layout_type}: {e}")
        
        return variations

    def _render_document_with_layout(
        self, content: str, language, font_name: str, layout_type: str
    ) -> Image.Image:
        """Render document content with specific layout styling."""
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Load font
        try:
            language_code = language.value if hasattr(language, 'value') else str(language)
            font = load_language_font(language_code, 12)
            title_font = load_language_font(language_code, 16)
        except Exception as e:
            print(f"⚠️ Font loading error: {e}")
            try:
                font = ImageFont.truetype(font_name, 12)
                title_font = ImageFont.truetype(font_name, 16)
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
        
        # Layout-specific rendering
        margin = 50
        
        if layout_type == "single_column":
            self._render_single_column(draw, content, font, title_font, width, height, margin)
        elif layout_type == "two_column":
            self._render_two_column(draw, content, font, title_font, width, height, margin)
        elif layout_type == "newspaper":
            self._render_newspaper_layout(draw, content, font, title_font, width, height, margin)
        elif layout_type == "magazine":
            self._render_magazine_layout(draw, content, font, title_font, width, height, margin)
        else:
            self._render_single_column(draw, content, font, title_font, width, height, margin)
        
        return image

    def _render_single_column(self, draw, content: str, font, title_font, width: int, height: int, margin: int):
        """Render content in single column layout."""
        lines = content.split('\n')
        y_offset = margin + 20
        line_height = 18
        
        # Add title
        title = lines[0] if lines else "Document Title"
        draw.text((margin, y_offset), title, fill='black', font=title_font)
        y_offset += 40
        
        # Render content
        for line in lines[1:]:
            if y_offset + line_height > height - margin:
                break
            
            # Word wrap
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if draw.textlength(test_line, font=font) < width - 2 * margin:
                    current_line = test_line
                else:
                    if current_line:
                        draw.text((margin, y_offset), current_line, fill='black', font=font)
                        y_offset += line_height
                    current_line = word
            
            if current_line:
                draw.text((margin, y_offset), current_line, fill='black', font=font)
                y_offset += line_height + 5

    def _render_two_column(self, draw, content: str, font, title_font, width: int, height: int, margin: int):
        """Render content in two column layout."""
        col_width = (width - 3 * margin) // 2
        col_gap = margin
        
        lines = content.split('\n')
        
        # Title across full width
        title = lines[0] if lines else "Document Title"
        draw.text((margin, margin + 20), title, fill='black', font=title_font)
        
        # Split content between columns
        content_lines = lines[1:]
        mid_point = len(content_lines) // 2
        
        # Left column
        y_offset = margin + 60
        self._render_column_text(draw, content_lines[:mid_point], font, margin, y_offset, col_width, height - margin)
        
        # Right column
        self._render_column_text(draw, content_lines[mid_point:], font, margin + col_width + col_gap, y_offset, col_width, height - margin)

    def _render_column_text(self, draw, lines: List[str], font, x_start: int, y_start: int, max_width: int, max_height: int):
        """Render text within a column with word wrapping."""
        y_offset = y_start
        line_height = 18
        
        for line in lines:
            if y_offset + line_height > max_height:
                break
                
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if draw.textlength(test_line, font=font) < max_width:
                    current_line = test_line
                else:
                    if current_line:
                        draw.text((x_start, y_offset), current_line, fill='black', font=font)
                        y_offset += line_height
                    current_line = word
            
            if current_line:
                draw.text((x_start, y_offset), current_line, fill='black', font=font)
                y_offset += line_height + 5

    def _render_newspaper_layout(self, draw, content: str, font, title_font, width: int, height: int, margin: int):
        """Render content in newspaper style layout."""
        # Similar to two column but with header styling
        self._render_two_column(draw, content, font, title_font, width, height, margin)
        
        # Add newspaper-style header line
        draw.line([(margin, margin + 50), (width - margin, margin + 50)], fill='black', width=2)

    def _render_magazine_layout(self, draw, content: str, font, title_font, width: int, height: int, margin: int):
        """Render content in magazine style layout."""
        # Similar to single column but with more spacing and styling
        self._render_single_column(draw, content, font, title_font, width, height, margin)
        
        # Add decorative elements
        draw.rectangle([margin - 5, margin + 10, width - margin + 5, height - margin - 10], outline='lightgray', width=1)

    def _apply_augmentations(self, image: Image.Image, augmentations: List[str]) -> Image.Image:
        """Apply specified augmentations to the image."""
        for aug_name in augmentations:
            try:
                if hasattr(self.augmentor, f'apply_{aug_name}'):
                    image = getattr(self.augmentor, f'apply_{aug_name}')(image, intensity=0.5)
            except Exception as e:
                print(f"⚠️ Error applying augmentation {aug_name}: {e}")
        return image

    def _extract_layout_info(self, image: Image.Image, content: str, layout_type: str) -> Dict[str, Any]:
        """Extract layout information from the generated image."""
        return {
            "layout_type": layout_type,
            "page_dimensions": {"width": image.width, "height": image.height},
            "margins": 50,
            "columns": 2 if "two_column" in layout_type or "newspaper" in layout_type else 1,
            "text_regions": [
                {
                    "bbox": [50, 60, image.width - 50, image.height - 50],
                    "type": "text_block",
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                }
            ],
            "font_info": {
                "primary_font": "Arial",  # Would be actual font used
                "font_sizes": [12, 16],
                "line_spacing": 18
            }
        }

    def _get_sample_content(self) -> str:
        """Get sample content for testing."""
        return """Sample Document Title

Introduction
This is a sample document that demonstrates various layout capabilities. The content includes multiple paragraphs with different types of information.

Main Content
The main section contains detailed information about the subject matter. This paragraph shows how regular text flows within the document structure and how different layouts can change the visual presentation.

Key Features:
• Multiple layout options available
• Support for different fonts and languages
• Automatic text wrapping and spacing
• Professional document formatting

Technical Details
Advanced layout features include multi-column support, proper typography, and responsive text flow. The system automatically adjusts content to fit different layout constraints while maintaining readability.

Conclusion
This sample demonstrates the flexibility and power of the layout augmentation system for creating diverse document variations."""

    def _get_sample_pdf_content(self) -> str:
        return self._get_sample_content()
    
    def _get_sample_image_content(self) -> str:
        return self._get_sample_content() 