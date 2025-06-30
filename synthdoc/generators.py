"""
Document generators for different types of synthetic documents.

This module contains generators for raw documents, layout-based documents,
VQA datasets, and handwritten documents.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import tempfile
import base64
from io import BytesIO

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfutils
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Base class for all document generators."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate documents."""
        pass


class DocumentRenderer:
    """Handles document rendering from text to images."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.temp_dir = Path(tempfile.gettempdir()) / "synthdoc"
        self.temp_dir.mkdir(exist_ok=True)
        
    def render_text_to_image(
        self, 
        text: str, 
        width: int = 800, 
        height: int = 1000,
        font_name: str = "Arial",
        font_size: int = 12,
        language: str = "en"
    ) -> Image.Image:
        """Render text content to an image."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for document rendering. Install with: pip install Pillow")
        
        # Create blank image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to load the specified font
            font = ImageFont.truetype(font_name, font_size)
        except (OSError, IOError):
            # Fallback to default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Split text into lines and render
        lines = text.split('\n')
        y_offset = 50
        line_height = font_size + 5 if font else 15
        
        for line in lines:
            if y_offset + line_height > height - 50:
                break
                
            # Word wrap for long lines
            words = line.split(' ')
            current_line = ""
            
            for word in words:
                test_line = current_line + word + " "
                if font:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    text_width = bbox[2] - bbox[0]
                else:
                    text_width = len(test_line) * 8  # Rough estimate
                
                if text_width <= width - 100:
                    current_line = test_line
                else:
                    if current_line:
                        draw.text((50, y_offset), current_line.strip(), fill='black', font=font)
                        y_offset += line_height
                    current_line = word + " "
            
            if current_line:
                draw.text((50, y_offset), current_line.strip(), fill='black', font=font)
                y_offset += line_height
        
        return image
    
    def render_text_to_pdf(
        self, 
        text: str, 
        output_path: Path,
        font_name: str = "Helvetica",
        font_size: int = 12
    ) -> Path:
        """Render text content to PDF."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF rendering. Install with: pip install reportlab")
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Create custom style
        custom_style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontSize=font_size,
            fontName=font_name,
            leading=font_size * 1.2
        )
        
        # Split text into paragraphs and add to story
        paragraphs = text.split('\n\n')
        for para_text in paragraphs:
            if para_text.strip():
                para = Paragraph(para_text.strip(), custom_style)
                story.append(para)
                story.append(Spacer(1, 12))
        
        doc.build(story)
        return output_path


class DocumentGenerator(BaseGenerator):
    """Generator for raw document content using LLMs."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize DocumentGenerator.

        Args:
            model: Model name for LiteLLM (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "groq/llama-3.1-8b-instant")
            api_key: API key for the model provider
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.renderer = DocumentRenderer()

        if not LITELLM_AVAILABLE:
            self.logger.warning(
                "LiteLLM not available. Install with: pip install litellm"
            )
            self._llm_enabled = False
        else:
            self._llm_enabled = True
            if api_key:
                # Set API key if provided
                import os
                if "gpt" in model.lower() or "openai" in model.lower():
                    os.environ["OPENAI_API_KEY"] = api_key
                elif "claude" in model.lower() or "anthropic" in model.lower():
                    os.environ["ANTHROPIC_API_KEY"] = api_key

    def generate_raw_documents(
        self,
        language: str,
        num_pages: int,
        prompt: Optional[str] = None,
        augmentations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate raw documents from scratch.

        Args:
            language: Target language code
            num_pages: Number of pages to generate
            prompt: Custom prompt for content generation
            augmentations: List of augmentation techniques

        Returns:
            List of generated documents
        """
        self.logger.info(f"Generating {num_pages} documents in {language}")

        documents = []
        for i in range(num_pages):
            # Generate content using LLM
            content = self._generate_content(language, prompt)
            
            # Render to image
            try:
                image = self.renderer.render_text_to_image(
                    content, 
                    language=language,
                    font_size=14
                )
                
                # Save image temporarily
                temp_image_path = self.renderer.temp_dir / f"doc_{i:04d}.png"
                image.save(temp_image_path)
                
                doc = {
                    "id": f"doc_{i:04d}",
                    "language": language,
                    "content": content,
                    "image_path": str(temp_image_path),
                    "image": image,
                    "metadata": {
                        "page_number": i,
                        "generation_prompt": prompt,
                        "augmentations": augmentations or [],
                        "image_width": image.width,
                        "image_height": image.height,
                        "pdf_name": f"generated_doc_{i}",
                        "markdown": content,
                        "html": f"<p>{content.replace(chr(10), '</p><p>')}</p>",
                        "page_size": (image.width, image.height),
                        "content_list": content.split('\n'),
                    },
                }
                documents.append(doc)
                
            except Exception as e:
                self.logger.error(f"Failed to render document {i}: {e}")
                # Create a fallback document
                doc = {
                    "id": f"doc_{i:04d}",
                    "language": language,
                    "content": content,
                    "image_path": None,
                    "image": None,
                    "metadata": {
                        "page_number": i,
                        "generation_prompt": prompt,
                        "augmentations": augmentations or [],
                        "error": str(e),
                    },
                }
                documents.append(doc)

        return documents

    def generate_handwriting(
        self,
        content: Optional[str],
        language: str,
        handwriting_template: Optional[str],
        writing_style: str,
        paper_template: str,
    ) -> Dict[str, Any]:
        """
        Generate handwritten documents.

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style

        Returns:
            Handwritten document dataset
        """
        self.logger.info(f"Generating handwritten document in {language}")

        if not content:
            content = self._generate_content(language)
        
        # Create handwritten-style image
        try:
            image = self._render_handwriting(
                content, 
                language, 
                writing_style, 
                paper_template
            )
            
            temp_image_path = self.renderer.temp_dir / f"handwritten_{language}.png"
            image.save(temp_image_path)
            
            return {
                "image": image,
                "image_path": str(temp_image_path),
                "content": content,
                "style": writing_style,
                "template": handwriting_template,
                "paper": paper_template,
                "language": language,
                "metadata": {
                    "document_type": "handwritten",
                    "writing_style": writing_style,
                    "paper_template": paper_template,
                    "image_width": image.width,
                    "image_height": image.height,
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to generate handwriting: {e}")
            return {
                "image": None,
                "content": content,
                "style": writing_style,
                "template": handwriting_template,
                "paper": paper_template,
                "language": language,
                "error": str(e),
            }

    def _render_handwriting(
        self, 
        text: str, 
        language: str, 
        style: str, 
        paper: str
    ) -> Image.Image:
        """Render text in handwriting style."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for handwriting rendering")
        
        # Create paper background
        width, height = 800, 1000
        if paper == "lined":
            image = self._create_lined_paper(width, height)
        elif paper == "grid":
            image = self._create_grid_paper(width, height)
        else:
            image = Image.new('RGB', (width, height), color='white')
        
        draw = ImageDraw.Draw(image)
        
        # Use a more handwriting-like font if available
        try:
            if style == "cursive":
                font = ImageFont.truetype("Comic Sans MS", 16)
            else:
                font = ImageFont.truetype("Arial", 14)
        except:
            font = ImageFont.load_default()
        
        # Add slight randomness to simulate handwriting
        import random
        lines = text.split('\n')
        y_offset = 80
        line_height = 25
        
        for line in lines:
            if y_offset + line_height > height - 50:
                break
            
            # Add some horizontal variation
            x_offset = 60 + random.randint(-10, 10)
            y_position = y_offset + random.randint(-3, 3)
            
            draw.text((x_offset, y_position), line, fill='blue', font=font)
            y_offset += line_height + random.randint(-2, 5)
        
        return image
    
    def _create_lined_paper(self, width: int, height: int) -> Image.Image:
        """Create lined paper background."""
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw horizontal lines
        for y in range(80, height - 50, 25):
            draw.line([(50, y), (width - 50, y)], fill='lightblue', width=1)
        
        # Draw margin line
        draw.line([(80, 50), (80, height - 50)], fill='red', width=2)
        
        return image
    
    def _create_grid_paper(self, width: int, height: int) -> Image.Image:
        """Create grid paper background."""
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw grid
        for x in range(50, width - 50, 25):
            draw.line([(x, 50), (x, height - 50)], fill='lightgray', width=1)
        
        for y in range(50, height - 50, 25):
            draw.line([(50, y), (width - 50, y)], fill='lightgray', width=1)
        
        return image

    def _generate_content(self, language: str, prompt: Optional[str] = None) -> str:
        """Generate text content for documents using LiteLLM."""
        if not self._llm_enabled:
            # Fallback to sample content if LLM not available
            if prompt:
                return f"Generated content based on: {prompt}\n\nLanguage: {language}\n\nThis is sample document content that demonstrates the basic structure and layout of a document. It includes multiple paragraphs to showcase text flow and formatting capabilities.\n\nSecond paragraph with more content to fill the page appropriately. This content can be used for testing layout and rendering functionality."
            else:
                return self._get_fallback_content(language)

        try:
            # Create system prompt for document generation
            system_prompt = f"""You are a document content generator. Generate realistic document content in {language}. 
            The content should be appropriate for training document understanding models and include various text structures like:
            - Headers and subheaders
            - Paragraphs of varying lengths  
            - Lists (numbered and bulleted)
            - Technical terms when appropriate
            - Natural language that would appear in real documents
            
            Make the content focused, coherent, and between 200-500 words. Format it with proper line breaks for readability."""

            # Create user prompt
            if prompt:
                user_prompt = f"Generate document content based on this request: {prompt}"
            else:
                user_prompt = f"Generate diverse, realistic document content in {language} that would be suitable for training document understanding models. Include headers, paragraphs, and lists."

            # Call LiteLLM
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}. Using fallback content.")
            return self._get_fallback_content(language)
    
    def _get_fallback_content(self, language: str) -> str:
        """Get fallback content for different languages."""
        fallback_templates = {
            "en": """Document Title: Sample Technical Report

Introduction
This document demonstrates the basic structure of a technical report. It includes multiple sections with varying content types to showcase document layout and formatting.

Main Content
The main section contains detailed information about the subject matter. This paragraph shows how regular text flows within the document structure.

Key Points:
• First important point
• Second critical consideration  
• Third essential element

Numbered List:
1. Primary objective
2. Secondary goal
3. Final outcome

Conclusion
This concludes the sample document content, providing sufficient text for layout testing and document understanding model training.""",
            
            "hi": """दस्तावेज़ शीर्षक: नमूना तकनीकी रिपोर्ट

प्रस्तावना
यह दस्तावेज़ एक तकनीकी रिपोर्ट की मूल संरचना प्रदर्शित करता है। इसमें दस्तावेज़ लेआउट और फॉर्मेटिंग दिखाने के लिए विभिन्न सामग्री प्रकारों के साथ कई खंड शामिल हैं।

मुख्य सामग्री
मुख्य खंड में विषय वस्तु के बारे में विस्तृत जानकारी होती है। यह पैराग्राफ दिखाता है कि दस्तावेज़ संरचना के भीतर नियमित पाठ कैसे प्रवाहित होता है।

मुख्य बिंदु:
• पहला महत्वपूर्ण बिंदु
• दूसरा महत्वपूर्ण विचार
• तीसरा आवश्यक तत्व

संख्यित सूची:
1. प्राथमिक उद्देश्य
2. द्वितीयक लक्ष्य
3. अंतिम परिणाम

निष्कर्ष
यह नमूना दस्तावेज़ सामग्री का समापन करता है, लेआउट परीक्षण और दस्तावेज़ समझ मॉडल प्रशिक्षण के लिए पर्याप्त पाठ प्रदान करता है।""",
            
            "zh": """文档标题：示例技术报告

简介
本文档展示了技术报告的基本结构。它包含多个部分，具有不同的内容类型，以展示文档布局和格式。

主要内容
主要部分包含有关主题的详细信息。本段展示了常规文本如何在文档结构中流动。

要点：
• 第一个重要点
• 第二个关键考虑
• 第三个基本要素

编号列表：
1. 主要目标
2. 次要目标
3. 最终结果

结论
这总结了示例文档内容，为布局测试和文档理解模型训练提供了足够的文本。""",
        }
        
        return fallback_templates.get(language, fallback_templates["en"])

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.generate_raw_documents(*args, **kwargs)


class LayoutGenerator(BaseGenerator):
    """Generator for layout-based document transformations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.renderer = DocumentRenderer()

    def augment_layouts(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        document_paths: Optional[List[Union[str, Path]]] = None,
        languages: List[str] = None,
        fonts: Optional[List[str]] = None,
        augmentations: List[str] = None,
        layout_templates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply layout transformations to documents.

        Args:
            documents: Pre-generated documents
            document_paths: Paths to existing documents
            languages: Target languages
            fonts: Font families to apply
            augmentations: Visual augmentation techniques
            layout_templates: Predefined layout templates

        Returns:
            HuggingFace dataset with layout annotations
        """
        self.logger.info("Starting layout augmentation")

        dataset_items = []
        
        # Process existing documents
        if documents:
            for doc in documents:
                variations = self._create_layout_variations(
                    doc, fonts or ["Arial"], augmentations or []
                )
                dataset_items.extend(variations)
        
        # Process documents from paths
        if document_paths:
            for doc_path in document_paths:
                try:
                    variations = self._process_document_file(
                        doc_path, languages or ["en"], fonts or ["Arial"], augmentations or []
                    )
                    dataset_items.extend(variations)
                except Exception as e:
                    self.logger.error(f"Failed to process {doc_path}: {e}")

        # Format as HuggingFace dataset
        return self._format_as_hf_dataset(dataset_items)

    def _create_layout_variations(
        self, 
        document: Dict[str, Any], 
        fonts: List[str], 
        augmentations: List[str]
    ) -> List[Dict[str, Any]]:
        """Create layout variations of a document."""
        variations = []
        content = document.get("content", "")
        
        for font in fonts:
            try:
                # Re-render with different font
                image = self.renderer.render_text_to_image(
                    content,
                    font_name=font,
                    font_size=14,
                    language=document.get("language", "en")
                )
                
                # Apply augmentations
                for aug in augmentations:
                    augmented_image = self._apply_augmentation(image, aug)
                    
                    # Save augmented image
                    temp_path = self.renderer.temp_dir / f"layout_{font}_{aug}_{len(variations)}.png"
                    augmented_image.save(temp_path)
                    
                    variation = {
                        "image": augmented_image,
                        "image_path": str(temp_path),
                        "image_width": augmented_image.width,
                        "pdf_name": f"{document.get('id', 'doc')}_{font}_{aug}",
                        "page_number": 0,
                        "markdown": content,
                        "html": f"<p>{content.replace(chr(10), '</p><p>')}</p>",
                        "layout": self._extract_layout_info(augmented_image, content),
                        "lines": self._extract_line_info(content),
                        "images": [],
                        "equations": [],
                        "tables": [],
                        "page_size": (augmented_image.width, augmented_image.height),
                        "content_list": content.split('\n'),
                        "base_layout_detection": {},
                        "pdf_info": {"font": font, "augmentation": aug},
                        "metadata": {
                            "font": font,
                            "augmentation": aug,
                            "original_id": document.get("id"),
                        }
                    }
                    variations.append(variation)
                    
            except Exception as e:
                self.logger.error(f"Failed to create variation with font {font}: {e}")
        
        return variations

    def _process_document_file(
        self, 
        file_path: Union[str, Path], 
        languages: List[str], 
        fonts: List[str],
        augmentations: List[str]
    ) -> List[Dict[str, Any]]:
        """Process a document file and create variations."""
        # For now, create a placeholder implementation
        # In a full implementation, this would use PDF processing libraries
        file_path = Path(file_path)
        
        # Extract text content (placeholder)
        content = f"Content extracted from {file_path.name}\n\nThis is a placeholder for actual PDF text extraction functionality."
        
        # Create document structure
        doc = {
            "id": file_path.stem,
            "content": content,
            "language": languages[0] if languages else "en",
        }
        
        return self._create_layout_variations(doc, fonts, augmentations)

    def _apply_augmentation(self, image: Image.Image, augmentation: str) -> Image.Image:
        """Apply a specific augmentation to an image."""
        if not PIL_AVAILABLE:
            return image
        
        try:
            if augmentation == "rotation":
                return image.rotate(np.random.uniform(-5, 5), expand=True, fillcolor='white')
            elif augmentation == "scaling":
                scale = np.random.uniform(0.9, 1.1)
                new_size = (int(image.width * scale), int(image.height * scale))
                return image.resize(new_size, Image.LANCZOS)
            elif augmentation == "brightness":
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(image)
                factor = np.random.uniform(0.8, 1.2)
                return enhancer.enhance(factor)
            elif augmentation == "noise":
                # Add some noise
                img_array = np.array(image)
                noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
                noisy_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_array)
            else:
                return image
        except Exception as e:
            self.logger.error(f"Failed to apply augmentation {augmentation}: {e}")
            return image

    def _extract_layout_info(self, image: Image.Image, content: str) -> Dict[str, Any]:
        """Extract layout information from the document."""
        # Placeholder implementation
        lines = content.split('\n')
        return {
            "text_blocks": len([line for line in lines if line.strip()]),
            "total_lines": len(lines),
            "estimated_paragraphs": len([line for line in lines if line.strip() == ""]) + 1,
        }

    def _extract_line_info(self, content: str) -> List[Dict[str, Any]]:
        """Extract line-level information."""
        lines = content.split('\n')
        line_info = []
        
        for i, line in enumerate(lines):
            if line.strip():
                line_info.append({
                    "line_number": i,
                    "text": line.strip(),
                    "bbox": [50, 50 + i * 20, 750, 70 + i * 20],  # Estimated bbox
                })
        
        return line_info

    def _format_as_hf_dataset(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format items as HuggingFace dataset."""
        if not items:
            return {"images": [], "annotations": []}
        
        return {
            "images": [item.get("image") for item in items],
            "image_paths": [item.get("image_path") for item in items],
            "annotations": [
                {
                    "image_width": item.get("image_width"),
                    "pdf_name": item.get("pdf_name"),
                    "page_number": item.get("page_number"),
                    "markdown": item.get("markdown"),
                    "html": item.get("html"),
                    "layout": item.get("layout"),
                    "lines": item.get("lines"),
                    "metadata": item.get("metadata"),
                }
                for item in items
            ]
        }

    def augment_pdfs(
        self,
        corpus_paths: List[Union[str, Path]],
        extraction_elements: List[str],
        combination_strategy: str,
        output_layout_types: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Create new documents by recombining elements from existing documents.

        Args:
            corpus_paths: Paths to source documents
            extraction_elements: Types of elements to extract
            combination_strategy: Method for combining elements
            output_layout_types: Target layout styles

        Returns:
            HuggingFace dataset with recombined documents
        """
        self.logger.info(f"Starting PDF augmentation with {len(corpus_paths)} documents")

        # For now, return a placeholder implementation
        # In a full implementation, this would extract and recombine document elements
        return {
            "images": [],
            "annotations": [],
            "metadata": {
                "corpus_size": len(corpus_paths),
                "extraction_elements": extraction_elements,
                "combination_strategy": combination_strategy,
                "note": "PDF augmentation implementation in progress"
            }
        }

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.augment_layouts(*args, **kwargs)


class VQAGenerator(BaseGenerator):
    """Generator for Visual Question Answering datasets."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize VQAGenerator.

        Args:
            model: Model name for LiteLLM
            api_key: API key for the model provider
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        
        if not LITELLM_AVAILABLE:
            self.logger.warning("LiteLLM not available. VQA generation will be limited.")
            self._llm_enabled = False
        else:
            self._llm_enabled = True
            if api_key:
                import os
                if "gpt" in model.lower() or "openai" in model.lower():
                    os.environ["OPENAI_API_KEY"] = api_key
                elif "claude" in model.lower() or "anthropic" in model.lower():
                    os.environ["ANTHROPIC_API_KEY"] = api_key

    def generate_vqa_dataset(
        self,
        documents: List[Dict[str, Any]],
        question_types: List[str],
        difficulty_levels: List[str],
        hard_negative_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate VQA datasets with hard negatives.

        Args:
            documents: Documents to generate questions about
            question_types: Types of questions to generate
            difficulty_levels: Question complexity levels
            hard_negative_ratio: Ratio of hard negative examples

        Returns:
            Extended dataset with VQA annotations
        """
        self.logger.info(f"Generating VQA dataset for {len(documents)} documents")

        vqa_pairs = []
        
        for doc in documents:
            questions = self._generate_questions(doc, question_types, difficulty_levels)
            
            for question_data in questions:
                # Generate correct answer
                answer = self._generate_answer(doc, question_data["question"])
                
                # Generate hard negatives
                hard_negatives = []
                if hard_negative_ratio > 0:
                    num_negatives = int(hard_negative_ratio * 10)  # Scale to reasonable number
                    hard_negatives = self._generate_hard_negatives(
                        doc, question_data["question"], answer, num_negatives
                    )
                
                vqa_pair = {
                    "document_id": doc.get("id"),
                    "image": doc.get("image"),
                    "image_path": doc.get("image_path"),
                    "question": question_data["question"],
                    "answer": answer,
                    "question_type": question_data["type"],
                    "difficulty": question_data["difficulty"],
                    "hard_negatives": hard_negatives,
                    "similarity_scores": self._calculate_similarity_scores(answer, hard_negatives),
                }
                vqa_pairs.append(vqa_pair)

        return {
            "questions": [pair["question"] for pair in vqa_pairs],
            "answers": [pair["answer"] for pair in vqa_pairs],
            "hard_negatives": [pair["hard_negatives"] for pair in vqa_pairs],
            "question_types": [pair["question_type"] for pair in vqa_pairs],
            "difficulty_scores": [pair["difficulty"] for pair in vqa_pairs],
            "similarity_scores": [pair["similarity_scores"] for pair in vqa_pairs],
            "images": [pair["image"] for pair in vqa_pairs],
            "metadata": {
                "total_pairs": len(vqa_pairs),
                "question_types": question_types,
                "difficulty_levels": difficulty_levels,
                "hard_negative_ratio": hard_negative_ratio,
            }
        }

    def _generate_questions(
        self,
        document: Dict[str, Any],
        question_types: List[str],
        difficulty_levels: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate questions about a document."""
        content = document.get("content", "")
        
        if not self._llm_enabled:
            return self._generate_fallback_questions(content, question_types, difficulty_levels)
        
        questions = []
        
        for q_type in question_types:
            for difficulty in difficulty_levels:
                try:
                    question = self._generate_single_question(content, q_type, difficulty)
                    questions.append({
                        "question": question,
                        "type": q_type,
                        "difficulty": difficulty,
                    })
                except Exception as e:
                    self.logger.error(f"Failed to generate {q_type} question: {e}")
        
        return questions

    def _generate_single_question(self, content: str, question_type: str, difficulty: str) -> str:
        """Generate a single question using LLM."""
        system_prompt = f"""You are a question generator for document understanding tasks. 
        Generate a {question_type} question at {difficulty} difficulty level about the given document content.
        
        Question types:
        - factual: Questions about specific facts mentioned in the document
        - reasoning: Questions requiring inference or logical reasoning
        - comparative: Questions comparing different elements
        - structural: Questions about document layout or organization
        
        Keep questions clear, concise, and answerable from the document content."""
        
        user_prompt = f"""Generate a {question_type} question at {difficulty} difficulty about this document:

{content}

Return only the question, no explanations."""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"LLM question generation failed: {e}")
            return self._get_fallback_question(question_type, difficulty)

    def _generate_answer(self, document: Dict[str, Any], question: str) -> str:
        """Generate an answer for a question about the document."""
        content = document.get("content", "")
        
        if not self._llm_enabled:
            return "Sample answer based on the document content."
        
        try:
            system_prompt = """You are answering questions about documents. Provide accurate, concise answers based only on the information in the document. If the information is not in the document, say so."""
            
            user_prompt = f"""Document content:
{content}

Question: {question}

Answer:"""

            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer."

    def _generate_hard_negatives(
        self, 
        document: Dict[str, Any], 
        question: str, 
        correct_answer: str, 
        num_negatives: int
    ) -> List[str]:
        """Generate hard negative answers."""
        if not self._llm_enabled:
            return ["Incorrect answer 1", "Incorrect answer 2"]
        
        negatives = []
        
        try:
            system_prompt = """Generate plausible but incorrect answers to the given question. The answers should be semantically similar to the correct answer but factually wrong or misleading."""
            
            user_prompt = f"""Question: {question}
Correct answer: {correct_answer}
Document content: {document.get('content', '')}

Generate {num_negatives} plausible but incorrect answers. Return one answer per line."""

            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.8,
            )
            
            content = response.choices[0].message.content.strip()
            negatives = [line.strip() for line in content.split('\n') if line.strip()]
            
        except Exception as e:
            self.logger.error(f"Hard negative generation failed: {e}")
            negatives = [f"Incorrect answer {i+1}" for i in range(num_negatives)]
        
        return negatives[:num_negatives]

    def _calculate_similarity_scores(self, correct_answer: str, negatives: List[str]) -> List[float]:
        """Calculate similarity scores between correct answer and negatives."""
        # Simple similarity based on word overlap (could be improved with embeddings)
        correct_words = set(correct_answer.lower().split())
        scores = []
        
        for negative in negatives:
            negative_words = set(negative.lower().split())
            if not correct_words or not negative_words:
                scores.append(0.0)
            else:
                overlap = len(correct_words.intersection(negative_words))
                total = len(correct_words.union(negative_words))
                similarity = overlap / total if total > 0 else 0.0
                scores.append(similarity)
        
        return scores

    def _generate_fallback_questions(
        self, 
        content: str, 
        question_types: List[str], 
        difficulty_levels: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate fallback questions when LLM is not available."""
        questions = []
        
        for q_type in question_types:
            for difficulty in difficulty_levels:
                question = self._get_fallback_question(q_type, difficulty)
                questions.append({
                    "question": question,
                    "type": q_type,
                    "difficulty": difficulty,
                })
        
        return questions

    def _get_fallback_question(self, question_type: str, difficulty: str) -> str:
        """Get a fallback question for a given type and difficulty."""
        fallback_questions = {
            "factual": {
                "easy": "What is the main topic of this document?",
                "medium": "What specific information is provided about the main topic?",
                "hard": "What are the detailed characteristics mentioned in the document?"
            },
            "reasoning": {
                "easy": "Why is this document important?",
                "medium": "What can be inferred from the information presented?",
                "hard": "What logical conclusions can be drawn from the evidence provided?"
            },
            "comparative": {
                "easy": "How do the different sections compare?",
                "medium": "What are the similarities and differences between the key points?",
                "hard": "How do the various elements relate to each other structurally?"
            }
        }
        
        return fallback_questions.get(question_type, {}).get(difficulty, "What is this document about?")

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.generate_vqa_dataset(*args, **kwargs)
