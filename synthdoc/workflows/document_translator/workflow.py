"""
Document Translation Workflow for SynthDoc

This workflow uses YOLO layout detection, OCR, and translation to convert documents
to different languages while preserving the original layout and formatting.
"""

import os
import sys
import json
import time
import base64
import random
import glob
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset

# Optional imports with fallbacks
try:
    from doclayout_yolo import YOLOv10
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  doclayout_yolo not available - document translation will use fallback mode")

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"tesseract"
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not available - OCR will be disabled")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️  deep_translator not available - translation will be disabled")

from ..base import BaseWorkflow
from ...models import DocumentTranslationConfig, WorkflowResult


class ImageTranslator:
    """
    Enhanced ImageTranslator class integrated with SynthDoc.
    """
    
    def __init__(
        self,
        model_path: str,
        font_path: str,
        langs: List[str] = ["hi"],
        conf: float = 0.4,
        imgsz: int = 1024,
        logger: Optional[logging.Logger] = None
    ):
        self.model_path = model_path
        self.font_path = font_path
        self.langs = langs
        self.conf = conf
        self.imgsz = imgsz
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize outputs for each language
        self.output = {lang: {} for lang in self.langs}
        
        # Initialize translators
        if TRANSLATOR_AVAILABLE:
            self.translators = {
                lang: GoogleTranslator(source="auto", target=lang) 
                for lang in self.langs
            }
        else:
            self.translators = {}
            self.logger.warning("Translation not available - will preserve original text")
        
        # Layout class mapping
        self.class_mapping = {
            "plain text": "text",
            "title": "title", 
            "figure": "image",
            "isolate_formula": "formula",
            "figure_caption": "caption",
            "table": "table",
        }
        self.translatable_classes = ["plain text", "title", "figure_caption"]
    
    def get_background_color(self, image_region):
        """Get most frequent color from the image region"""
        pixels = image_region.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_frequent_color = unique_colors[counts.argmax()]
        return tuple(map(int, most_frequent_color))
    
    def get_text_color(self, bg_color):
        """Determine text color using weighted luminance calculation"""
        weights = np.array([0.299, 0.587, 0.114])
        luminance = np.sum(np.array(bg_color) * weights)
        return (0, 0, 0) if luminance > 127 else (255, 255, 255)
    
    def get_random_font(self, lang_code: str) -> str:
        """Select a random font file from the language-specific font directory."""
        lang_font_dir = os.path.join(self.font_path, lang_code)
        
        if not os.path.exists(lang_font_dir):
            self.logger.warning(f"Font directory not found: {lang_font_dir}")
            # Fallback to default font
            return None
            
        font_extensions = (".ttf", ".otf", ".TTF", ".OTF")
        font_files = []
        for ext in font_extensions:
            font_files.extend(glob.glob(os.path.join(lang_font_dir, f"*{ext}")))
            
        if not font_files:
            self.logger.warning(f"No font files found in directory: {lang_font_dir}")
            return None
            
        return random.choice(font_files)
    
    def wrap_text_for_box(self, text, box_width, font_path, font_size):
        """Wrap text for a box based on the font size and box width."""
        if not font_path:
            return text  # Return original if no font available
            
        try:
            font = ImageFont.truetype(font_path, font_size, layout_engine="raqm")
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
            
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            try:
                line_bbox = font.getbbox(test_line)
                line_width = line_bbox[2] - line_bbox[0]
            except Exception:
                line_width = len(test_line) * font_size * 0.6  # Rough estimate
                
            if line_width <= box_width * 0.95:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def fit_text_in_box(self, text, box_width, box_height, font_path, max_font_size=100, min_font_size=10):
        """Find the optimal font size to fit text in a given box using binary search."""
        if not font_path:
            # Fallback with default font
            try:
                font = ImageFont.load_default()
                lines = text.split('\n') if '\n' in text else [text]
                return font, lines
            except Exception:
                return None, [text]
        
        lower = min_font_size
        upper = max_font_size
        optimal_font = None
        optimal_lines = []
        
        while lower <= upper:
            mid = (lower + upper) // 2
            wrapped_text = self.wrap_text_for_box(text, box_width, font_path, mid)
            lines = wrapped_text.split("\n")
            
            try:
                font = ImageFont.truetype(font_path, mid, layout_engine="raqm")
            except Exception:
                font = ImageFont.load_default()
            
            line_widths = []
            total_height = 0
            for line in lines:
                try:
                    line_bbox = font.getbbox(line)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_height = line_bbox[3] - line_bbox[1]
                except Exception:
                    line_width = len(line) * mid * 0.6
                    line_height = mid
                    
                line_widths.append(line_width)
                total_height += line_height * 1.2
            
            max_line_width = max(line_widths) if line_widths else 0
            
            if max_line_width <= box_width * 0.95 and total_height <= box_height * 0.95:
                optimal_font = font
                optimal_lines = lines
                lower = mid + 1
            else:
                upper = mid - 1
        
        if not optimal_font:
            try:
                optimal_font = ImageFont.truetype(font_path, min_font_size, layout_engine="raqm")
            except Exception:
                optimal_font = ImageFont.load_default()
            optimal_lines = self.wrap_text_for_box(text, box_width, font_path, min_font_size).split("\n")
        
        return optimal_font, optimal_lines
    
    def detect_and_ocr_image(self, cv2_image, results):
        """Process layout detection and OCR for an image"""
        regions = []
        
        for i, box in enumerate(results.boxes):
            bbox = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            if class_name not in self.class_mapping:
                continue
            
            mapped_class_name = self.class_mapping[class_name]
            region_data = {
                "region_id": i + 1,
                "layout_type": mapped_class_name,
                "bbox": {
                    "xmin": int(xmin),
                    "ymin": int(ymin),
                    "xmax": int(xmax),
                    "ymax": int(ymax),
                },
            }
            
            if class_name in self.translatable_classes and TESSERACT_AVAILABLE:
                roi = cv2_image[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue
                
                try:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(binary, config="--oem 3 --psm 6").strip()
                    
                    if text:
                        region_data["english_text"] = text
                        regions.append(region_data)
                except Exception as e:
                    self.logger.warning(f"OCR failed for region {i}: {e}")
                    continue
            else:
                # Include non-text regions in output
                regions.append(region_data)
        
        return regions
    
    def process_single_image(self, idx: int, input_img: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Process a single image with optimized language handling"""
        image = input_img if isinstance(input_img, Image.Image) else Image.fromarray(input_img)
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - cannot perform layout detection")
            return {"error": "YOLO model not available"}
        
        try:
            model = YOLOv10(self.model_path)
            det_res = model.predict(cv2_image, imgsz=self.imgsz, conf=self.conf, device="cpu")
            regions = self.detect_and_ocr_image(cv2_image, det_res[0])
        except Exception as e:
            self.logger.error(f"Layout detection failed: {e}")
            return {"error": f"Layout detection failed: {e}"}
        
        results = {}
        
        for lang in self.langs:
            try:
                selected_font_path = self.get_random_font(lang)
                pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                translation_data = {
                    "image_id": f"image_{idx}",
                    "target_language": lang,
                    "font_used": os.path.basename(selected_font_path) if selected_font_path else "default",
                    "regions": [],
                }
                
                bg_color = self.get_background_color(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
                text_color = self.get_text_color(bg_color)
                
                for region in regions:
                    bbox = region["bbox"]
                    translated_region = region.copy()
                    
                    if (region["layout_type"] in ["text", "title", "caption"] and 
                        "english_text" in region and 
                        TRANSLATOR_AVAILABLE and 
                        lang in self.translators):
                        
                        try:
                            translated_text = self.translators[lang].translate(region["english_text"])
                            translated_region["translated_text"] = translated_text
                            
                            font, text_lines = self.fit_text_in_box(
                                translated_text,
                                bbox["xmax"] - bbox["xmin"],
                                bbox["ymax"] - bbox["ymin"],
                                selected_font_path,
                            )
                            
                            if font and text_lines:
                                # Clear the original text area
                                draw.rectangle(
                                    [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]],
                                    fill=bg_color,
                                )
                                
                                # Draw translated text
                                current_y = bbox["ymin"]
                                for line in text_lines:
                                    try:
                                        line_bbox = font.getbbox(line)
                                        line_height = line_bbox[3] - line_bbox[1]
                                    except Exception:
                                        line_height = font.size
                                        
                                    draw.text(
                                        (bbox["xmin"], current_y),
                                        line,
                                        font=font,
                                        fill=text_color,
                                    )
                                    current_y += line_height * 1.2
                                    
                        except Exception as e:
                            self.logger.warning(f"Translation error for region {region['region_id']}: {e}")
                            # Keep original text if translation fails
                            translated_region["translated_text"] = region.get("english_text", "")
                    
                    translation_data["regions"].append(translated_region)
                
                # Convert image to base64
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                translation_data["translated_image"] = img_base64
                
                results[lang] = translation_data
                
            except Exception as e:
                self.logger.error(f"Error processing image {idx} for language {lang}: {e}")
                results[lang] = {"error": str(e)}
        
        return results


class DocumentTranslator(BaseWorkflow):
    """
    Document Translation Workflow for SynthDoc.
    
    Translates documents to different languages while preserving layout using:
    - YOLO for layout detection
    - OCR for text extraction  
    - Translation APIs for text conversion
    - Smart font rendering for target languages
    """
    
    def __init__(self, save_dir: str = "document_translation_output"):
        super().__init__(save_dir)
        self.workflow_name = "document_translation"
    
    def process(self, config: DocumentTranslationConfig) -> WorkflowResult:
        """
        Process document translation according to configuration.
        
        Args:
            config: DocumentTranslationConfig with translation parameters
            
        Returns:
            WorkflowResult with translated documents dataset
        """
        start_time = time.time()
        
        self.logger.info(f"🌍 Starting document translation to languages: {config.target_languages}")
        
        # Validate dependencies
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO model not available - cannot perform document translation")
            return WorkflowResult(
                dataset=Dataset.from_dict({}),
                metadata={
                    "workflow_type": "document_translation",
                    "status": "failed",
                    "error": "YOLO dependency not available",
                    "processing_time": time.time() - start_time
                },
                num_samples=0
            )
        
        # Initialize translator
        translator = ImageTranslator(
            model_path=config.yolo_model_path,
            font_path=config.font_path,
            langs=config.target_languages,
            conf=config.confidence_threshold,
            imgsz=config.image_size,
            logger=self.logger
        )
        
        # Process input images
        samples = []
        image_paths = self._get_image_paths(config)
        
        self.logger.info(f"📄 Processing {len(image_paths)} images...")
        
        for idx, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {Path(image_path).name}")
                
                # Load image
                image = Image.open(image_path)
                
                # Process translation
                translation_results = translator.process_single_image(idx, image)
                
                # Create samples for each target language
                for lang, result_data in translation_results.items():
                    if "error" in result_data:
                        self.logger.warning(f"Translation failed for {lang}: {result_data['error']}")
                        continue
                    
                    # Save translated image
                    if "translated_image" in result_data:
                        img_bytes = base64.b64decode(result_data["translated_image"])
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        output_filename = f"translated_{lang}_{idx}_{Path(image_path).stem}.jpg"
                        output_path = os.path.join(self.save_dir, output_filename)
                        cv2.imwrite(output_path, img)
                        
                        # Create sample entry
                        sample = {
                            "id": f"translation_{lang}_{idx}",
                            "original_image_path": str(image_path),
                            "translated_image_path": output_path,
                            "source_language": "auto-detected",
                            "target_language": lang,
                            "font_used": result_data.get("font_used", "default"),
                            "translation_regions": json.dumps(result_data.get("regions", [])),
                            "num_regions": len(result_data.get("regions", [])),
                            "processing_status": "success"
                        }
                        
                        # Add comprehensive dataset fields
                        translated_img = Image.open(output_path)
                        sample.update({
                            "image": translated_img,
                            "image_width": translated_img.width,
                            "image_height": translated_img.height,
                            "image_path": output_path,
                            "pdf_name": f"translated_doc_{lang}_{idx}",
                            "page_number": 0,
                            "markdown": self._extract_markdown_from_regions(result_data.get("regions", [])),
                            "html": self._extract_html_from_regions(result_data.get("regions", [])),
                        })
                        
                        samples.append(sample)
                
            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                # Create error sample
                samples.append({
                    "id": f"translation_error_{idx}",
                    "original_image_path": str(image_path),
                    "processing_status": "failed",
                    "error": str(e)
                })
        
        # Create comprehensive dataset
        if samples:
            # Separate successful and failed samples
            successful_samples = [s for s in samples if s.get("processing_status") == "success"]
            
            if successful_samples:
                # Extract data for comprehensive dataset
                images = [s["image"] for s in successful_samples]
                image_paths = [s["image_path"] for s in successful_samples]
                pdf_names = [s["pdf_name"] for s in successful_samples]
                page_numbers = [s["page_number"] for s in successful_samples]
                markdown_content = [s["markdown"] for s in successful_samples]
                html_content = [s["html"] for s in successful_samples]
                
                dataset = self._create_comprehensive_hf_dataset(
                    images=images,
                    image_paths=image_paths,
                    pdf_names=pdf_names,
                    page_numbers=page_numbers,
                    markdown_content=markdown_content,
                    html_content=html_content,
                    additional_metadata={
                        "workflow": "document_translation",
                        "config": config.dict(),
                        "target_languages": config.target_languages,
                        "total_samples": len(successful_samples),
                        "failed_samples": len(samples) - len(successful_samples)
                    }
                )
            else:
                dataset = Dataset.from_dict({})
        else:
            dataset = Dataset.from_dict({})
        
        # Create output files list
        output_files = [s.get("translated_image_path") for s in samples if "translated_image_path" in s]
        
        processing_time = time.time() - start_time
        
        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "document_translation",
                "target_languages": config.target_languages,
                "total_input_images": len(image_paths),
                "successful_translations": len([s for s in samples if s.get("processing_status") == "success"]),
                "failed_translations": len([s for s in samples if s.get("processing_status") == "failed"]),
                "yolo_model": config.yolo_model_path,
                "font_path": config.font_path,
                "processing_time": processing_time,
                "generated_files": [os.path.basename(f) for f in output_files]
            },
            num_samples=len([s for s in samples if s.get("processing_status") == "success"]),
            output_files=output_files
        )
    
    def _get_image_paths(self, config: DocumentTranslationConfig) -> List[str]:
        """Extract image paths from various input formats."""
        image_paths = []
        
        # Handle input images
        if config.input_images:
            for img_input in config.input_images:
                if isinstance(img_input, (str, Path)):
                    img_path = Path(img_input)
                    if img_path.is_file():
                        image_paths.append(str(img_path))
                    elif img_path.is_dir():
                        # Add all images in directory
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                            image_paths.extend(glob.glob(str(img_path / f"*{ext}")))
                            image_paths.extend(glob.glob(str(img_path / f"*{ext.upper()}")))
        
        # Handle dataset input
        if config.input_dataset:
            for sample in config.input_dataset:
                if 'image_path' in sample:
                    image_paths.append(sample['image_path'])
                elif 'image' in sample and hasattr(sample['image'], 'save'):
                    # Save PIL image to temp file
                    temp_path = os.path.join(self.save_dir, f"temp_input_{len(image_paths)}.png")
                    sample['image'].save(temp_path)
                    image_paths.append(temp_path)
        
        return image_paths
    
    def _extract_markdown_from_regions(self, regions: List[Dict]) -> str:
        """Extract markdown representation from translation regions."""
        markdown_lines = []
        
        for region in regions:
            layout_type = region.get("layout_type", "")
            text = region.get("translated_text") or region.get("english_text", "")
            
            if not text:
                continue
                
            if layout_type == "title":
                markdown_lines.append(f"# {text}")
            elif layout_type == "text":
                markdown_lines.append(text)
            elif layout_type == "caption":
                markdown_lines.append(f"*{text}*")
            elif layout_type == "formula":
                markdown_lines.append(f"$$\n{text}\n$$")
        
        return "\n\n".join(markdown_lines)
    
    def _extract_html_from_regions(self, regions: List[Dict]) -> str:
        """Extract HTML representation from translation regions."""
        html_parts = []
        
        for region in regions:
            layout_type = region.get("layout_type", "")
            text = region.get("translated_text") or region.get("english_text", "")
            
            if not text:
                continue
                
            if layout_type == "title":
                html_parts.append(f"<h1>{text}</h1>")
            elif layout_type == "text":
                html_parts.append(f"<p>{text}</p>")
            elif layout_type == "caption":
                html_parts.append(f"<em>{text}</em>")
            elif layout_type == "formula":
                html_parts.append(f"<div class='formula'>{text}</div>")
        
        return "\n".join(html_parts) 