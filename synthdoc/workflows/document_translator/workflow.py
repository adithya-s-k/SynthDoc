"""
Document Translation Workflow for SynthDoc

This workflow uses YOLO layout detection, OCR, and translation to convert documents
to different languages while preserving the original layout and formatting.

Supports input formats:
- Single image files (PNG, JPG, JPEG, TIFF, BMP)
- Single PDF files
- Folders containing images and/or PDFs
- Lists of file paths

Pipeline:
1. Convert PDFs to images (if needed)
2. YOLO layout detection to identify text regions
3. OCR text extraction from detected regions
4. Translation to target languages
5. Font rendering with appropriate fonts for target languages
6. Image reconstruction with translated text while preserving layout
"""

import os
import sys
import json
import time
import base64
import random
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

# Import SynthDoc model manager
try:
    from ...models_manager import ensure_model

    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

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
    print(
        "⚠️  doclayout_yolo not available - document translation will use fallback mode"
    )

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

# Optional PDF processing
try:
    import fitz  # PyMuPDF

    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdf2image

        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("⚠️  PDF processing not available - install PyMuPDF or pdf2image")

from ..base import BaseWorkflow
from ...models import DocumentTranslationConfig, WorkflowResult


def pdf_to_images(pdf_path: str, output_dir: str = None) -> List[str]:
    """Convert PDF to images. Returns list of image paths."""
    if not PDF_AVAILABLE:
        raise ValueError("PDF processing not available. Install PyMuPDF or pdf2image")

    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = pdf_path.parent / "temp_images"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    try:
        # Try PyMuPDF first (faster)
        import fitz

        doc = fitz.open(str(pdf_path))
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )  # 2x zoom for better quality
            img_path = output_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))
        doc.close()

    except ImportError:
        # Fallback to pdf2image
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=200)
        for i, image in enumerate(images):
            img_path = output_dir / f"{pdf_path.stem}_page_{i + 1}.png"
            image.save(str(img_path))
            image_paths.append(str(img_path))

    return image_paths


def collect_input_files(input_paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
    """
    Collect and categorize input files from various sources.
    Returns dict with 'images' and 'pdfs' lists.
    """
    images = []
    pdfs = []

    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
    pdf_extensions = {".pdf"}

    for input_path in input_paths:
        path = Path(input_path)

        if path.is_file():
            # Single file
            if path.suffix.lower() in image_extensions:
                images.append(str(path))
            elif path.suffix.lower() in pdf_extensions:
                pdfs.append(str(path))

        elif path.is_dir():
            # Directory - recursively find files
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix.lower() in image_extensions:
                        images.append(str(file_path))
                    elif file_path.suffix.lower() in pdf_extensions:
                        pdfs.append(str(file_path))

    return {"images": images, "pdfs": pdfs}


def get_background_color(image_region):
    """Get most frequent color from the image region"""
    pixels = image_region.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_frequent_color = unique_colors[counts.argmax()]
    return tuple(map(int, most_frequent_color))


def get_text_color(bg_color):
    """Determine text color using weighted luminance calculation"""
    # Weights based on human perception of colors
    weights = np.array([0.299, 0.587, 0.114])
    luminance = np.sum(np.array(bg_color) * weights)
    return (0, 0, 0) if luminance > 127 else (255, 255, 255)


def get_random_font(font_path: str, lang_code: str) -> Optional[str]:
    """
    Select a random font file from the language-specific font directory.
    Args:
        font_path: Parent directory containing language-specific font subdirectories
        lang_code: Language code (e.g. 'hi', 'zh', 'ar')
    Returns:
        Path to a random font file or None if not found
    """
    lang_font_dir = os.path.join(font_path, lang_code)

    if not os.path.exists(lang_font_dir):
        return None

    font_extensions = (".ttf", ".otf", ".TTF", ".OTF")
    font_files = []
    for ext in font_extensions:
        font_files.extend(glob.glob(os.path.join(lang_font_dir, f"*{ext}")))

    if not font_files:
        return None

    return random.choice(font_files)


def chunk_text_for_translation(text: str, max_length: int = 4500) -> List[str]:
    """
    Split text into chunks suitable for translation APIs with character limits.

    Args:
        text: Input text to chunk
        max_length: Maximum length per chunk (default 4500 to stay under 5000 limit)

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []

    # Try to split by sentences first
    sentences = text.split(". ")
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                word_chunk = ""
                for word in words:
                    if len(word_chunk) + len(word) + 1 > max_length:
                        if word_chunk:
                            chunks.append(word_chunk.strip())
                            word_chunk = word + " "
                        else:
                            # Single word is too long, force split
                            chunks.append(word[:max_length])
                            word_chunk = (
                                word[max_length:] + " "
                                if len(word) > max_length
                                else ""
                            )
                    else:
                        word_chunk += word + " "
                if word_chunk:
                    current_chunk = word_chunk + ". "
        else:
            current_chunk += sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def translate_text_chunks(translator, text: str) -> str:
    """
    Translate text by chunking it if necessary to handle API limits.

    Args:
        translator: GoogleTranslator instance
        text: Text to translate

    Returns:
        Translated text
    """
    chunks = chunk_text_for_translation(text)

    if len(chunks) == 1:
        # Single chunk, translate directly
        return translator.translate(text)

    # Multiple chunks, translate each and combine
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
        except Exception as e:
            print(f"Warning: Failed to translate chunk: {str(e)[:100]}...")
            # If translation fails, keep original text for this chunk
            translated_chunks.append(chunk)

    return " ".join(translated_chunks)


def wrap_text_for_box(text, box_width, font_path, font_size):
    """
    Wrap text for a box based on the font size and box width.
    Returns text with newline characters.
    """
    try:
        font = ImageFont.truetype(font_path, font_size, layout_engine="raqm")
    except Exception:
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

        if line_width <= box_width * 0.95:  # 95% of box width
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def fit_text_in_box(
    text, box_width, box_height, font_path, max_font_size=100, min_font_size=10
):
    """
    Find the optimal font size to fit text in a given box using binary search.
    Returns the optimal font and the list of text lines.
    """
    if not font_path or not os.path.exists(font_path):
        try:
            font = ImageFont.load_default()
            lines = text.split("\n") if "\n" in text else [text]
            return font, lines
        except Exception:
            return None, [text]

    lower = min_font_size
    upper = max_font_size
    optimal_font = None
    optimal_lines = []

    while lower <= upper:
        mid = (lower + upper) // 2
        wrapped_text = wrap_text_for_box(text, box_width, font_path, mid)
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
            total_height += line_height * 1.2  # Including line spacing

        max_line_width = max(line_widths) if line_widths else 0

        if max_line_width <= box_width * 0.95 and total_height <= box_height * 0.95:
            optimal_font = font
            optimal_lines = lines
            lower = mid + 1
        else:
            upper = mid - 1

    if not optimal_font:
        try:
            optimal_font = ImageFont.truetype(
                font_path, min_font_size, layout_engine="raqm"
            )
        except Exception:
            optimal_font = ImageFont.load_default()
        optimal_lines = wrap_text_for_box(
            text, box_width, font_path, min_font_size
        ).split("\n")

    return optimal_font, optimal_lines


class ImageTranslator:
    """
    Document translation pipeline using YOLO layout detection, OCR, and translation.
    """

    def __init__(
        self,
        model_path: str,
        font_path: str,
        langs: List[str] = ["hi"],
        conf: float = 0.4,
        imgsz: int = 1024,
        logger: Optional[logging.Logger] = None,
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
            self.logger.warning(
                "Translation not available - will preserve original text"
            )

        # Layout class mapping for YOLO detection
        self.class_mapping = {
            "plain text": "text",
            "title": "title",
            "figure": "image",
            "isolate_formula": "formula",
            "figure_caption": "caption",
            "table": "table",
        }
        self.translatable_classes = ["plain text", "title", "figure_caption"]

    def detect_and_ocr_image(self, cv2_image, results):
        """Process layout detection and OCR for an image once"""
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

            if class_name in self.translatable_classes:
                roi = cv2_image[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                text = pytesseract.image_to_string(
                    binary, config="--oem 3 --psm 6"
                ).strip()

                if text:
                    region_data["english_text"] = text
                    regions.append(region_data)

        return regions

    def process_single_image(self, idx, input_img):
        """Process a single image with optimized language handling"""
        image = (
            input_img
            if isinstance(input_img, Image.Image)
            else Image.fromarray(input_img)
        )
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        model = YOLOv10(self.model_path)
        det_res = model.predict(
            cv2_image, imgsz=self.imgsz, conf=self.conf, device="cpu"
        )
        regions = self.detect_and_ocr_image(cv2_image, det_res[0])

        for lang in self.langs:
            try:
                selected_font_path = get_random_font(self.font_path, lang)
                pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                translation_data = {
                    "image_id": f"image_{idx}",
                    "font_used": os.path.basename(selected_font_path)
                    if selected_font_path
                    else "default",
                    "regions": [],
                }

                bg_color = get_background_color(
                    cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                )
                text_color = get_text_color(bg_color)

                for region in regions:
                    bbox = region["bbox"]
                    translated_region = region.copy()

                    if region["layout_type"] in ["text", "title", "caption"]:
                        try:
                            # Use chunking to handle long text that exceeds API limits
                            translated_text = translate_text_chunks(
                                self.translators[lang], region["english_text"]
                            )
                            translated_region["translated_text"] = translated_text

                            font, text_lines = fit_text_in_box(
                                translated_text,
                                bbox["xmax"] - bbox["xmin"],
                                bbox["ymax"] - bbox["ymin"],
                                selected_font_path,
                            )

                            if font and text_lines:
                                draw.rectangle(
                                    [
                                        bbox["xmin"],
                                        bbox["ymin"],
                                        bbox["xmax"],
                                        bbox["ymax"],
                                    ],
                                    fill=bg_color,
                                )
                                current_y = bbox["ymin"]

                                for line in text_lines:
                                    line_bbox = font.getbbox(line)
                                    line_height = line_bbox[3] - line_bbox[1]
                                    draw.text(
                                        (bbox["xmin"], current_y),
                                        line,
                                        font=font,
                                        fill=text_color,
                                    )
                                    current_y += line_height * 1.2

                        except Exception as e:
                            self.logger.warning(
                                f"Translation error in image {idx}, region {region['region_id']}: {e}"
                            )
                            continue

                    translation_data["regions"].append(translated_region)

                _, buffer = cv2.imencode(
                    ".jpg", cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                )
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                translation_data["translated_image"] = img_base64
                self.output[lang] = translation_data

            except Exception as e:
                self.logger.error(
                    f"Error processing image {idx} for language {lang}: {e}"
                )
                continue

        return json.dumps(self.output, ensure_ascii=False)


class DocumentTranslator(BaseWorkflow):
    """
    Document Translation Workflow for SynthDoc.

    Translates documents to different languages while preserving layout using:
    - YOLO for layout detection
    - OCR for text extraction
    - Translation APIs for text conversion
    - Smart font rendering for target languages

    Supports:
    - Single images, PDFs, or folders containing mixed content
    - Multiple target languages simultaneously
    - Automatic font selection for target languages
    - Layout preservation through bounding box detection
    """

    def __init__(self, save_dir: str = "document_translation_output"):
        super().__init__()
        self.save_dir = save_dir
        self.workflow_name = "document_translation"
        self.logger = logging.getLogger(__name__)
        self._setup_save_directory()

    def _setup_save_directory(self):
        """Create save directory for document translation with simplified structure."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_dir = os.path.join(self.save_dir, "images")
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        os.makedirs(self.images_dir, exist_ok=True)

        # Create metadata.jsonl if it doesn't exist
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                pass  # Create empty file

        print(f"✅ Save directory created: {self.save_dir}")
        print(f"📂 Images will be saved to: {self.images_dir}")
        print(f"📄 Metadata will be saved to: {self.metadata_file}")

    def process(self, config: DocumentTranslationConfig) -> WorkflowResult:
        """
        Process document translation according to configuration.

        Pipeline:
        1. Collect input files (images/PDFs from files or folders)
        2. Convert PDFs to images
        3. Run translation pipeline on all images
        4. Save translated images and create dataset

        Args:
            config: DocumentTranslationConfig with translation parameters

        Returns:
            WorkflowResult with translated documents dataset
        """
        start_time = time.time()

        self.logger.info(
            f"🌍 Starting document translation to languages: {config.target_languages}"
        )

        # Validate dependencies
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO model not available")
            return self._create_fallback_result(
                "YOLO dependency not available", start_time
            )

        if not TESSERACT_AVAILABLE:
            self.logger.error("Tesseract not available")
            return self._create_fallback_result(
                "Tesseract dependency not available", start_time
            )

        if not TRANSLATOR_AVAILABLE:
            self.logger.error("Deep-translator not available")
            return self._create_fallback_result(
                "Translation dependency not available", start_time
            )

        # Step 1: Collect input files
        input_paths = []
        if config.input_images:
            input_paths.extend(config.input_images)

        if not input_paths:
            self.logger.error("No input images provided")
            return self._create_fallback_result("No input images provided", start_time)

        files = collect_input_files(input_paths)
        self.logger.info(
            f"📁 Found {len(files['images'])} images and {len(files['pdfs'])} PDFs"
        )

        # Step 2: Convert PDFs to images
        all_image_paths = files["images"].copy()
        pdf_temp_dirs = []

        for pdf_path in files["pdfs"]:
            try:
                self.logger.info(f"📄 Converting PDF: {Path(pdf_path).name}")
                temp_dir = Path(self.save_dir) / "temp_pdf_images" / Path(pdf_path).stem
                pdf_images = pdf_to_images(pdf_path, str(temp_dir))
                all_image_paths.extend(pdf_images)
                pdf_temp_dirs.append(temp_dir)
                self.logger.info(f"   ✅ Converted to {len(pdf_images)} images")
            except Exception as e:
                self.logger.error(f"   ❌ Failed to convert PDF {pdf_path}: {e}")

        self.logger.info(f"📄 Total images to process: {len(all_image_paths)}")

        # Step 3: Ensure YOLO model is available and initialize translator
        yolo_model_path = config.yolo_model_path
        if yolo_model_path is None:
            self.logger.info(
                "🔄 YOLO model path not provided, downloading default model..."
            )
            if MODEL_MANAGER_AVAILABLE:
                yolo_model_path = str(ensure_model("doclayout-yolo"))
            else:
                raise ValueError(
                    "No YOLO model path provided and model manager not available. "
                    "Please provide a valid yolo_model_path."
                )

        translator = ImageTranslator(
            model_path=yolo_model_path,
            font_path=config.font_path,
            langs=config.target_languages,
            conf=config.confidence_threshold,
            imgsz=config.image_size,
            logger=self.logger,
        )

        # Step 4: Process all images
        all_results = []

        for idx, image_path in enumerate(all_image_paths):
            try:
                self.logger.info(
                    f"🔄 Processing image {idx + 1}/{len(all_image_paths)}: {Path(image_path).name}"
                )

                # Load and process image
                image = Image.open(image_path)
                json_output = translator.process_single_image(idx, image)
                translations = json.loads(json_output)

                # Save results for each language
                for lang, data in translations.items():
                    if "translated_image" not in data:
                        continue

                    # Decode and save translated image directly to images folder
                    img_base64 = data["translated_image"]
                    img_bytes = base64.b64decode(img_base64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    output_filename = (
                        f"translated_{lang}_{idx}_{Path(image_path).stem}.jpg"
                    )
                    output_path = os.path.join(self.images_dir, output_filename)
                    cv2.imwrite(output_path, img)

                    # Create result record
                    result = {
                        "id": f"translation_{lang}_{idx}",
                        "original_image_path": str(image_path),
                        "translated_image_path": output_path,
                        "target_language": lang,
                        "font_used": data.get("font_used", "default"),
                        "num_regions": len(data.get("regions", [])),
                        "regions_data": json.dumps(
                            data.get("regions", []), ensure_ascii=False
                        ),
                        "source_type": "pdf"
                        if str(image_path).find("temp_pdf_images") != -1
                        else "image",
                    }

                    # Add image to result for dataset
                    translated_img = Image.open(output_path)
                    result["image"] = translated_img
                    result["image_path"] = output_path

                    all_results.append(result)

            except Exception as e:
                self.logger.error(f"❌ Failed to process {image_path}: {e}")
                continue

        # Step 5: Use unified output structure (no separate dataset folder)
        output_files = []

        with open(self.metadata_file, "a", encoding="utf-8") as metadata_file:
            for result in all_results:
                # Images are already saved directly to images folder
                img_path = result["translated_image_path"]
                filename = os.path.basename(img_path)
                output_files.append(img_path)

                # Create metadata entry
                metadata_entry = {
                    "file_name": filename,
                    "image_path": f"images/{filename}",
                    "id": result["id"],
                    "original_image_path": result["original_image_path"],
                    "target_language": result["target_language"],
                    "font_used": result["font_used"],
                    "num_regions": result["num_regions"],
                    "regions_data": result["regions_data"],
                    "source_type": result["source_type"],
                }

                # Write to metadata.jsonl
                metadata_file.write(
                    json.dumps(metadata_entry, ensure_ascii=False) + "\n"
                )

                # Step 6: Create HuggingFace dataset directly from results
        if all_results:
            # Create dataset dict directly from results
            dataset_dict = {
                "image": [],
                "file_name": [],
                "image_path": [],
                "id": [],
                "original_image_path": [],
                "target_language": [],
                "font_used": [],
                "num_regions": [],
                "regions_data": [],
                "source_type": [],
            }

            # Add each result to the dataset
            for result in all_results:
                filename = os.path.basename(result["translated_image_path"])
                img_path = result["translated_image_path"]  # Already in images folder

                # Load the image for the dataset
                image = Image.open(img_path)

                dataset_dict["image"].append(image)
                dataset_dict["file_name"].append(filename)
                dataset_dict["image_path"].append(f"images/{filename}")
                dataset_dict["id"].append(result["id"])
                dataset_dict["original_image_path"].append(
                    result["original_image_path"]
                )
                dataset_dict["target_language"].append(result["target_language"])
                dataset_dict["font_used"].append(result["font_used"])
                dataset_dict["num_regions"].append(result["num_regions"])
                dataset_dict["regions_data"].append(result["regions_data"])
                dataset_dict["source_type"].append(result["source_type"])

            dataset = Dataset.from_dict(dataset_dict)
            self.logger.info(
                f"📦 Created HuggingFace dataset with {len(dataset)} samples"
            )
            self.logger.info(f"📁 Unified output structure: {self.save_dir}")
            self.logger.info(f"   - Images: {self.images_dir}")
            self.logger.info(f"   - Metadata: {self.metadata_file}")

        else:
            dataset = Dataset.from_dict({})
            output_files = []

        # Clean up temporary PDF image directories
        for temp_dir in pdf_temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors

        processing_time = time.time() - start_time

        self.logger.info(
            f"✅ Translation completed: {len(all_results)} translated documents in {processing_time:.2f}s"
        )

        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "document_translation",
                "target_languages": config.target_languages,
                "total_input_files": len(input_paths),
                "total_images_processed": len(all_image_paths),
                "successful_translations": len(all_results),
                "pdfs_converted": len(files["pdfs"]),
                "images_processed": len(files["images"]),
                "yolo_model": yolo_model_path,
                "font_path": config.font_path,
                "processing_time": processing_time,
                "output_structure": {
                    "output_dir": self.save_dir,
                    "images_dir": self.images_dir,
                    "metadata_file": self.metadata_file,
                    "total_images": len(output_files),
                },
                "generated_files": [os.path.basename(f) for f in output_files],
            },
            num_samples=len(all_results),
            output_files=output_files,
        )

    def _create_fallback_result(
        self, error_msg: str, start_time: float
    ) -> WorkflowResult:
        """Create a fallback result for errors."""
        return WorkflowResult(
            dataset=Dataset.from_dict({}),
            metadata={
                "workflow_type": "document_translation",
                "status": "failed",
                "error": error_msg,
                "processing_time": time.time() - start_time,
            },
            num_samples=0,
        )

    @classmethod
    def load_dataset_from_directory(cls, dataset_dir: str) -> Dataset:
        """
        Load a HuggingFace dataset from a directory containing images and metadata.jsonl.

        Args:
            dataset_dir: Path to the directory containing 'images/' folder and 'metadata.jsonl'

        Returns:
            Dataset: HuggingFace dataset with images and metadata
        """
        metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.jsonl not found in {dataset_dir}")

        dataset_records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                # Load the image
                img_path = os.path.join(dataset_dir, record["image_path"])
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    record["image"] = image
                    dataset_records.append(record)

        if not dataset_records:
            return Dataset.from_dict({})

        # Create dataset from records
        dataset_dict = {
            "image": [r["image"] for r in dataset_records],
            "file_name": [r["file_name"] for r in dataset_records],
            "image_path": [r["image_path"] for r in dataset_records],
            "id": [r["id"] for r in dataset_records],
            "original_image_path": [r["original_image_path"] for r in dataset_records],
            "target_language": [r["target_language"] for r in dataset_records],
            "font_used": [r["font_used"] for r in dataset_records],
            "num_regions": [r["num_regions"] for r in dataset_records],
            "regions_data": [r["regions_data"] for r in dataset_records],
            "source_type": [r["source_type"] for r in dataset_records],
        }

        return Dataset.from_dict(dataset_dict)
