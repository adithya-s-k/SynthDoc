import random
import json
import time
import shutil
import os
import enum
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from synthdoc.workflows.base import BaseWorkflow
from synthdoc.models import VQAGenerationConfig, WorkflowResult
from synthdoc.utils import CostTracker

# LiteLLM for unified LLM access
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# PDF processing imports
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OCR imports
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Base64 encoding for images
import base64

# Pydantic models for structured output
from pydantic import BaseModel, Field
from typing import List as TypingList

if LITELLM_AVAILABLE:
    class VQADifficulty(str, enum.Enum):
        EASY = "easy"
        MEDIUM = "medium"
        HARD = "hard"

    class VQAType(str, enum.Enum):
        DESCRIPTIVE = "descriptive"
        MCQ = "mcq"

    class VQAPair(BaseModel):
        id: str = Field(description="Unique identifier for the VQA pair")
        type: VQAType = Field(description="Type of question: descriptive or mcq")
        difficulty: VQADifficulty = Field(
            description="Difficulty level of the question"
        )
        question: str = Field(description="The question text")
        answer: str = Field(description="The correct answer")
        explanation: str = Field(description="Brief explanation of the answer")
        hard_negatives: Optional[List[str]] = Field(
            default=None,
            description="List of plausible but incorrect answers for descriptive questions",
        )
        choices: Optional[List[str]] = Field(
            default=None, description="List of all choices for MCQ questions"
        )


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

    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return []

    return image_paths


class VQAGenerator(BaseWorkflow):
    """Generate visual question-answering datasets with LiteLLM integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        save_dir: str = "vqa_output",
    ):
        super().__init__()

        # Auto-detect API key if not provided
        if api_key is None:
            from ...config import get_api_key
            api_key = get_api_key("auto")

        self.api_key = api_key
        self.llm_model = llm_model
        self.save_dir = save_dir
        self.cost_tracker = CostTracker()
        self.temp_dir = Path("temp_pdf_images")
        self.temp_dir.mkdir(exist_ok=True)
        self._setup_save_directory()

        #setup litellm
        self.llm_available = False
        if api_key and LITELLM_AVAILABLE:
            try:
                # Configure LiteLLM
                if api_key != "your_api_key_here":
                    self._set_api_key_for_model(llm_model, api_key)
                litellm.set_verbose = False
                litellm.num_retries = 3
                litellm.request_timeout = 120
                litellm.drop_params = True
                self.llm_available = True
                print(f"LiteLLM initialized with model: {llm_model}")
            except Exception as e:
                print(f"Failed to initialize LiteLLM: {e}")
        elif api_key == "your_api_key_here":
            print("Please set a valid API key in your .env file")
        elif not LITELLM_AVAILABLE:
            print("LiteLLM not available. Install with: pip install litellm")
        elif not api_key:
            print("No API key found in environment variables")

    def _setup_save_directory(self):
        """Create save directory structure with images folder and metadata.jsonl directly in save_dir."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Create images directory and metadata file directly in save_dir
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

    def _set_api_key_for_model(self, model: str, api_key: str):
        """Set the appropriate API key based on model type."""
        if model.startswith("gpt") or model.startswith("openai"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif model.startswith("claude") or model.startswith("anthropic"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif model.startswith("gemini") or model.startswith("google"):
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key  # LiteLLM also checks this
        else:
            print("Model Not Initialized")


    def _get_image_mime_type(self, image_path: str) -> str:
        """Get the correct MIME type for the image."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff'
        }
        return mime_types.get(ext, 'image/jpeg')

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "English"

        try:
            lang_code = detect(text)
            # Map common language codes to full names
            lang_map = {
                "en": "English",
                "hi": "Hindi",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "ar": "Arabic",
                "bn": "Bengali",
                "ur": "Urdu",
                "ta": "Tamil",
                "te": "Telugu",
                "mr": "Marathi",
                "gu": "Gujarati",
            }
            return lang_map.get(lang_code, "English")
        except:
            return "English"

    def _extract_text_with_ocr(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR."""
        try:
            if not PYTESSERACT_AVAILABLE:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "error": "Tesseract not available",
                }

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)

            # Simple confidence estimation based on text length and characters
            confidence = min(0.9, len(text.strip()) / 100) if text.strip() else 0.0

            return {"text": text.strip(), "confidence": confidence}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def process(self, config: VQAGenerationConfig) -> WorkflowResult:
        """Generate VQA dataset based on configuration."""
        start_time = time.time()

        print(f"🤖 Starting VQA generation for {len(config.documents)} documents...")

        # Step 1: Collect input files (similar to document translator)
        input_paths = config.documents or []
        files = collect_input_files(input_paths)
        print(f"📁 Found {len(files['images'])} images and {len(files['pdfs'])} PDFs")

        # Step 2: Convert PDFs to images
        all_image_paths = files["images"].copy()
        pdf_temp_dirs = []

        for pdf_path in files["pdfs"]:
            try:
                print(f"📄 Converting PDF: {Path(pdf_path).name}")
                temp_dir = self.temp_dir / Path(pdf_path).stem
                pdf_images = pdf_to_images(pdf_path, str(temp_dir))
                all_image_paths.extend(pdf_images)
                pdf_temp_dirs.append(temp_dir)
                print(f"   ✅ Converted to {len(pdf_images)} images")
            except Exception as e:
                print(f"   ❌ Failed to convert PDF {pdf_path}: {e}")

        print(f"📄 Total images to process: {len(all_image_paths)}")

        # Step 3: Process all images for VQA (generate exactly 5 VQA pairs per image)
        all_results = []
        total_vqa_pairs = 0

        for idx, image_path in enumerate(all_image_paths):
            try:
                print(
                    f"🔄 Processing image {idx + 1}/{len(all_image_paths)}: {Path(image_path).name}"
                )

                # Generate exactly 5 VQA pairs for this image
                vqa_result = self._generate_vqa_for_image(image_path, config, idx)
                all_results.append(vqa_result)
                total_vqa_pairs += vqa_result.get("num_vqa_pairs", 5)

            except Exception as e:
                print(f"❌ Failed to process {image_path}: {e}")
                continue

        # Step 4: Save results with images folder and metadata.jsonl structure
        output_files = self._save_vqa_results(all_results)

        # Cleanup temporary PDF images
        self._cleanup_temp_files()

        return WorkflowResult(
            dataset=None,
            metadata={
                "workflow_type": "vqa_generation",
                "total_images": len(all_results),
                "total_vqa_pairs": total_vqa_pairs,
                "vqa_pairs_per_image": 5,
                "processing_time": time.time() - start_time,
                "cost_summary": self.cost_tracker.get_summary(),
                "output_structure": {
                    "images_folder": self.images_dir,
                    "metadata_file": self.metadata_file,
                    "total_image_files": len(output_files),
                    "structure": "Each image has 5 VQA pairs saved in single JSONL entry",
                },
            },
            num_samples=total_vqa_pairs,  # Total number of VQA pairs generated
            output_files=output_files,
        )

    def _generate_vqa_for_image(
        self, image_path: str, config: VQAGenerationConfig, idx: int
    ) -> Dict:
        """Generate exactly 5 VQA pairs for a single image and return as single result."""
        # Extract text content using OCR
        ocr_result = self._extract_text_with_ocr(image_path)
        detected_language = self._detect_language(ocr_result["text"])

        vqa_pairs = []
        processing_mode = "template"  # Default fallback

        if self.llm_available:
            # Use LiteLLM for VQA generation - generates exactly 5 pairs
            llm_pairs = self._generate_llm_vqa(image_path, ocr_result["text"])

            if llm_pairs and len(llm_pairs) >= 5:
                # Use LLM-generated pairs (take first 5 to ensure exactly 5)
                processing_mode = "VLM"
                for q_idx, pair in enumerate(llm_pairs[:5]):  # Ensure exactly 5 pairs
                    question = pair.get("question", f"Sample question {q_idx + 1}")
                    answer = pair.get("answer", f"Sample answer {q_idx + 1}")
                    explanation = pair.get("explanation", "Generated by LiteLLM")
                    question_type = pair.get("type", "descriptive")
                    difficulty = pair.get("difficulty", "medium")

                    # Handle hard negatives based on question type
                    if question_type == "mcq" and "choices" in pair:
                        # For MCQ, remove the correct answer from choices to get negatives
                        neg_answers = [
                            choice for choice in pair["choices"] if choice != answer
                        ]
                    else:
                        neg_answers = pair.get("hard_negatives", [])

                    vqa_pair = {
                        "question": question,
                        "answer": answer,
                        "explanation": explanation,
                        "hard_negatives": neg_answers
                        if config.include_hard_negatives
                        else [],
                        "question_type": question_type,
                        "difficulty": difficulty,
                    }
                    vqa_pairs.append(vqa_pair)
            else:
                # Fallback to template-based generation if LLM fails or returns insufficient pairs
                print(
                    f"LLM generated {len(llm_pairs) if llm_pairs else 0} pairs, falling back to template"
                )
                vqa_pairs = self._generate_template_vqa_pairs(ocr_result["text"])
        else:
            # Use fallback template-based generation for exactly 5 pairs
            vqa_pairs = self._generate_template_vqa_pairs(ocr_result["text"])

        # Ensure we have exactly 5 pairs
        while len(vqa_pairs) < 5:
            # Fill with template pairs if needed
            additional_pairs = self._generate_template_vqa_pairs(ocr_result["text"])
            vqa_pairs.extend(additional_pairs)

        # Trim to exactly 5 pairs
        vqa_pairs = vqa_pairs[:5]

        # Create single result entry with all 5 VQA pairs
        result = {
            "id": f"vqa_image_{idx}",
            "image_path": image_path,
            "vqa_pairs": vqa_pairs,
            "source_text": ocr_result["text"],
            "detected_language": detected_language,
            "ocr_confidence": ocr_result.get("confidence", 0.0),
            "num_vqa_pairs": len(vqa_pairs),
            "metadata": {
                "source_file": str(image_path),
                "processing_mode": processing_mode,
                "generated_by": "synthdoc_vqa_generator",
            },
        }

        return result

    def _generate_llm_vqa(self, image_path: str, ocr_text: str) -> List[Dict]:
        """Generate VQA pairs using LiteLLM with structured output."""
        try:
            # Create the VQA generation prompt
            prompt = self._create_vqa_prompt_v2(ocr_text)

            # Encode image as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Create messages with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{self._get_image_mime_type(image_path)};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]

            # Generate with LiteLLM using structured output
            response = litellm.completion(
                model=self.llm_model,
                messages=messages,
                max_tokens=4000,
                temperature=0.7,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "vqa_pairs",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "vqa_pairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "type": {"type": "string", "enum": ["descriptive", "mcq"]},
                                            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                                            "question": {"type": "string"},
                                            "answer": {"type": "string"},
                                            "explanation": {"type": "string"},
                                            "hard_negatives": {"type": "array", "items": {"type": "string"}},
                                            "choices": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["id", "type", "difficulty", "question", "answer", "explanation"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["vqa_pairs"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            # Track cost
            cost = self.cost_tracker.track_usage(response, model=self.llm_model)
            print(f"VQA Generation: ${cost:.6f}")

            # Parse the structured response
            content = response.choices[0].message.content
            parsed_data = json.loads(content)
            vqa_pairs = parsed_data.get("vqa_pairs", [])
            
            print(f"✅ Generated {len(vqa_pairs)} VQA pairs with structured output")
            return vqa_pairs

        except Exception as e:
            print(f"LiteLLM VQA generation failed: {e}")
            return []

    

    def _create_vqa_prompt_v2(self, ocr_text: str) -> str:
        """Create VQA generation prompt for structured output."""
        ocr_section = (
            f"\n\n**Extracted Text from Document:**\n{ocr_text}\n"
            if ocr_text.strip()
            else ""
        )

        prompt = f"""### Visual Question Answering (VQA) Pair Generation Task

You are an expert Visual Question Answering (VQA) pair generator. Analyze the provided document image and create **exactly 5 high-quality VQA pairs**.{ocr_section}

**Requirements:**

Generate **5 VQA pairs** with this distribution:
* **4 Descriptive questions** (type: "descriptive") - Open-ended questions with hard negative answers
* **1 Multiple Choice question** (type: "mcq") - With 4 choices including 1 correct answer

**Difficulty Distribution:**
* At least 1 "easy" question (direct fact retrieval)
* At least 1 "medium" question (minor reasoning required)  
* At least 1 "hard" question (multi-step inference)

**Guidelines:**
1. **Questions must be answerable from the document content only**
2. **For descriptive questions**: Provide 4 plausible but incorrect "hard_negatives"
3. **For MCQ questions**: Provide 4 "choices" (1 correct + 3 distractors)
4. **Hard negatives/distractors** should be based on document content but incorrect
5. Use diverse question types: factual, reasoning, numerical, comparative, etc.

**Question ID Format:** Use "vqa_1", "vqa_2", "vqa_3", "vqa_4", "vqa_5"

Focus on the actual document content, text, data, and information rather than just visual appearance.

"""

        return prompt

    def _generate_template_vqa_pairs(self, content: str) -> List[Dict]:
        """Generate exactly 5 VQA pairs using templates when LLM is not available."""
        templates = {
            "factual": {
                "questions": [
                    "What is the main topic discussed in this document?",
                    "What are the key points mentioned?",
                    "What specific details are provided?",
                    "What information is presented in this document?",
                    "What type of document is this?",
                ],
                "answers": [
                    "The main topic is document processing and analysis.",
                    "Key points include methodology and results.",
                    "Specific technical details and measurements.",
                    "The document presents comprehensive analysis and findings.",
                    "This is a technical or research document.",
                ],
            },
            "reasoning": {
                "questions": [
                    "Why is this approach effective?",
                    "How do the different components work together?",
                    "What can be inferred from the results?",
                    "What conclusions can be drawn?",
                    "How does this methodology address the problem?",
                ],
                "answers": [
                    "This approach is effective because it combines multiple techniques.",
                    "Components work together through systematic integration.",
                    "Results suggest improved performance and reliability.",
                    "The conclusions indicate successful implementation.",
                    "The methodology provides a structured solution approach.",
                ],
            },
            "comparative": {
                "questions": [
                    "How does this compare to alternative approaches?",
                    "What are the advantages and disadvantages?",
                    "Which method performs better?",
                    "What differences are highlighted?",
                    "How do these results compare to previous work?",
                ],
                "answers": [
                    "This approach offers better accuracy than alternatives.",
                    "Advantages include speed; disadvantages include complexity.",
                    "The proposed method shows superior performance.",
                    "Key differences include improved efficiency and effectiveness.",
                    "Results show significant improvement over previous methods.",
                ],
            },
        }

        question_types = [
            "factual",
            "reasoning",
            "comparative",
            "factual",
            "reasoning",
        ]  # 5 questions with distribution
        difficulties = ["easy", "medium", "hard", "easy", "medium"]  # Varied difficulty

        vqa_pairs = []

        for i in range(5):
            question_type = question_types[i]
            template = templates[question_type]

            # Use modulo to cycle through questions if we have fewer templates than needed
            q_idx = i % len(template["questions"])

            question = template["questions"][q_idx]
            answer = template["answers"][q_idx]
            explanation = (
                f"This is based on {question_type} analysis of the document content."
            )
            difficulty = difficulties[i]

            # Generate hard negatives
            hard_negatives = [
                f"Incorrect answer option A for question {i + 1}",
                f"Incorrect answer option B for question {i + 1}",
                f"Incorrect answer option C for question {i + 1}",
                f"Incorrect answer option D for question {i + 1}",
            ]

            vqa_pair = {
                "question": question,
                "answer": answer,
                "explanation": explanation,
                "hard_negatives": hard_negatives,
                "question_type": question_type,
                "difficulty": difficulty,
            }
            vqa_pairs.append(vqa_pair)

        return vqa_pairs

    def _generate_template_vqa(
        self, content: str, question_type: str, idx: int
    ) -> tuple:
        """Generate VQA using templates when LLM is not available (legacy method for compatibility)."""
        templates = {
            "factual": {
                "questions": [
                    "What is the main topic discussed in this document?",
                    "What are the key points mentioned?",
                    "What specific details are provided?",
                ],
                "answers": [
                    "The main topic is document processing and analysis.",
                    "Key points include methodology and results.",
                    "Specific technical details and measurements.",
                ],
            },
            "reasoning": {
                "questions": [
                    "Why is this approach effective?",
                    "How do the different components work together?",
                    "What can be inferred from the results?",
                ],
                "answers": [
                    "This approach is effective because it combines multiple techniques.",
                    "Components work together through systematic integration.",
                    "Results suggest improved performance and reliability.",
                ],
            },
            "comparative": {
                "questions": [
                    "How does this compare to alternative approaches?",
                    "What are the advantages and disadvantages?",
                    "Which method performs better?",
                ],
                "answers": [
                    "This approach offers better accuracy than alternatives.",
                    "Advantages include speed; disadvantages include complexity.",
                    "The proposed method shows superior performance.",
                ],
            },
        }

        template = templates.get(question_type, templates["factual"])
        q_idx = idx % len(template["questions"])

        question = template["questions"][q_idx]
        answer = template["answers"][q_idx]
        explanation = (
            f"This is based on {question_type} analysis of the document content."
        )

        # Generate hard negatives
        hard_negatives = [
            f"Incorrect answer option A for question {idx + 1}",
            f"Incorrect answer option B for question {idx + 1}",
            f"Incorrect answer option C for question {idx + 1}",
        ]

        return question, answer, explanation, hard_negatives

    def _assess_difficulty(self, question: str) -> str:
        """Assess question difficulty based on complexity indicators."""
        difficulty_indicators = {
            "easy": ["what", "where", "when", "who"],
            "medium": ["how", "why", "explain", "describe"],
            "hard": ["analyze", "compare", "evaluate", "synthesize", "infer"],
        }

        question_lower = question.lower()

        for difficulty, indicators in difficulty_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return difficulty

        return "medium"  # Default

    def _save_vqa_results(self, results: List[Dict]) -> List[str]:
        """Save VQA results with images folder and metadata.jsonl structure.

        Each result now contains multiple VQA pairs for a single image.
        Save one image per result and create a single JSONL entry with all VQA pairs.
        """
        output_files = []
        total_vqa_pairs = 0

        with open(self.metadata_file, "w", encoding="utf-8") as metadata_file:
            for result in results:
                # Copy image to dataset images folder (once per image)
                original_img_path = result["image_path"]
                filename = f"{result['id']}.png"
                new_img_path = os.path.join(self.images_dir, filename)

                # Copy the image only once
                shutil.copy2(original_img_path, new_img_path)
                output_files.append(new_img_path)

                # Count VQA pairs for this image
                num_pairs = result.get(
                    "num_vqa_pairs", len(result.get("vqa_pairs", []))
                )
                total_vqa_pairs += num_pairs

                # Create metadata entry with all VQA pairs for this image
                metadata_entry = {
                    "file_name": filename,
                    "image_path": f"images/{filename}",
                    "id": result["id"],
                    "vqa_pairs": result["vqa_pairs"],  # All 5 VQA pairs for this image
                    "num_vqa_pairs": num_pairs,
                    "source_text": result["source_text"],
                    "detected_language": result["detected_language"],
                    "ocr_confidence": result["ocr_confidence"],
                    "source_file": result["image_path"],
                    "processing_mode": result["metadata"]["processing_mode"],
                    "generated_by": result["metadata"]["generated_by"],
                }

                # Write to metadata.jsonl (one entry per image with all VQA pairs)
                metadata_file.write(
                    json.dumps(metadata_entry, ensure_ascii=False) + "\n"
                )

        print(
            f"💾 Saved {len(results)} images with {total_vqa_pairs} total VQA pairs to:"
        )
        print(f"   - Images: {self.images_dir} ({len(results)} unique images)")
        print(
            f"   - Metadata: {self.metadata_file} ({len(results)} entries, {total_vqa_pairs} VQA pairs)"
        )

        return output_files

    def _cleanup_temp_files(self):
        """Clean up temporary PDF image files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp files: {e}")

    def _extract_document_content(self, doc_path: str) -> str:
        """Extract text content from document."""
        try:
            if isinstance(doc_path, dict) and "content" in doc_path:
                return doc_path["content"]
            elif str(doc_path).endswith((".txt", ".md")):
                with open(doc_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                # Use OCR for images
                ocr_result = self._extract_text_with_ocr(str(doc_path))
                return ocr_result.get("text", f"Sample document content for {doc_path}")
        except Exception as e:
            print(f"⚠️ Error extracting content from {doc_path}: {e}")
            return f"Sample document content for {doc_path}"

    def _calculate_similarity_scores(
        self, correct_answer: str, hard_negatives: List[str]
    ) -> List[float]:
        """Calculate similarity scores between correct answer and hard negatives."""

        def word_overlap_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0

        scores = []
        for negative in hard_negatives:
            score = word_overlap_similarity(correct_answer, negative)
            scores.append(round(score, 3))

        return scores
