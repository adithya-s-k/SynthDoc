import random
import json
import time
import io
import base64
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from datasets import Dataset
from ..base import BaseWorkflow
from ...models import VQAGenerationConfig, WorkflowResult
from ...utils import CostTracker

#PDF processing imports
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# LiteLLM imports
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class VQAGenerator(BaseWorkflow):
    """Generate visual question-answering datasets with LLM integration."""

    def __init__(self, api_key: Optional[str] = None, llm_model: str = "gemini-2.5-flash", use_litellm: bool = False):
        super().__init__()
        self.api_key = api_key
        self.llm_model = llm_model
        self.use_litellm = use_litellm
        self.cost_tracker = CostTracker()

        # Setup model client based on provider
        if use_litellm:
            # Setup LiteLLM for any model
            if not LITELLM_AVAILABLE:
                raise ImportError("LiteLLM is required. Install with: pip install litellm")
            if api_key:
                import os
                os.environ["OPENAI_API_KEY"] = api_key  # LiteLLM uses this for most providers
            self.client = None  # Will use litellm.completion directly
        else:
            # Setup Google GenAI client (default)
            if api_key:
                import os
                os.environ["GOOGLE_API_KEY"] = api_key
            self.client = genai.Client()

        # PDF conversion setup
        self.temp_dir = Path("temp_pdf_images")
        self.temp_dir.mkdir(exist_ok=True)

    def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images."""
        if not PDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install PyMuPDF")

        image_paths = []
        pdf_name = Path(pdf_path).stem

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image with high DPI for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)

                # Save as PNG
                image_path = self.temp_dir / f"{pdf_name}_page_{page_num + 1}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))

            doc.close()
            print(f"Converted PDF to {len(image_paths)} images")
            return image_paths

        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return []

    def _cleanup_temp_files(self):
        """Clean up temporary PDF image files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp files: {e}")

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "English"

        try:
            lang_code = detect(text)
            # Map common language codes to full names
            lang_map = {
                'en': 'English',
                'hi': 'Hindi',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese',
                'ar': 'Arabic',
                'bn': 'Bengali',
                'ur': 'Urdu',
                'ta': 'Tamil',
                'te': 'Telugu',
                'mr': 'Marathi',
                'gu': 'Gujarati'
            }
            return lang_map.get(lang_code, 'English')
        except:
            return "English"

    def _make_api_call(self, messages: List[Dict], image_data: bytes = None, mime_type: str = None) -> str:
        """Make API call using either Google GenAI or LiteLLM."""
        if self.use_litellm:
            # Use LiteLLM for any model
            try:
                if image_data:
                    # For vision models with LiteLLM
                    import base64
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    messages_with_image = []
                    for msg in messages:
                        if msg["role"] == "user":
                            messages_with_image.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": msg["content"]},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                                ]
                            })
                        else:
                            messages_with_image.append(msg)
                    response = litellm.completion(model=self.llm_model, messages=messages_with_image)
                else:
                    # Text-only with LiteLLM
                    response = litellm.completion(model=self.llm_model, messages=messages)
                return response.choices[0].message.content
            except Exception as e:
                print(f"LiteLLM API error: {e}")
                return ""
        else:
            # Use Google GenAI (default)
            try:
                if image_data:
                    # Vision call with Google GenAI
                    content_parts = [messages[0]["content"]]
                    if image_data:
                        content_parts.append(types.Part.from_bytes(image_data, mime_type=mime_type))

                    response = self.client.models.generate_content(
                        model=self.llm_model,
                        contents=content_parts,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                else:
                    # Text-only with Google GenAI
                    response = self.client.models.generate_content(
                        model=self.llm_model,
                        contents=messages[0]["content"],
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                return response.text.strip()
            except Exception as e:
                print(f"Google GenAI API error: {e}")
                return ""

    def process(self, config: VQAGenerationConfig) -> WorkflowResult:
        """Generate VQA dataset based on configuration."""

        # Handle folder input
        if hasattr(config, 'input_folder') and config.input_folder:
            folder_files = self._scan_folder_for_files(str(config.input_folder))
            config.documents.extend(folder_files)

        print(f"Starting VQA generation for {len(config.documents)} documents...")

        all_questions = []
        all_answers = []
        all_explanations = []
        all_hard_negatives = []
        all_metadata = []

        for doc_idx, doc_path in enumerate(config.documents):
            print(f"Processing document {doc_idx + 1}/{len(config.documents)}: {doc_path}")

            # Handle PDF conversion
            if doc_path.lower().endswith('.pdf'):
                print(f"Converting PDF to images...")
                pdf_images = self._convert_pdf_to_images(doc_path)
                if not pdf_images:
                    print(f"Skipping PDF {doc_path} - conversion failed")
                    continue
                documents_to_process = pdf_images
            else:
                documents_to_process = [doc_path]

            # Process each document/image
            for img_path in documents_to_process:
                print(f"  Processing: {Path(img_path).name}")

                # Load document content
                doc_content = self._extract_document_content(img_path)

                # Generate questions for this document
                questions, answers, hard_negatives, explanations, metadata = self._generate_vqa_for_document(
                    doc_content, img_path, config
                )

                all_questions.extend(questions)
                all_answers.extend(answers)
                all_explanations.extend(explanations)
                all_hard_negatives.extend(hard_negatives)
                all_metadata.extend(metadata)
        
                # Create dataset samples for image-based VQA
        samples = []
        for i in range(len(all_questions)):
            # Handle both document paths and image objects
            doc_path = all_metadata[i]["document_path"]
            
            # Try to load image if document is an image file
            image = None
            if isinstance(doc_path, str) and doc_path.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    from PIL import Image
                    image = Image.open(doc_path)
                except Exception as e:
                    print(f" Could not load image {doc_path}: {e}")
            
            sample = {
                "id": f"vqa_{i}",
                "image": image,  # Include image for VQA
                "text": self._extract_document_content(doc_path),  # Include text content
                "document_path": doc_path,
                "question": all_questions[i],
                "answer": all_answers[i],
                "explanation": all_explanations[i] if i < len(all_explanations) else all_answers[i],
                "hard_negatives": all_hard_negatives[i] if i < len(all_hard_negatives) else [],
                "question_type": all_metadata[i]["question_type"],
                "difficulty": all_metadata[i]["difficulty"],
                "similarity_scores": all_metadata[i].get("similarity_scores", []),
                "metadata": {
                    "source": str(doc_path),
                    "language": "en",  # Default language
                    "question_type": all_metadata[i]["question_type"]
                }
            }
            samples.append(sample)

        # Create comprehensive dataset using README schema
        if not samples:
            dataset = Dataset.from_dict({})
        else:
            # Extract data for comprehensive dataset creation
            images = [s['image'] for s in samples]
            image_paths = [s.get('image_path', '') for s in samples]
            pdf_names = [s.get('pdf_name', f"vqa_doc_{i}") for i, s in enumerate(samples)]
            page_numbers = [s.get('page_number', 0) for s in samples]
            
            # VQA-specific content with questions and answers
            markdown_content = []
            html_content = []
            for s in samples:
                q = s.get('question', '')
                a = s.get('answer', '')
                base_md = s.get('markdown', f"Document: {s.get('text', '')}")
                base_html = s.get('html', f"<p>{s.get('text', '')}</p>")
                
                enhanced_md = f"{base_md}\n\n**Question:** {q}\n**Answer:** {a}"
                enhanced_html = f"{base_html}<br><strong>Q:</strong> {q}<br><strong>A:</strong> {a}"
                
                markdown_content.append(enhanced_md)
                html_content.append(enhanced_html)
            
            # Create VQA triplets (individual samples)
            triplets = self._create_vqa_dataset(samples, additional_metadata={
                "workflow": "vqa_generation",
                "config": config.model_dump(),
                "total_cost": self.cost_tracker.get_summary()
            })

        # Save the triplets
        output_files = self._save_dataset(triplets, config)

        # Cleanup temporary PDF images
        self._cleanup_temp_files()

        return WorkflowResult(
            dataset=triplets,
            metadata={
                "workflow_type": "vqa_generation",
                "total_questions": len(all_questions),
                "cost_summary": self.cost_tracker.get_summary()
            },
            num_samples=len(samples),
            output_files=output_files
        )

    def _create_vqa_dataset(self, samples: List[Dict], additional_metadata: Dict = None) -> Dataset:
        """Create comprehensive VQA dataset with individual samples."""
        if not samples:
            return Dataset.from_dict({})

        # Create comprehensive samples list (each sample is a complete VQA entry)
        comprehensive_samples = []

        for i, sample in enumerate(samples):
            comprehensive_sample = {
                'id': f'vqa_sample_{i+1}',
                'image_path': sample.get('document_path', ''),
                'image': sample.get('image'),  # PIL Image object
                'question': sample.get('question', ''),
                'answer': sample.get('answer', ''),
                'explanation': sample.get('explanation', sample.get('answer', '')),  # Use answer as explanation if no separate explanation
                'question_type': sample.get('question_type', 'factual'),
                'difficulty': sample.get('difficulty', 'medium'),
                'hard_negatives': sample.get('hard_negatives', []),
                'source_text': sample.get('text', ''),
                'metadata': {
                    'language': 'en',
                    'processing_mode': sample.get('metadata', {}).get('processing_mode', 'VLM'),
                    'source_file': sample.get('document_path', ''),
                    'question_type': sample.get('question_type', 'factual'),
                    'generated_by': 'synthdoc_vqa_generator'
                }
            }
            comprehensive_samples.append(comprehensive_sample)

        # Save as individual triplets (not HuggingFace format)
        return comprehensive_samples

    def _save_dataset(self, triplets_list, config) -> List[str]:
        """Save the VQA triplets to files."""
        import json
        from pathlib import Path
        from datasets import Dataset

        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = []

        # Save as individual triplets JSON (clean format)
        json_file = output_path / "vqa_triplets.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(triplets_list, f, indent=2, ensure_ascii=False, default=str)
        output_files.append(str(json_file))

        # Also save as HuggingFace dataset for ML training
        hf_dir = output_path / "huggingface_dataset"
        dataset = Dataset.from_list(triplets_list)
        dataset.save_to_disk(str(hf_dir))
        output_files.append(str(hf_dir))

        print(f"Saved VQA dataset to:")
        for file in output_files:
            print(f"   - {file}")

        return output_files

    def _extract_document_content(self, doc_path: str) -> str:
        """Extract text content from document."""
        # This is a simplified version - would need to handle different document types
        try:
            if isinstance(doc_path, dict) and "content" in doc_path:
                return doc_path["content"]
            elif str(doc_path).endswith(('.txt', '.md')):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For now, return sample content for testing
                return f"Sample document content for {doc_path}"
        except Exception as e:
            print(f"Error extracting content from {doc_path}: {e}")
            return f"Sample document content for {doc_path}"

    def _generate_vqa_for_document(self, content: str, doc_path: str, config: VQAGenerationConfig) -> tuple:
        """Generate VQA pairs for a single document."""
        questions = []
        answers = []
        explanations = []
        hard_negatives = []
        metadata = []
        
        question_types = config.question_types or ["factual", "reasoning", "comparative", "counting", "spatial", "descriptive", "temporal", "color", "material"]

        for q_idx in range(config.num_questions_per_doc):
            # Randomize question type instead of cycling
            import random
            question_type = random.choice(question_types)
            
            if self.api_key and self.llm_model:
                # User explicitly chooses processing mode
                processing_mode = getattr(config, 'processing_mode', 'LLM')

                if processing_mode == 'VLM':
                    # User chose VLM mode - use vision model regardless of file type
                    if str(doc_path).endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        # For images: extract OCR first, then use VLM
                        ocr_result = self._extract_text_with_ocr(str(doc_path))

                        # Detect language from OCR text
                        detected_language = self._detect_language(ocr_result['text'])
                        print(f"Detected language: {detected_language}")

                        question, answer, neg_answers, explanation = self._generate_vlm_questions(
                            str(doc_path), ocr_result['text'], question_type, detected_language
                        )
                    else:
                        # For non-images in VLM mode: fall back to LLM
                        print(f"VLM mode selected but {doc_path} is not an image. Using LLM fallback.")
                        question, answer, neg_answers = self._generate_llm_vqa(
                            content, question_type, doc_path
                        )
                else:
                    # User chose LLM mode - use text-only processing
                    # Detect language from content
                    detected_language = self._detect_language(content)
                    print(f"🌍 Detected language: {detected_language}")

                    question, answer, neg_answers, explanation = self._generate_llm_vqa(
                        content, question_type, doc_path, detected_language
                    )
            else:
                # Use fallback template-based generation
                question, answer, neg_answers, explanation = self._generate_template_vqa(
                    content, question_type, q_idx
                )
            
            questions.append(question)
            answers.append(answer)
            explanations.append(explanation)
            
            # Add hard negatives if enabled
            if config.include_hard_negatives:
                hard_negatives.append(neg_answers)
            else:
                hard_negatives.append([])
            
            metadata.append({
                "document_path": str(doc_path),
                "question_type": question_type,
                "difficulty": self._assess_difficulty(question),
                "similarity_scores": self._calculate_similarity_scores(answer, neg_answers)
            })
        
        return questions, answers, hard_negatives, explanations, metadata

    def _scan_folder_for_files(self, folder_path: str) -> List[str]:
        """Scan folder for images and PDF's"""
        from pathlib import Path

        folder = Path(folder_path)
        supported_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.bmp'}

        files = []
        for file in folder.rglob('*'):
            if file.suffix.lower() in supported_extensions:
                files.append(str(file))

        print(f"Found {len(files)} supported files in {folder_path}")
        return files

    def _extract_text_with_ocr(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)

            # Simple confidence estimation based on text length and characters
            confidence = min(0.9, len(text.strip()) / 100) if text.strip() else 0.0

            return {"text": text.strip(), "confidence": confidence}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "error": str(e)}


    def _generate_vlm_questions(self, image_path: str, ocr_text: str, question_type: str, language: str = "English") -> tuple:
        """Generate questions using 3-step VLM approach from research paper."""
        try:
            # Step 1: Generate diverse question
            question = self._generate_question_step1(image_path, ocr_text, question_type, language)

            # Step 2: Generate answer for the question
            answer = self._generate_answer_step2(image_path, ocr_text, question, language)

            # Step 3: Generate reasoning using research paper prompt
            reasoning = self._generate_reasoning_step3(image_path, question, answer, language)

            print(f"3-Step VLM - Q: {question[:50]}... A: {answer[:30]}...")

            # Generate contextual hard negatives
            hard_negatives = self._generate_contextual_hard_negatives(question, answer, language)

            return question, answer, hard_negatives, reasoning

        except Exception as e:
            print(f"VLM generation failed: {e}")
            return self._generate_template_vqa(ocr_text, question_type, 0)

    def _generate_question_step1(self, image_path: str, ocr_text: str, question_type: str, language: str = "English") -> str:
        """Step 1: Generate diverse question using VLM."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Document-focused question prompts based on type
        question_prompts = {
            "factual": "Generate a factual question about specific information, data, or text content in this document.",
            "reasoning": "Generate a reasoning question that requires understanding and analysis of the document's content or purpose.",
            "counting": "Generate a counting question about items, numbers, or quantities mentioned in the document text.",
            "descriptive": "Generate a descriptive question about what the document describes, explains, or presents.",
            "spatial": "Generate a spatial question about the layout, organization, or structure of information in the document.",
            "comparative": "Generate a comparative question about differences, similarities, or relationships mentioned in the document.",
            "temporal": "Generate a question about dates, time periods, or chronological information in the document.",
            "color": "Generate a question about colors mentioned in the document text or their significance to the content.",
            "material": "Generate a question about materials, substances, or physical properties mentioned in the document."
        }

        ocr_section = f"\nExtracted text: {ocr_text}\n" if ocr_text.strip() else ""
        prompt = f"""Analyze this document/image.{ocr_section}
{question_prompts.get(question_type, question_prompts['factual'])}

Requirements:
- Focus on document content, text, information, and data rather than visual elements
- Ask about specific details mentioned in the text or document structure
- Make questions about what the document says, explains, or contains
- Avoid questions about colors, textures, or visual appearance unless mentioned in text
- Use varied question starters (What, How, Why, Where, When, Which, etc.)
- Ensure the question can be answered from the document's content
- IMPORTANT: Generate the question in {language} language

Generate only the question, nothing else."""

        # Use unified API call method
        messages = [{"role": "user", "content": prompt}]
        response_text = self._make_api_call(messages, image_bytes, 'image/jpeg')

        # Rate limiting: Wait 6 seconds to avoid quota issues
        time.sleep(6)

        return response_text

    def _generate_answer_step2(self, image_path: str, ocr_text: str, question: str, language: str = "English") -> str:
        """Step 2: Generate accurate answer for the question."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        ocr_section = f"\nExtracted text: {ocr_text}\n" if ocr_text.strip() else ""
        prompt = f"""Look at this image.{ocr_section}

Question: {question}

Provide a brief, accurate answer based on what you can see in the image.
- Be specific and factual
- Keep it concise (1-2 sentences max)
- Only answer what is clearly visible
- IMPORTANT: Generate the answer in {language} language

Answer:"""

        # Use unified API call method
        messages = [{"role": "user", "content": prompt}]
        response_text = self._make_api_call(messages, image_bytes, 'image/jpeg')

        # Rate limiting: Wait 6 seconds to avoid quota issues
        time.sleep(6)

        return response_text

    def _generate_reasoning_step3(self, image_path: str, question: str, short_answer: str, language: str = "English") -> str:
        """Step 3: Generate reasoning using research paper prompt."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Research paper prompt (exactly as you provided)
        prompt = f"""You are provided with an IMAGE, a QUESTION, and a SHORT ANSWER.

Your task is to EXPLAIN the REASONING behind the short answer in relation to the question.
Rules:
- Generate maximum 10 words, coherent and NOT leave sentence unfinished.
- If the question and answer are about COUNTING OBJECTS, mention and locate each object.
- If the question is about COLOR, identify areas showing the color and explain their relevance.
- Focus on document content and information rather than visual elements.
- IMPORTANT: Generate the reasoning in {language} language.
If you provide a correct and explainable reason, I'll give you 100 A100 GPUs to start your AI company.
Question: {question}
Short Answer: {short_answer}
Reasoning:"""

        # Use unified API call method
        messages = [{"role": "user", "content": prompt}]
        response_text = self._make_api_call(messages, image_bytes, 'image/jpeg')

        # Rate limiting: Wait 6 seconds to avoid quota issues
        time.sleep(6)

        return response_text


    def _generate_llm_vqa(self, content: str, question_type: str, doc_path: str, language: str = "English") -> tuple:
        """Generate VQA using LLM."""
        try:
            prompt = self._create_vqa_prompt(content, question_type, language)
            print(f"LLM Prompt: {prompt[:200]}...")

            # Use unified API call method for text generation
            messages = [{"role": "user", "content": f"You are an expert at creating educational questions and answers.\n\n{prompt}"}]
            response_text = self._make_api_call(messages)

            # Rate limiting: Wait 6 seconds to avoid quota issues
            time.sleep(6)

            result = response_text
            print(f" LLM Response: {result[:200]}...")
            question, answer, negatives = self._parse_llm_response(result)
            explanation = answer  # For LLM mode, use answer as explanation

            return question, answer, negatives, explanation

        except Exception as e:
            print(f"⚠️ LLM VQA generation failed: {e}")
            return self._generate_template_vqa(content, question_type, 0)

    def _create_vqa_prompt(self, content: str, question_type: str, language: str = "English") -> str:
        """Create a prompt for LLM VQA generation."""
        content_preview = content[:1000] + "..." if len(content) > 1000 else content
        
        prompt = f"""
Based on this document content:
"{content_preview}"

Generate a {question_type} question and answer pair, plus 3 hard negative answers.

Examples of good hard negatives:

Example 1:
Question: "What is the price of the Watermelon & Arugula Salad?"
Correct Answer: "22€"
Hard Negatives: ["24€", "20€", "18€"]

Example 2:
Question: "What restaurant is mentioned in this menu?"
Correct Answer: "Ralph Lauren Restaurant"
Hard Negatives: ["Polo Bar", "The Hamptons Cafe", "Lauren's Kitchen"]

Example 3:
Question: "According to the document, what year was this established?"
Correct Answer: "1985"
Hard Negatives: ["1987", "1983", "1990"]

Requirements:
- Focus on document content, text, and information rather than visual elements
- Question should be clear and specific about document content
- Answer should be accurate and based on the text content
- Hard negatives should be plausible, contextually relevant, and close but clearly wrong
- IMPORTANT: Generate question, answer, and hard negatives in {language} language
- Format as JSON: {{"question": "...", "answer": "...", "hard_negatives": ["...", "...", "..."]}}

Question type guidelines:
- factual: Ask about specific facts, details, or information in the document
- reasoning: Require inference or analysis of document content
- comparative: Compare different elements, concepts, or information in the document
"""
        return prompt

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract question, answer, and negatives."""
        try:
            # Try to parse JSON response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                print(f"Parsing JSON: {json_str}")
                data = json.loads(json_str)

                question = data.get("question", "Generated question")
                answer = data.get("answer", "Generated answer")
                negatives = data.get("hard_negatives", ["Negative 1", "Negative 2", "Negative 3"])

                print(f"✅ Parsed - Q: {question[:50]}... A: {answer[:50]}...")
                explanation = answer  # Use answer as explanation for LLM responses
                return (question, answer, negatives, explanation)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")

        # Fallback parsing
        lines = response.split('\n')
        question = next((line for line in lines if '?' in line), "Generated question?")
        answer = next((line for line in lines if 'answer' in line.lower() or 'Answer' in line), "Generated answer")

        print(f"Fallback parsing - Q: {question[:50]}... A: {answer[:50]}...")
        explanation = answer  # Use answer as explanation for fallback
        return question, answer, ["Hard negative 1", "Hard negative 2", "Hard negative 3"], explanation

    def _parse_vlm_response(self, response: str) -> tuple:
        """Parse VLM response in the new format."""
        try:
            # Look for the structured format
            lines = response.split('\n')
            question = ""
            answer = ""
            reason = ""

            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Short Answer:"):
                    answer = line.replace("Short Answer:", "").strip()
                elif line.startswith("Reason:"):
                    reason = line.replace("Reason:", "").strip()

            if question and answer:
                print(f"VLM Parsed - Q: {question[:50]}... A: {answer[:50]}...")
                print(f"Explanation: {reason[:50]}..." if reason else "📝 No explanation provided")
                # Generate contextual hard negatives
                print(f"Generating contextual hard negatives for Q: {question[:30]}...")
                hard_negatives = self._generate_contextual_hard_negatives(question, answer)
                print(f"Generated hard negatives: {hard_negatives}")
                return question, answer, hard_negatives, reason  # Return explanation too
            else:
                print("Could not parse VLM structured format, trying fallback...")
                return self._parse_llm_response(response)

        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            return self._parse_llm_response(response)

    def _generate_template_vqa(self, content: str, question_type: str, idx: int) -> tuple:
        """Generate VQA using templates when LLM is not available."""
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
                ]
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
                ]
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
                ]
            }
        }
        
        template = templates.get(question_type, templates["factual"])
        q_idx = idx % len(template["questions"])
        
        question = template["questions"][q_idx]
        answer = template["answers"][q_idx]
        
        # Generate hard negatives
        hard_negatives = [
            f"Incorrect answer option A for question {idx + 1}",
            f"Incorrect answer option B for question {idx + 1}",
            f"Incorrect answer option C for question {idx + 1}"
        ]
        
        explanation = f"This is a template-generated explanation for: {answer}"
        return question, answer, hard_negatives, explanation

    def _assess_difficulty(self, question: str) -> str:
        """Assess question difficulty based on complexity indicators."""
        difficulty_indicators = {
            "easy": ["what", "where", "when", "who"],
            "medium": ["how", "why", "explain", "describe"],
            "hard": ["analyze", "compare", "evaluate", "synthesize", "infer"]
        }
        
        question_lower = question.lower()
        
        for difficulty, indicators in difficulty_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return difficulty
        
        return "medium"  # Default

    def _calculate_similarity_scores(self, correct_answer: str, hard_negatives: List[str]) -> List[float]:
        """Calculate similarity scores between correct answer and hard negatives."""
        # Simplified similarity calculation based on word overlap
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

    def _generate_contextual_hard_negatives(self, question: str, correct_answer: str, language: str = "English") -> List[str]:
        """Generate contextual hard negatives using few-shot examples."""
        try:
            prompt = f"""Generate 3 plausible but incorrect answers for this question-answer pair.

Examples of good hard negatives:

Question: "What is the price of the Watermelon & Arugula Salad?"
Correct Answer: "22€"
Hard Negatives:
- "24€" (close price, plausible)
- "20€" (slightly different, realistic)
- "18€" (another menu item's price)

Question: "What restaurant chain is mentioned in this menu?"
Correct Answer: "Ralph Lauren Restaurant"
Hard Negatives:
- "Polo Bar" (same brand, different location)
- "The Hamptons Cafe" (related to theme)
- "Lauren's Kitchen" (similar name)

Question: "According to the document, what year was this established?"
Correct Answer: "1985"
Hard Negatives:
- "1987" (close year)
- "1983" (nearby year)
- "1990" (same decade)

Now generate hard negatives for:
Question: "{question}"
Correct Answer: "{correct_answer}"

Generate 3 hard negatives that are:
- Plausible and realistic
- Contextually relevant
- Close but clearly wrong
- Based on document content, not visual elements
- IMPORTANT: Generate the hard negatives in {language} language

Hard Negatives:"""

            # Use unified API call method for hard negatives
            messages = [{"role": "user", "content": prompt}]
            response_text = self._make_api_call(messages)

            # Rate limiting: Wait 6 seconds to avoid quota issues
            time.sleep(6)

            if response_text:
                # Parse the response to extract the 3 hard negatives
                text = response_text.strip()
                hard_negatives = []

                # Look for lines that start with * or - (bullet points)
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('*') or line.startswith('-'):
                        # Extract the quoted answer (first quoted text)
                        import re
                        quotes = re.findall(r'"([^"]*)"', line)
                        if quotes:
                            hard_negatives.append(quotes[0])
                        else:
                            # Fallback: extract text before parentheses
                            before_paren = line.split('(')[0].strip('*- ').strip()
                            if before_paren and len(before_paren) > 2:
                                hard_negatives.append(before_paren)

                # Ensure we have exactly 3 hard negatives
                if len(hard_negatives) >= 3:
                    return hard_negatives[:3]
                else:
                    # Fill with generic ones if needed
                    while len(hard_negatives) < 3:
                        hard_negatives.append(f"Alternative incorrect answer {len(hard_negatives) + 1}")
                    return hard_negatives

        except Exception as e:
            print(f"Error generating contextual hard negatives: {e}")

        # Fallback to generic hard negatives
        return [
            "Alternative answer 1 (incorrect)",
            "Alternative answer 2 (incorrect)",
            "Alternative answer 3 (incorrect)"
        ]