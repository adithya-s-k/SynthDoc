"""
Document generators for different types of synthetic documents.

This module contains generators for raw documents, layout-based documents,
VQA datasets, and handwritten documents.
"""


from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import litellm
import os
import random
from config import DocumentConfig
from PIL import Image, ImageDraw
from font import font, title_font
from datasets import Dataset 
from augmentations import Augmentor, AugmentationType 
from utils import image_to_base64
from image_gen import create_document_image_with_content
from tables import text_to_html, text_to_markdown


logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Base class for all document generators."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate documents."""
        pass


class DocumentGenerator:
    """Generator for raw document content using LLMs."""

    def __init__(self, groq_api_key: str):
        os.environ["GROQ_API_KEY"] = groq_api_key
        litellm.set_verbose = False 
        
        # Set encoding for litellm
        os.environ["PYTHONIOENCODING"] = "utf-8"

        self.language_prompts = {
            'en': "Write a comprehensive, well-structured document about",
            'es': "Escribe un documento integral y bien estructurado sobre",
            'fr': "Rédigez un document complet et bien structuré sur",
            'de': "Schreiben Sie ein umfassendes, gut strukturiertes Dokument über",
            'hi': "इस विषय पर एक व्यापक और अच्छी तरह से संरचित दस्तावेज़ लिखें",
            'sa': "अस्मिन् विषये व्यापकं सुसंस्कृतं च प्रलेखं लिखत"
        }
        self.augmentor = Augmentor()

    def generate_text_content(self, config: DocumentConfig) -> str:
        """Generate high-quality text content using LiteLLM + Groq"""
        try:
            if config.prompt:
                topic = config.prompt
            else:
                topic = random.choice(self.topics)
            
            base_prompt = self.language_prompts.get(config.language, self.language_prompts['en'])
            
            words_per_page = 300
            target_words = config.num_pages * words_per_page
            
            #ensures proper utf8 handling in prompt
            full_prompt = f"""
                {base_prompt} {topic}.

                Create a professional document with the following requirements:
                - Write approximately {target_words} words
                - Include a clear title and section headings
                - Use formal, academic tone
                - Provide specific examples and case studies
                - Include statistics or data where relevant
                - Structure with introduction, main body, and conclusion
                - Write in {config.language} language
                - Make it informative and engaging for professionals in the field

                Format the document with proper paragraphs and natural flow.
                """

            # Ensure messages are properly encoded
            system_msg = f"You are an expert professional writer who creates high-quality, informative documents. Always write in {config.language} language with proper structure and formatting."
            
            # Try to encode/decode to ensure UTF-8 compatibility
            try:
                system_msg_bytes = system_msg.encode('utf-8')
                full_prompt_bytes = full_prompt.encode('utf-8')
                system_msg = system_msg_bytes.decode('utf-8')
                full_prompt = full_prompt_bytes.decode('utf-8')
            except UnicodeError:
                print("⚠️ Unicode encoding issue in prompt")

            response = litellm.completion(
                model="groq/llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=8000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Ensure content is properly decoded
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            content = content.strip()
            
            return content
            
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            print(f"⚠️ Error type: {type(e)}")
            # return self._generate_fallback_content(config, config.prompt or random.choice(self.topics))

    def create_document_image(self, text: str, page_width: int = 800, page_height: int = 1000) -> tuple[Image.Image, dict]:
        """Create a professional-looking document image with proper font support"""
        img = Image.new('RGB', (page_width, page_height), 'white')
        draw = ImageDraw.Draw(img)
        margin = 60
        line_height = 24  #Slightly more for nonLatin scripts
        max_width = page_width - 2 * margin
        
        #Better text wrapping with unicode support
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font:
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_width = bbox[2] - bbox[0]
                except Exception as e:
                    # Fallback width calculation
                    line_width = len(test_line) * 8  # Conservative estimate
            else:
                line_width = len(test_line) * 8  # Approximate
                
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))

        # Draw text with better formatting
        y_position = margin
        text_coordinates = []
        
        for i, line in enumerate(lines):
            if y_position > page_height - margin - line_height:
                break
                
            # Make first line larger (title)
            current_font = title_font if i == 0 and title_font else font
            
            try:
                if current_font:
                    draw.text((margin, y_position), line, fill='black', font=current_font)
                else:
                    # Basic fallback - won't handle complex scripts well
                    try:
                        # Simple character-by-character rendering
                        x_offset = 0
                        for char in line:
                            try:
                                draw.text((margin + x_offset, y_position), char, fill='black')
                                x_offset += 8
                            except:
                                # Skip problematic characters
                                x_offset += 8
                                continue
                    except:
                        # Ultimate fallback
                        draw.text((margin, y_position), "Text rendering error", fill='black')
            except Exception as e:
                print(f"⚠️ Text rendering error for line {i}: {e}")
                continue
            
            # Store text coordinates for each word
            for word_idx, word in enumerate(line.split()):
                text_coordinates.append({
                    "word": word,
                    "x": margin + word_idx * 50,  # Approximate positioning
                    "y": y_position,
                    "line": i
                })
            
            y_position += line_height

        layout_info = {
            "text_coordinates": text_coordinates,
            "layout_type": "single_column",
            "page_dimensions": {"width": page_width, "height": page_height},
            "margins": {"top": margin, "bottom": margin, "left": margin, "right": margin},
            "line_height": line_height,
            "total_lines": len(lines)
        }

        return img, layout_info
    

    def generate_pages(self, text: str, config: DocumentConfig) -> List[Dict]:
        """Generate multiple pages from text content"""
        pages = []
        
        # Calculate text per page (approximate)
        words_per_page = 1000  
        words = text.split()
        
        # Content variety tracking
        used_graph_types = set()
        used_table_types = set()
        
        # Split text into chunks for pages
        for page_num in range(config.num_pages):
            start_idx = page_num * words_per_page
            end_idx = min((page_num + 1) * words_per_page, len(words))
            
            if start_idx >= len(words):
                # Generate NEW unique content for additional pages
                additional_topics = [
                    "implementation challenges and solutions",
                    "case studies and real-world applications", 
                    "future trends and emerging technologies",
                    "technical specifications and requirements",
                    "best practices and methodologies",
                    "comparative analysis and benchmarking",
                    "risk assessment and mitigation strategies",
                    "performance optimization techniques"
                ]
                topic_suffix = additional_topics[page_num % len(additional_topics)]
                
                # Generate completely new content
                extended_config = DocumentConfig(
                    language=config.language,
                    num_pages=1,
                    prompt=f"{config.prompt or 'Advanced analysis'} - {topic_suffix}",
                    include_graphs=config.include_graphs,
                    include_tables=config.include_tables
                )
                page_text = self.generate_text_content(extended_config)
            else:
                page_text = " ".join(words[start_idx:end_idx])
            
            # Add unique page header
            page_text = f"Page {page_num + 1} - Section {page_num + 1}\n\n{page_text}"
            
            # Create page image with unique content
            page_img, layout_info = create_document_image_with_content(
                    page_text, config, page_num, used_graph_types, used_table_types
                )
              # Apply augmentations
            if config.augmentations:
                page_img = self.augmentor.apply_augmentations(page_img, config.augmentations)
            
            pages.append({
                'image': image_to_base64(page_img),
                'text': page_text,
                'page_number': page_num,
                'layout_info': layout_info
            })
        
        return pages

    def generate_document(self, config: DocumentConfig) -> List[Dict[str, Any]]:
        """Generate document with multiple pages"""
        print(f"🔄 Generating: {config.language}, {config.num_pages} pages")
        
        # Generate base text content
        text_content = self.generate_text_content(config)
        
        # Generate multiple pages
        pages = self.generate_pages(text_content, config)
        
        # Create document entries for each page
        documents = []
        pdf_name = config.pdf_name or f"document_{config.language}_{random.randint(10000, 99999)}.pdf"
        
        for page_data in pages:
            html_content = text_to_html(page_data['text'], config, page_data['page_number'])
            markdown_content = text_to_markdown(page_data['text'], config, page_data['page_number'])
            doc = {
                'image': page_data['image'],
                'text': page_data['text'],
                'html': html_content,  
                'markdown': markdown_content,
                'pdf_name': pdf_name,
                'language': config.language,
                'page_number': page_data['page_number'],
                'total_pages': config.num_pages,  
                'word_count': len(page_data['text'].split()),
                'layout_info': page_data['layout_info'],
                'augmentations': [aug.value for aug in config.augmentations] if config.augmentations else [],
                'metadata': {
                    'page_size': "800x1000",
                    'num_pages': config.num_pages,
                    'generation_method': 'synthetic',
                    'topic': config.prompt,
                    'has_graphs': config.include_graphs,
                    'has_tables': config.include_tables,
                    'has_html': True,  
                    'has_markdown': True
                }
            }
            documents.append(doc)
        
        return documents
    

    def generate_dataset(self, configs: List[DocumentConfig]) -> Dataset:
        """Generate HuggingFace dataset"""
        documents = []
        
        print(f"🚀 Starting generation of {len(configs)} documents...")
        
        for i, config in enumerate(configs, 1):
            try:
                print(f"📄 Processing document {i}/{len(configs)}")
                doc_pages = self.generate_document(config)
                documents.extend(doc_pages)
                print(f"✅ Document {i} completed ({len(doc_pages)} pages)")

            except Exception as e:
                print(f"❌ Error generating document {i}: {e}")
                print(f"❌ Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                continue 

        if not documents:
            print("❌ No documents were generated!")
            return None
            
        print(f"\n✅ Successfully generated {len(documents)} documents")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(documents)
        return dataset 
#Document generator ends
######################################################################################

class LayoutGenerator(BaseGenerator):
    """Generator for layout-based document transformations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

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

        Returns:
            HuggingFace dataset with layout annotations
        """
        self.logger.info("Augmenting document layouts")

        # TODO: Implement layout augmentation logic
        dataset = {
            "images": [],
            "annotations": [],
            "metadata": {
                "languages": languages,
                "fonts": fonts,
                "augmentations": augmentations,
                "templates": layout_templates,
            },
        }

        return dataset

    def augment_pdfs(
        self,
        corpus_paths: List[Union[str, Path]],
        extraction_elements: List[str],
        combination_strategy: str,
        output_layout_types: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Create new documents by recombining PDF elements.

        Returns:
            HuggingFace dataset with recombined documents
        """
        self.logger.info("Augmenting PDFs with element recombination")

        # TODO: Implement PDF element extraction and recombination
        dataset = {
            "images": [],
            "annotations": [],
            "metadata": {
                "source_count": len(corpus_paths),
                "extraction_elements": extraction_elements,
                "combination_strategy": combination_strategy,
                "output_layouts": output_layout_types,
            },
        }

        return dataset

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.augment_layouts(*args, **kwargs)


class VQAGenerator(BaseGenerator):
    """Generator for Visual Question Answering datasets."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize VQAGenerator.

        Args:
            model: Model name for LiteLLM
            api_key: API key for the model provider
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model

        import os
        os.environ["OPENAI_API_KEY"] = api_key

    def generate_vqa_dataset(
        self,
        documents: List[Dict[str, Any]],
        question_types: List[str],
        difficulty_levels: List[str],
        hard_negative_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate VQA datasets with hard negatives.

        Returns:
            Extended dataset with VQA annotations
        """
        self.logger.info(f"Generating VQA dataset for {len(documents)} documents")

        # TODO: Implement VQA generation logic
        vqa_data = {
            "questions": [],
            "answers": [],
            "hard_negatives": [],
            "metadata": {
                "document_count": len(documents),
                "question_types": question_types,
                "difficulty_levels": difficulty_levels,
                "hard_negative_ratio": hard_negative_ratio,
            },
        }

        for doc in documents:
            # Generate questions for each document
            questions = self._generate_questions(doc, question_types, difficulty_levels)
            vqa_data["questions"].extend(questions)

        return vqa_data

    def _generate_questions(
        self,
        document: Dict[str, Any],
        question_types: List[str],
        difficulty_levels: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate questions for a single document using LiteLLM."""
        questions = []

        if not self._llm_enabled:
            # Fallback to sample questions if LLM not available
            for q_type in question_types:
                for difficulty in difficulty_levels:
                    question = {
                        "question": f"Sample {q_type} question ({difficulty})",
                        "answer": "Sample answer",
                        "type": q_type,
                        "difficulty": difficulty,
                        "document_id": document.get("id"),
                    }
                    questions.append(question)
            return questions

        document_content = document.get("content", "")

        for q_type in question_types:
            for difficulty in difficulty_levels:
                try:
                    # Create system prompt for VQA generation
                    system_prompt = f"""You are a VQA dataset generator. Generate a {q_type} question with {difficulty} difficulty level about the given document content. 
                    
                    Question types:
                    - factual: Direct questions about facts in the document
                    - reasoning: Questions requiring logical inference
                    - comparative: Questions comparing different parts of the document
                    - spatial: Questions about layout and positioning
                    - counting: Questions about quantities in the document
                    
                    Difficulty levels:
                    - easy: Simple, direct questions
                    - medium: Requires some understanding of context
                    - hard: Requires complex reasoning or inference
                    
                    Return ONLY a JSON object with "question" and "answer" keys."""

                    user_prompt = f"""Document content: {document_content[:1000]}...
                    
                    Generate a {q_type} question with {difficulty} difficulty about this document."""

                    # Call LiteLLM
                    response = litellm.completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=200,
                        temperature=0.7,
                    )

                    content = response.choices[0].message.content.strip()

                    # Try to parse JSON response
                    import json

                    try:
                        qa_data = json.loads(content)
                        question_data = {
                            "question": qa_data.get(
                                "question", f"Sample {q_type} question"
                            ),
                            "answer": qa_data.get("answer", "Sample answer"),
                            "type": q_type,
                            "difficulty": difficulty,
                            "document_id": document.get("id"),
                        }
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        question_data = {
                            "question": content
                            if len(content) < 200
                            else f"Generated {q_type} question",
                            "answer": "Generated answer",
                            "type": q_type,
                            "difficulty": difficulty,
                            "document_id": document.get("id"),
                        }

                    questions.append(question_data)

                except Exception as e:
                    self.logger.warning(
                        f"VQA generation failed for {q_type}/{difficulty}: {e}"
                    )
                    # Fallback question
                    question_data = {
                        "question": f"Sample {q_type} question ({difficulty})",
                        "answer": "Sample answer",
                        "type": q_type,
                        "difficulty": difficulty,
                        "document_id": document.get("id"),
                    }
                    questions.append(question_data)

        return questions

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.generate_vqa_dataset(*args, **kwargs)
