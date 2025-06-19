"""
Document generators for different types of synthetic documents.

This module contains generators for raw documents, layout-based documents,
VQA datasets, and handwritten documents.
"""


from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
import time
from abc import ABC, abstractmethod
import litellm
import os
import random
from config import DocumentConfig
from PIL import Image, ImageDraw
from font import font, title_font
from datasets import Dataset
from huggingface_hub import login, HfApi, create_repo
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

    def __init__(self, groq_api_key: str, hf_token: Optional[str] = None):
        os.environ["GROQ_API_KEY"] = groq_api_key
        litellm.set_verbose = False 
        
        #Setencoding for litellm
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        # Setup Hugging Face authentication
        self.hf_token = hf_token
        if hf_token:
            login(token=hf_token)
            self.hf_api = HfApi()
        else:
            self.hf_api = None

        self.language_prompts = {
            'en': "Write a comprehensive, well-structured document about",
            'es': "Escribe un documento integral y bien estructurado sobre",
            'fr': "Rédigez un document complet et bien structuré sur",
            'de': "Schreiben Sie ein umfassendes, gut strukturiertes Dokument über",
            'hi': "इस विषय पर एक व्यापक और अच्छी तरह से संरचित दस्तावेज़ लिखें",
            'sa': "अस्मिन् विषये व्यापकं सुसंस्कृतं च प्रलेखं लिखत"
        }
        self.augmentor = Augmentor()

    def enhance_prompt(self, original_prompt: str, num_pages: int, language: str) -> str:
        """Enhance a simple prompt to generate enough content for multiple pages"""
        try:
            enhancement_prompt = f"""
            Take this topic: "{original_prompt}"
            
            Create a comprehensive, detailed outline and expanded description that would be suitable for a {num_pages}-page professional document in {language}.
            
            Please expand this into a rich, detailed prompt that includes:
            - Multiple subtopics and sections
            - Specific areas to cover in depth
            - Examples and case studies to include
            - Technical details and specifications
            - Current trends and future implications
            - Best practices and methodologies
            - Challenges and solutions
            - Comparative analysis where relevant
            
            Return only the enhanced, expanded prompt that I can use to generate the full document.
            """
            
            response = litellm.completion(
                model="groq/llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert content strategist who creates comprehensive document outlines."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            print(f"✅ Enhanced prompt: {enhanced_prompt[:100]}...")
            time.sleep(3)  # Rate limit protection
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            # Fallback: manually enhance the prompt
            return f"""
            {original_prompt} - A comprehensive analysis covering:
            1. Introduction and background
            2. Current state and market overview  
            3. Technical specifications and requirements
            4. Implementation strategies and best practices
            5. Case studies and real-world applications
            6. Challenges and solutions
            7. Future trends and emerging technologies
            8. Conclusion and recommendations
            
            Include detailed examples, statistics, and technical depth throughout.
            """

    def generate_text_content(self, config: DocumentConfig) -> str:
        """Generate high-quality text content using LiteLLM + Groq with rate limiting"""
        max_retries = 3
        base_delay = 2 
        
        for attempt in range(max_retries):
            try:
                if config.prompt:
                    topic = self.enhance_prompt(config.prompt, config.num_pages, config.language)
                else:
                    topic = random.choice(self.topics)
                
                base_prompt = self.language_prompts.get(config.language, self.language_prompts['en'])
                
                words_per_page = 300
                target_words = config.num_pages * words_per_page
                
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

                system_msg = f"You are an expert professional writer who creates high-quality, informative documents. Always write in {config.language} language with proper structure and formatting."
                
                #encode/decode to ensure UTF-8 compatibility
                try:
                    system_msg_bytes = system_msg.encode('utf-8')
                    full_prompt_bytes = full_prompt.encode('utf-8')
                    system_msg = system_msg_bytes.decode('utf-8')
                    full_prompt = full_prompt_bytes.decode('utf-8')
                except UnicodeError:
                    print("⚠️ Unicode encoding issue in prompt")

                #Add delay before API call
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏱️ Rate limit hit, waiting {delay} seconds before retry {attempt + 1}...")
                    time.sleep(delay)
                else:
                    time.sleep(2) 

                print(f"🤖 Generating content (attempt {attempt + 1}/{max_retries})...")
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
                
                if content and len(content.strip()) > 50:  # Ensure we got meaningful content
                    print(f"✅ Generated {len(content)} characters")
                    return content
                else:
                    raise ValueError("Generated content too short or empty")
                    
            except Exception as e:
                print(f"⚠️ API Error (attempt {attempt + 1}): {e}")
                print(f"⚠️ Error type: {type(e)}")
                
                if attempt == max_retries - 1:  # Last attempt
                    print("❌ All retries failed, using fallback content")
                    # Return fallback content instead of failing completely
                    fallback_content = f"""
                    {topic}
                    
                    This document provides an overview of {topic}. Due to API limitations, this is fallback content.
                    
                    Introduction
                    {topic} is an important subject that requires detailed analysis and understanding.
                    The complexity of this topic necessitates a comprehensive approach to ensure proper coverage.
                    
                    Main Content
                    The key aspects of {topic} include various elements that need to be considered carefully.
                    This section would normally contain detailed information generated by AI, but due to rate limits,
                    we are providing this comprehensive fallback content that still maintains professional quality.
                    
                    Technical Analysis
                    From a technical perspective, {topic} involves multiple components that work together.
                    Understanding these components is crucial for anyone working in this field.
                    The implementation details often require careful consideration of various factors.
                    
                    Practical Applications
                    In practical terms, {topic} has numerous applications across different domains.
                    These applications demonstrate the versatility and importance of this subject area.
                    Real-world implementations often reveal additional complexities not apparent in theoretical discussions.
                    
                    Future Considerations
                    Looking ahead, {topic} continues to evolve with new developments and innovations.
                    These advances promise to open new possibilities and applications.
                    Staying current with these developments is essential for professionals in the field.
                    
                    Conclusion
                    In conclusion, {topic} represents a significant area of study that warrants continued attention.
                    Future research in this area would be beneficial for advancing our understanding.
                    The insights gained from this analysis provide a solid foundation for further exploration.
                    """
                    return fallback_content * (config.num_pages // 2 + 1)  # Scale for page count
                
                # Wait before next attempt
                time.sleep(base_delay * (attempt + 1))
    
    def create_document_image(self, text: str, page_width: int = 800, page_height: int = 1000) -> tuple[Image.Image, dict]:
        """Create a professional-looking document image with detailed layout information"""
        img = Image.new('RGB', (page_width, page_height), 'white')
        draw = ImageDraw.Draw(img)
        margin = 60
        line_height = 24  #slightly more for nonLatin scripts
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
                    #Fallback width calculation
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

        #Draw text with detailed coordinate tracking
        y_position = margin
        text_coordinates = []
        word_bboxes = []
        
        for i, line in enumerate(lines):
            if y_position > page_height - margin - line_height:
                break
                
            # Make first line larger (title)
            current_font = title_font if i == 0 and title_font else font
            
            try:
                if current_font:
                    draw.text((margin, y_position), line, fill='black', font=current_font)
                else:
                    #Basic fallback
                    try:
                        x_offset = 0
                        for char in line:
                            try:
                                draw.text((margin + x_offset, y_position), char, fill='black')
                                x_offset += 8
                            except:
                                x_offset += 8
                                continue
                    except:
                        draw.text((margin, y_position), "Text rendering error", fill='black')
            except Exception as e:
                print(f"⚠️ Text rendering error for line {i}: {e}")
                continue
            
            #store detailed coordinates for each word
            x_offset = margin
            for word_idx, word in enumerate(line.split()):
                # Calculate word width
                if current_font:
                    try:
                        word_bbox = draw.textbbox((0, 0), word, font=current_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        word_height = word_bbox[3] - word_bbox[1]
                    except:
                        word_width = len(word) * 8
                        word_height = line_height
                else:
                    word_width = len(word) * 8
                    word_height = line_height
                
                #store word information as in vivid dataset
                word_info = {
                    "word": word,
                    "x": x_offset,
                    "y": y_position,
                    "width": word_width,
                    "height": word_height,
                    "line": i,
                    "word_in_line": word_idx,
                    "bbox": [x_offset, y_position, x_offset + word_width, y_position + word_height]
                }
                text_coordinates.append(word_info)
                word_bboxes.append(word_info["bbox"])
                
                x_offset += word_width + 8  #add space between words
            
            y_position += line_height

        layout_info = {
            "text_coordinates": text_coordinates,
            "word_bboxes": word_bboxes,
            "layout_type": "single_column",
            "page_dimensions": {"width": page_width, "height": page_height},
            "margins": {"top": margin, "bottom": margin, "left": margin, "right": margin},
            "line_height": line_height,
            "total_lines": len(lines),
            "total_words": len(text_coordinates),
            "text_regions": [
                {
                    "type": "text_block",
                    "bbox": [margin, margin, page_width - margin, y_position],
                    "content": text[:500] + "..." if len(text) > 500 else text
                }
            ],
            "font_info": {
                "main_font": str(font) if font else "default",
                "title_font": str(title_font) if title_font else "default",
                "font_size": 12
            }
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
              #create page image with unique content using advanced layout
            from image_gen import create_multicolumn_document_image
            page_img, layout_info = create_multicolumn_document_image(
                    page_text, config, page_num, used_graph_types, used_table_types
                )            # Apply augmentations
            if config.augmentations:
                page_img = self.augmentor.apply_augmentations(page_img, config.augmentations)
            
            pages.append({
                'image': page_img,  #or it will display bunch of encoded text keep as PIL Image for HuggingFace
                'image_base64': image_to_base64(page_img),  #yep also keep base64 for compatibility
                'text': page_text,
                'page_number': page_num + 1,
                'layout_info': layout_info,
                'layout_type': config.layout_type.value,
                'language': config.language,
                'has_graphs': config.include_graphs,
                'has_tables': config.include_tables
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
    def generate_dataset(self, configs: List[DocumentConfig], auto_upload: bool = False,
                         repo_name: Optional[str] = None,
                         repo_description: str = "Synthetic document dataset",
                         private: bool = False) -> Dataset:
        """Generate HuggingFace dataset with rate limiting"""
        documents = []
        
        print(f"🚀 Starting generation of {len(configs)} documents...")
        
        for i, config in enumerate(configs, 1):
            try:
                print(f"📄 Processing document {i}/{len(configs)}")
                
                # Add delay between documents to avoid rate limits
                if i > 1:
                    print("⏱️ Adding delay to avoid rate limits...")
                    time.sleep(3)  # 3 second delay between documents
                
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
        # Auto-upload if requested
        if auto_upload and self.hf_token:
            if not repo_name:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                repo_name = f"synthetic-docs-{timestamp}"
            
            try:
                print(f"🚀 Auto-uploading to Hugging Face as '{repo_name}'...")
                self.upload_to_huggingface(dataset, repo_name, repo_description, private)
            except Exception as e:
                print(f"❌ Auto-upload failed: {e}")
        elif auto_upload and not self.hf_token:
            print("⚠️ Auto-upload requested but no HF token provided during initialization")
        
        return dataset
    def upload_to_huggingface(self, dataset: Dataset, repo_name: str, 
                         repo_description: str = "Synthetic document dataset",
                         private: bool = False) -> str:
        """Upload dataset to Hugging Face Hub, appending to existing if it exists"""
        if not self.hf_api:
            raise ValueError("Hugging Face token not provided during initialization")
        
        try:
            # Create repository if it doesn't exist
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.hf_token,
                private=private,
                exist_ok=True
            )
            
            # Try to load existing dataset and append
            try:
                print(f"🔍 Checking if dataset '{repo_name}' already exists...")
                existing_dataset = Dataset.from_hub(repo_name, token=self.hf_token)
                print(f"📊 Found existing dataset with {len(existing_dataset)} entries")
                
                # Combine datasets
                from datasets import concatenate_datasets
                combined_dataset = concatenate_datasets([existing_dataset, dataset])
                print(f"🔄 Combined dataset now has {len(combined_dataset)} entries")
                
                # Push combined dataset
                combined_dataset.push_to_hub(
                    repo_id=repo_name,
                    token=self.hf_token,
                    private=private
                )
                print(f"✅ Dataset appended successfully! Total entries: {len(combined_dataset)}")
                
            except Exception as load_error:
                print(f"📝 No existing dataset found, creating new one...")
                # Push new dataset
                dataset.push_to_hub(
                    repo_id=repo_name,
                    token=self.hf_token,
                    private=private
                )
                print(f"✅ New dataset uploaded successfully with {len(dataset)} entries")
            
            print(f"🌐 Dataset URL: {repo_url}")
            return repo_url
        
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
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