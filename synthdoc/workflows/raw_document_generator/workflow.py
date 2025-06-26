import os
import time
import random
import io
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import litellm
from datasets import Dataset, concatenate_datasets
from huggingface_hub import create_repo, login, HfApi
from ..base import BaseWorkflow
from ...models import RawDocumentGenerationConfig, WorkflowResult
from ...languages import Language, get_language_fonts
from .image_gen import create_multicolumn_document_image


class RawDocumentGenerator(BaseWorkflow):
    """Generate synthetic documents from scratch using LLMs."""

    def __init__(self, groq_api_key: str, hf_token: Optional[str] = None, save_dir: str = "generated_docs"):
        super().__init__()
        self.save_dir = save_dir
        self._setup_save_directory()
        self._setup_apis(groq_api_key, hf_token)
        self._setup_language_prompts()
        self._setup_cost_tracking()

    def _setup_save_directory(self):
        """Create save directory for generated documents."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_dir = os.path.join(self.save_dir, "images")
        self.metadata_dir = os.path.join(self.save_dir, "metadata")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        print(f"✅ Save directories created: {self.save_dir}")

    def _setup_apis(self, groq_api_key: str, hf_token: Optional[str]):
        """Initialize API configurations."""
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["PYTHONIOENCODING"] = "utf-8"
        litellm.set_verbose = False
        
        self.hf_token = hf_token
        self.hf_api = None
        if hf_token:
            try:
                login(token=hf_token)
                self.hf_api = HfApi()
                print("✅ Hugging Face authentication successful")
            except Exception as e:
                print(f"⚠️ HF authentication failed: {e}")

    def _setup_language_prompts(self):
        """Setup language-specific prompts."""
        self.language_prompts = {
            Language.EN.value: "Write a comprehensive, well-structured document about",
            Language.ES.value: "Escribe un documento integral y bien estructurado sobre",
            Language.FR.value: "Rédigez un document complet et bien structuré sur",
            Language.DE.value: "Schreiben Sie ein umfassendes, gut strukturiertes Dokument über",
            Language.HI.value: "इस विषय पर एक व्यापक और अच्छी तरह से संरचित दस्तावेज़ लिखें",
            Language.SA.value: "अस्मिन् विषये व्यापकं सुसंस्कृतं च प्रलेखं लिखत"
        }

    def _setup_cost_tracking(self):
        """Initialize cost tracking variables."""
        self.total_cost = 0.0
        self.api_calls_count = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.pricing = {
            "groq/llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
            "groq/llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
            "groq/mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024}
        }

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate API call cost."""
        if model in self.pricing:
            input_cost = (prompt_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (completion_tokens / 1000) * self.pricing[model]["output"]
            return input_cost + output_cost
        return 0.0

    def _track_usage(self, response, model: str = "groq/llama3-8b-8192"):
        """Track API usage and costs."""
        self.api_calls_count += 1
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            self.token_usage["prompt_tokens"] += prompt_tokens
            self.token_usage["completion_tokens"] += completion_tokens
            self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
            
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
            self.total_cost += cost
            return cost
        return 0.0

    def _enhance_prompt(self, prompt: str, num_pages: int, language: Language) -> str:
        """Enhance prompt for better content generation."""
        enhancement_prompt = f"""
        Take this topic: "{prompt}"
        
        Create a comprehensive outline for a {num_pages}-page professional document in {language.value}.
        Include: multiple subtopics, technical details, examples, case studies, current trends,
        best practices, challenges, solutions, and comparative analysis.
        
        Return only the enhanced prompt for document generation.
        """
        
        try:
            response = litellm.completion(
                model="groq/llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert content strategist."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            cost = self._track_usage(response)
            print(f"💰 Prompt Enhancement: ${cost:.6f}")
            time.sleep(2)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            return f"{prompt} - Comprehensive analysis with technical depth and examples"

    def _generate_content(self, config: RawDocumentGenerationConfig) -> str:
        """Generate text content using LLM."""
        try:
            enhanced_topic = self._enhance_prompt(config.prompt or "document", config.num_pages, config.language)
        except:
            enhanced_topic = f"{config.prompt or 'document'} - Comprehensive analysis with technical depth and examples"
            
        base_prompt = self.language_prompts.get(config.language.value, self.language_prompts[Language.EN.value])
        
        words_per_page = 1000  # Even MORE text per page
        target_words = config.num_pages * words_per_page
        
        # Language-specific prompts - MUCH MORE DETAILED
        if config.language == Language.HI:
            full_prompt = f"""
            {base_prompt} {enhanced_topic}.
            लगभग {target_words} शब्द लिखें। बहुत विस्तृत और गहन विश्लेषण लिखें।
            कई पैराग्राफ, उदाहरण, केस स्टडी, तकनीकी विवरण, और व्यापक चर्चा शामिल करें।
            परिचय, मुख्य भाग (कई सेक्शन में), और विस्तृत निष्कर्ष के साथ संरचित करें।
            प्रत्येक सेक्शन में कम से कम 3-4 पैराग्राफ होने चाहिए।
            """
        else:
            full_prompt = f"""
            {base_prompt} {enhanced_topic}.
            Write approximately {target_words} words with EXTENSIVE detail and comprehensive analysis.
            Include multiple detailed sections, each with 4-5 paragraphs minimum.
            Cover: introduction, background, technical details, implementation strategies, 
            case studies, challenges, solutions, best practices, future trends, and detailed conclusions.
            Use formal academic tone with specific examples, statistics, and in-depth explanations.
            Make each paragraph substantial with detailed explanations and examples.
            Focus on creating dense, informative content that fills the entire page with text.
            Write long paragraphs with detailed explanations, technical specifications, 
            comparative analysis, step-by-step processes, and comprehensive coverage.
            """

        system_msg = f"You are an expert writer creating high-quality documents in {config.language.value}."
        if config.language == Language.HI:
            system_msg = "आप एक विशेषज्ञ लेखक हैं जो हिंदी में उच्च गुणवत्ता वाले दस्तावेज़ बनाते हैं।"

        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                
                response = litellm.completion(
                    model="groq/llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=8000,
                    temperature=0.7
                )
                
                cost = self._track_usage(response)
                print(f"💰 Content Generation: ${cost:.6f}")
                
                content = response.choices[0].message.content.strip()
                if len(content) > 50:
                    return content
                    
            except Exception as e:
                print(f"⚠️ API Error (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt
                    raise Exception(f"Failed to generate content after 3 attempts: {e}")
        
        raise Exception("Content generation failed - no fallback content")

    def _create_document_image(self, text: str, config: RawDocumentGenerationConfig, 
                              page_num: int) -> Tuple[Image.Image, Dict]:
        """Create document image with advanced multi-column layout."""
        
        # Use the advanced multicolumn document generation - NO FALLBACK
        # Initialize tracking sets if not exist
        if not hasattr(self, '_used_graph_types'):
            self._used_graph_types = set()
        if not hasattr(self, '_used_table_types'):
            self._used_table_types = set()
        
        # Call the advanced function - MUST WORK
        img, metadata = create_multicolumn_document_image(
            text=text,
            config=config,
            page_num=page_num,
            used_graph_types=self._used_graph_types,
            used_table_types=self._used_table_types,
            page_width=800,
            page_height=1000
        )
        
        # Convert metadata to expected format
        layout_info = {
            "page_dimensions": {"width": 800, "height": 1000},
            "margins": metadata.get('layout_info', {}).get('margin', 60),
            "layout_type": getattr(config, 'layout_type', 'SINGLE_COLUMN'),
            "has_graphs": config.include_graphs,
            "has_tables": config.include_tables,
            "has_ai_images": config.include_ai_images,
            "visual_elements": metadata.get('visual_elements', {}),
            "word_coordinates": metadata.get('word_coords', []),
            "content_height": metadata.get('content_height', 800),
            "visual_elements_height": metadata.get('visual_elements_height', 0),
            "page_metadata": metadata
        }
        
        print(f"✅ Generated advanced document page {page_num} with layout: {getattr(config, 'layout_type', 'SINGLE_COLUMN')}")
        print(f"   📊 Visual elements: graphs={config.include_graphs}, tables={config.include_tables}, images={config.include_ai_images}")
        
        return img, layout_info
        
    def _save_document_data(self, doc_data: Dict[str, Any], page_num: int, pdf_name: str):
        """Save document data incrementally."""
        # Save image
        img = doc_data['image']
        img_filename = f"{pdf_name}_page_{page_num}.png"
        img_path = os.path.join(self.images_dir, img_filename)
        img.save(img_path)
        
        # Save metadata
        metadata = {k: v for k, v in doc_data.items() if k != 'image'}
        metadata['image_path'] = img_path
        metadata_filename = f"{pdf_name}_page_{page_num}_metadata.json"
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved: {img_filename} + metadata")
        return img_path, metadata_path

    def _create_hf_dataset_format(self, samples: list, metadata: dict = None) -> Dataset:
        """Helper method to create HuggingFace dataset format."""
        if samples:
            return Dataset.from_list(samples)
        else:
            return Dataset.from_dict({})

    def _create_document_data(self, text: str, config: RawDocumentGenerationConfig, 
                             page_num: int, pdf_name: str) -> Dict[str, Any]:
        """Create document data structure."""
        img, layout_info = self._create_document_image(text, config, page_num)
        
        # Create required data structures
        lines_data = [{"text": line, "bbox": [60, 60 + i*24, 740, 84 + i*24], 
                      "confidence": 0.95} for i, line in enumerate(text.split('\n')[:30])]
        
        images_data = []
        if config.include_graphs:
            images_data.append({"type": "graph", "bbox": [200, 300, 600, 550], 
                               "description": "Data visualization", "confidence": 0.95})
        
        tables_data = []
        if config.include_tables:
            tables_data.append({"type": "data_table", "bbox": [60, 400, 560, 550], 
                               "rows": 5, "columns": 3, "confidence": 0.95})
        
        content_list = [{"id": i+1, "type": "text", "content": line, 
                        "bbox": [60, 60 + i*24, 740, 84 + i*24]} 
                       for i, line in enumerate(text.split('\n')[:20])]
        
        return {
            'image': img,
            'imagewidth': 800,
            'pdf_name': pdf_name,
            'page_number': page_num - 1,
            'markdown': f"# Page {page_num}\n\n{text}",
            'html': f"<h1>Page {page_num}</h1>\n<p>{text.replace(chr(10), '</p><p>')}</p>",
            'layout': str(layout_info),
            'lines': str(lines_data),
            'images': str(images_data),
            'equations': "[]",
            'tables': str(tables_data),
            'page_size': "800x1000",
            'content_list': str(content_list),
            'base_layout_detection': str({"page_layout": {"type": "single_column"}}),
            'pdf_info': str({"filename": pdf_name, "page_count": config.num_pages}),
            'system_prompt': "Generate high-quality synthetic documents for training ML models",
            'response': text
        }

    def process(self, config: RawDocumentGenerationConfig) -> WorkflowResult:
        """Generate synthetic documents based on configuration."""
        print(f"🔄 Generating: {config.language.value}, {config.num_pages} pages")
        
        # Generate base content
        content = self._generate_content(config)
        
        # Split content into pages
        words = content.split()
        words_per_page = len(words) // config.num_pages if config.num_pages > 0 else len(words)
        
        documents = []
        pdf_name = f"document_{config.language.value}_{random.randint(10000, 99999)}.pdf"
        
        for page_num in range(1, config.num_pages + 1):
            start_idx = (page_num - 1) * words_per_page
            end_idx = min(page_num * words_per_page, len(words))
            
            if start_idx < len(words):
                page_text = " ".join(words[start_idx:end_idx])
            else:
                # Generate additional content for extra pages
                additional_content = self._generate_content(
                    RawDocumentGenerationConfig(
                        language=config.language,
                        num_pages=1,
                        prompt=f"{config.prompt} - Additional analysis"
                    )
                )
                page_text = additional_content[:1000]
            
            page_text = f"Page {page_num}\n\n{page_text}"
            doc_data = self._create_document_data(page_text, config, page_num, pdf_name)
            
            # Save immediately after generation
            self._save_document_data(doc_data, page_num, pdf_name)
            
            documents.append(doc_data)
        
        # Print cost summary
        print(f"\n💰 Total Cost: ${self.total_cost:.6f} | API Calls: {self.api_calls_count}")
        print(f"   Tokens: {self.token_usage['total_tokens']:,}")
        
        # Save generation summary
        summary = {
            "workflow_type": "raw_document_generation",
            "total_cost": self.total_cost,
            "api_calls": self.api_calls_count,
            "tokens_used": self.token_usage['total_tokens'],
            "num_samples": len(documents),
            "config": {
                "language": config.language.value,
                "num_pages": config.num_pages,
                "prompt": config.prompt,
                "include_graphs": config.include_graphs,
                "include_tables": config.include_tables,
                "include_ai_images": config.include_ai_images,
                "output_format": config.output_format.value if hasattr(config.output_format, 'value') else str(config.output_format)
            },
            "generated_files": [f"{pdf_name}_page_{i}.png" for i in range(1, config.num_pages + 1)]
        }
        
        summary_path = os.path.join(self.save_dir, f"{pdf_name}_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"📋 Summary saved: {summary_path}")
        
        # Create dataset
        dataset = self._create_hf_dataset_format(
            documents, 
            {"workflow": "raw_document_generation", "config": summary["config"]}
        )
        
        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "raw_document_generation",
                "total_cost": self.total_cost,
                "api_calls": self.api_calls_count,
                "tokens_used": self.token_usage['total_tokens']
            },
            num_samples=len(documents),
        )

    def upload_to_hub(self, dataset: Dataset, repo_name: str, 
                     private: bool = False) -> str:
        """Upload dataset to Hugging Face Hub."""
        if not self.hf_api:
            raise ValueError("HF token not provided during initialization")
        
        try:
            repo_url = create_repo(repo_id=repo_name, token=self.hf_token, 
                                 private=private, exist_ok=True)
            
            # Try to append to existing dataset
            try:
                existing = Dataset.from_hub(repo_name, token=self.hf_token)
                dataset = concatenate_datasets([existing, dataset])
                print(f"📊 Appended to existing dataset: {len(dataset)} total entries")
            except:
                print(f"📝 Creating new dataset: {len(dataset)} entries")
            
            dataset.push_to_hub(repo_id=repo_name, token=self.hf_token, private=private)
            print(f"✅ Upload successful: {repo_url}")
            return repo_url
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise