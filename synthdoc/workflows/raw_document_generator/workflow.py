import os
import time
import random
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import litellm
from datasets import Dataset
from ..base import BaseWorkflow
from ...models import RawDocumentGenerationConfig, WorkflowResult
from ...languages import Language, get_language_name
from .image_gen import create_multicolumn_document_image
from ...utils import CostTracker


class RawDocumentGenerator(BaseWorkflow):
    """Generate synthetic documents from scratch using LLMs via LiteLLM."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, save_dir: str = "raw_document_output"):
        super().__init__()

        # Auto-detect API key if not provided
        if api_key is None:
            from ...config import get_api_key
            api_key = get_api_key("auto")

        self.save_dir = save_dir
        self.workflow_name = "raw_document_generation"
        self.api_key = api_key
        self.model = model
        self._setup_save_directory()
        self._setup_apis()
        self.base_prompt_template = "Write a comprehensive, well-structured document in {language} about"
        self.cost_tracker = CostTracker()

    def _setup_save_directory(self):
        """Create save directory structure with images folder and metadata.jsonl directly in save_dir."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create images directory and metadata file directly in save_dir (same as other workflows)
        self.images_dir = os.path.join(self.save_dir, "images")
        self.metadata_file = os.path.join(self.save_dir, "metadata.jsonl")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Create metadata.jsonl if it doesn't exist
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file
        
        print(f"✅ Save directory created: {self.save_dir}")
        print(f"📂 Images will be saved to: {self.images_dir}")
        print(f"📄 Metadata will be saved to: {self.metadata_file}")

    def _setup_apis(self):
        """Initialize API configurations for LiteLLM."""
        # Set up API key if provided
        if self.api_key:
            # LiteLLM will auto-detect the provider based on model name
            if self.model and "gemini" in self.model.lower():
                os.environ["GEMINI_API_KEY"] = self.api_key
                os.environ["GOOGLE_API_KEY"] = self.api_key  # LiteLLM also checks this
            elif self.model and "gpt" in self.model.lower():
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.model and "claude" in self.model.lower():
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            else:
                print("No api key found")

        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Configure LiteLLM with built-in retry logic
        litellm.set_verbose = False
        litellm.num_retries = 3  # Built-in retry handling
        litellm.request_timeout = 120  # Longer timeout for document generation
        litellm.drop_params = True  # Auto-handle unsupported parameters

    def _enhance_prompt(self, prompt: str, num_pages: int, language: Language) -> str:
        """Enhance prompt for better content generation."""
        language_name = get_language_name(language.value if isinstance(language, Language) else language)
        
        enhancement_prompt = f"""
        Take this topic: "{prompt}"
        
        Create a comprehensive outline for a {num_pages}-page professional document in {language_name}.
        Include: multiple subtopics, technical details, examples, case studies, current trends,
        best practices, challenges, solutions, and comparative analysis.
        
        Return only the enhanced prompt for document generation.
        """
        
        try:
            # Use instance model if provided, otherwise get from config
            if self.model:
                model = self.model
            else:
                from ...config import get_llm_model
                model = get_llm_model("auto")

            # Use LiteLLM's built-in retry logic
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert content strategist."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
                # Built-in retry and timeout handled by global config
            )

            # Track cost with dynamic pricing
            cost = self.cost_tracker.track_usage(response, model=model)
            print(f"💰 Prompt Enhancement: ${cost:.6f}")
            time.sleep(2)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ Prompt enhancement failed: {e}")
            return f"{prompt} - Comprehensive analysis with technical depth and examples"

    def _generate_content(self, config: RawDocumentGenerationConfig) -> str:
        """Generate text content using LLM with built-in retry."""
        try:
            enhanced_topic = self._enhance_prompt(config.prompt or "document", config.num_pages, config.language)
        except:
            enhanced_topic = f"{config.prompt or 'document'} - Comprehensive analysis with technical depth and examples"
        
        # Use English prompt template with language variable
        language_code = config.language.value if isinstance(config.language, Language) else config.language
        language_name = get_language_name(language_code)
        
        words_per_page = 1000
        target_words = config.num_pages * words_per_page
        
        # Strengthen language instructions for better compliance
        if language_name.lower() == "english":
            lang_instruction = "Write the entire document in English."
        else:
            lang_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT: You MUST write the ENTIRE document in {language_name} language only.
            Do NOT use English or any other language. Every single word, sentence, and paragraph must be in {language_name}.
            If you cannot write in {language_name}, respond with "I cannot generate content in {language_name}".
            """

        # Single prompt with strong language enforcement
        full_prompt = f"""
        {self.base_prompt_template.format(language=language_name)} {enhanced_topic}.
        Write approximately {target_words} words with EXTENSIVE detail and comprehensive analysis.
        Include multiple detailed sections, each with 4-5 paragraphs minimum.

        {lang_instruction}
        """

        # Strong system message with language enforcement
        system_msg = f"""You are an expert multilingual writer. {lang_instruction}

        STRICT RULES:
        1. Write ONLY in {language_name} language
        2. Do NOT mix languages
        3. Do NOT use English unless the target language IS English
        4. Every word must be in {language_name}"""
        
        # Use instance model if provided, otherwise get from config
        if self.model:
            model = self.model
        else:
            from ...config import get_llm_model
            model = get_llm_model("auto")

        # Use LiteLLM's built-in retry logic
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=8000,
            temperature=0.7
            # num_retries and timeout handled by global LiteLLM config
        )

        # Track cost using dynamic pricing
        cost = self.cost_tracker.track_usage(response, model=model)
        print(f"Content Generation: ${cost:.6f}")
        
        content = response.choices[0].message.content.strip()
        if len(content) > 50:
            return content
        else:
            raise Exception("Generated content too short")

    def _create_document_image(self, text: str, config: RawDocumentGenerationConfig, 
                              page_num: int) -> Tuple[Image.Image, Dict]:
        """Create document image with advanced multi-column layout."""
        
        # Use the advanced multicolumn document generation
        if not hasattr(self, '_used_graph_types'):
            self._used_graph_types = set()
        if not hasattr(self, '_used_table_types'):
            self._used_table_types = set()
        
        # Call the advanced function 
        img, metadata = create_multicolumn_document_image(
            text=text,
            config=config,
            page_num=page_num,
            used_graph_types=self._used_graph_types,
            used_table_types=self._used_table_types,
            page_width=800,
            page_height=1000
        )
        
        return img, metadata

    def process(self, config: RawDocumentGenerationConfig) -> WorkflowResult:
        """Generate synthetic documents based on configuration."""
        start_time = time.time()
        
        print(f"🔄 Generating: {config.language.value}, {config.num_pages} pages")
        
        # Generate base content
        content = self._generate_content(config)

        words = content.split()
        words_per_page = len(words) // config.num_pages if config.num_pages > 0 else len(words)
        
        lang_code = config.language.value if hasattr(config.language, 'value') else str(config.language)
        document_id = f"document_{lang_code}_{random.randint(10000, 99999)}"
        
        all_results = []
        
        for page_num in range(1, config.num_pages + 1):
            start_idx = (page_num - 1) * words_per_page
            end_idx = min(page_num * words_per_page, len(words))

            if start_idx < len(words):
                page_text = ' '.join(words[start_idx:end_idx])
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
            
            # Create document image
            img, layout_info = self._create_document_image(page_text, config, page_num)
            
            # Save image to images folder
            filename = f"{document_id}_page_{page_num}.png"
            img_path = os.path.join(self.images_dir, filename)
            img.save(img_path)
            
            # Convert LayoutType enum to string if needed (for JSON serialization)
            layout_type = getattr(config, 'layout_type', 'SINGLE_COLUMN')
            layout_type_str = layout_type.value if hasattr(layout_type, 'value') else str(layout_type)
            
            # Create minimal metadata entry (following pattern from other workflows)
            
            metadata_entry = {
                "file_name": filename,
                "image_path": f"images/{filename}",
                "id": f"{document_id}_page_{page_num}",
                "document_id": document_id,
                "page_number": page_num,
                "language": config.language.value if hasattr(config.language, 'value') else str(config.language),
                "prompt": config.prompt or "document generation",
                "content_preview": page_text[:200] + "..." if len(page_text) > 200 else page_text,
                "layout_type": layout_type_str,
                "has_graphs": getattr(config, 'include_graphs', False),
                "has_tables": getattr(config, 'include_tables', False),
                "has_ai_images": getattr(config, 'include_ai_images', False),
                "generated_by": "synthdoc_raw_document_generator"
            }
            
            # Write to metadata.jsonl
            with open(self.metadata_file, 'a', encoding='utf-8') as f:
                json.dump(metadata_entry, f, ensure_ascii=False)
                f.write('\n')
            
            # Create result for dataset
            result = {
                "image": img,
                "file_name": filename,
                "image_path": f"images/{filename}",
                "id": f"{document_id}_page_{page_num}",
                "document_id": document_id,
                "page_number": page_num,
                "language": config.language.value if hasattr(config.language, 'value') else str(config.language),
                "prompt": config.prompt or "document generation",
                "content_preview": page_text[:200] + "..." if len(page_text) > 200 else page_text,
                "layout_type": layout_type_str,
                "has_graphs": getattr(config, 'include_graphs', False),
                "has_tables": getattr(config, 'include_tables', False),
                "has_ai_images": getattr(config, 'include_ai_images', False),
                "generated_by": "synthdoc_raw_document_generator"
            }
            
            all_results.append(result)
            print(f"Saved: {filename}")

        # Create HuggingFace dataset directly from results
        if all_results:
            dataset_dict = {
                "image": [],
                "file_name": [],
                "image_path": [],
                "id": [],
                "document_id": [],
                "page_number": [],
                "language": [],
                "prompt": [],
                "content_preview": [],
                "layout_type": [],
                "has_graphs": [],
                "has_tables": [],
                "has_ai_images": [],
                "generated_by": []
            }
            
            for result in all_results:
                for key in dataset_dict.keys():
                    dataset_dict[key].append(result[key])
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Clear images from memory after dataset creation
            for result in all_results:
                if 'image' in result and result['image']:
                    result['image'].close()
        else:
            dataset = Dataset.from_dict({})

        cost_summary = self.cost_tracker.get_summary()
        processing_time = time.time() - start_time
        
        print(f"\n✅ Document generation completed: {len(all_results)} pages in {processing_time:.2f}s")
        print(f"💰 Total Cost: ${cost_summary['total_cost']:.6f} | API Calls: {cost_summary['api_calls']}")
        print(f"📊 Tokens: {cost_summary['tokens_used']:,}")
        print(f"📁 Output structure: {self.save_dir}")
        print(f"   - Images: {self.images_dir}")
        print(f"   - Metadata: {self.metadata_file}")
        
        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "raw_document_generation",
                "document_id": document_id,
                "total_pages": len(all_results),
                "language": config.language.value if hasattr(config.language, 'value') else str(config.language),
                "prompt": config.prompt,
                "processing_time": processing_time,
                **cost_summary,
                "output_structure": {
                    "output_dir": self.save_dir,
                    "images_dir": self.images_dir,
                    "metadata_file": self.metadata_file,
                    "total_images": len(all_results)
                },
                "config": {
                    "language": config.language.value if hasattr(config.language, 'value') else str(config.language),
                    "num_pages": config.num_pages,
                    "prompt": config.prompt,
                    "include_graphs": getattr(config, 'include_graphs', False),
                    "include_tables": getattr(config, 'include_tables', False),
                    "include_ai_images": getattr(config, 'include_ai_images', False),
                    "layout_type": layout_type_str
                }
            },
            num_samples=len(all_results),
            output_files=[os.path.join(self.images_dir, result["file_name"]) for result in all_results]
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
                line = line.strip()
                if line:
                    record = json.loads(line)
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
            "document_id": [r["document_id"] for r in dataset_records],
            "page_number": [r["page_number"] for r in dataset_records],
            "language": [r["language"] for r in dataset_records],
            "prompt": [r["prompt"] for r in dataset_records],
            "content_preview": [r["content_preview"] for r in dataset_records],
            "layout_type": [r["layout_type"] for r in dataset_records],
            "has_graphs": [r["has_graphs"] for r in dataset_records],
            "has_tables": [r["has_tables"] for r in dataset_records],
            "has_ai_images": [r["has_ai_images"] for r in dataset_records],
            "generated_by": [r["generated_by"] for r in dataset_records]
        }
        
        return Dataset.from_dict(dataset_dict)