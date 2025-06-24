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
import io
import os
import random
from config import DocumentConfig
from PIL import Image, ImageDraw
from font import font, title_font, load_font
from datasets import Dataset
from huggingface_hub import login, HfApi, create_repo
from augmentations import Augmentor, AugmentationType 
from utils import image_to_base64
from image_gen import create_document_image_with_content
from tables import text_to_html, text_to_markdown, parse_markdown_text
import warnings

# Suppress Pydantic serializer warnings for cleaner output
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

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
            try:
                login(token=hf_token)
                self.hf_api = HfApi()
                print("✅ Hugging Face authentication successful")
            except Exception as e:
                print(f"⚠️ Hugging Face authentication failed: {e}")
                print("   Continuing without HF integration...")
                self.hf_api = None
                self.hf_token = None
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
        
        # Cost tracking
        self.total_cost = 0.0
        self.api_calls_count = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Groq pricing
        self.pricing = {
            "groq/llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
            "groq/llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
            "groq/mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024}
        }

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for API call based on token usage"""
        if model in self.pricing:
            input_cost = (prompt_tokens / 1000) * self.pricing[model]["input"]
            output_cost = (completion_tokens / 1000) * self.pricing[model]["output"]
            return input_cost + output_cost
        return 0.0
    
    def get_cost_summary(self) -> dict:
        """Get summary of costs and usage"""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_api_calls": self.api_calls_count,
            "total_tokens": self.token_usage["total_tokens"],
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "cost_breakdown": {
                "input_cost": round((self.token_usage["prompt_tokens"] / 1000) * 0.00005, 6),
                "output_cost": round((self.token_usage["completion_tokens"] / 1000) * 0.00008, 6)
            }
        }
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
            
            # Track API usage and costs for prompt enhancement
            self.api_calls_count += 1
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                # Update token tracking
                self.token_usage["prompt_tokens"] += prompt_tokens
                self.token_usage["completion_tokens"] += completion_tokens
                self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
                
                # Calculate and track cost
                cost = self.calculate_cost("groq/llama3-8b-8192", prompt_tokens, completion_tokens)
                self.total_cost += cost
                
                print(f"💰 Prompt Enhancement Cost: ${cost:.6f}")
            
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
                
                if config.language == 'hi':
                    full_prompt = f"""
                    {base_prompt} {topic}.

                    एक पेशेवर दस्तावेज़ बनाएं जिसमें निम्नलिखित आवश्यकताएं हों:
                    - लगभग {target_words} शब्द लिखें
                    - स्पष्ट शीर्षक और अनुभाग शीर्षक शामिल करें
                    - औपचारिक, शैक्षणिक स्वर का उपयोग करें
                    - विशिष्ट उदाहरण और केस स्टडी प्रदान करें
                    - जहां प्रासंगिक हो वहां आंकड़े या डेटा शामिल करें
                    - परिचय, मुख्य भाग और निष्कर्ष के साथ संरचित करें
                    - हिंदी भाषा में लिखें और देवनागरी लिपि का सही उपयोग करें
                    - इसे क्षेत्र के पेशेवरों के लिए जानकारीपूर्ण और आकर्षक बनाएं

                    उचित पैराग्राफ और प्राकृतिक प्रवाह के साथ दस्तावेज़ को प्रारूपित करें। केवल हिंदी में उत्तर दें।
                    """
                else:
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
                
                # Enhanced system message for Hindi and other Indic languages
                if config.language == 'hi':
                    system_msg = "आप एक विशेषज्ञ पेशेवर लेखक हैं जो उच्च गुणवत्ता वाले, सूचनाप्रद दस्तावेज़ बनाते हैं। हमेशा हिंदी भाषा में उचित संरचना और प्रारूपण के साथ लिखें। देवनागरी लिपि का सही उपयोग करें।"
                elif config.language in ['sa', 'sanskrit']:
                    system_msg = "भवान् एकः विशेषज्ञः व्यावसायिकः लेखकः अस्ति यः उच्चगुणवत्तायुक्तानि सूचनाप्रधानानि प्रलेखानि रचयति। सदैव संस्कृतभाषायां समुचितसंरचनया प्रारूपणेन च लिखतु।"
                  # Encode/decode to ensure UTF-8 compatibility
                try:
                    system_msg_bytes = system_msg.encode('utf-8')
                    full_prompt_bytes = full_prompt.encode('utf-8')
                    system_msg = system_msg_bytes.decode('utf-8')
                    full_prompt = full_prompt_bytes.decode('utf-8')
                except UnicodeError:
                    print("⚠️ Unicode encoding issue in prompt")

                # Add delay before API call
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
                
                # Track API usage and costs
                self.api_calls_count += 1
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    
                    # Update token tracking
                    self.token_usage["prompt_tokens"] += prompt_tokens
                    self.token_usage["completion_tokens"] += completion_tokens
                    self.token_usage["total_tokens"] += total_tokens
                    
                    # Calculate and track cost
                    cost = self.calculate_cost("groq/llama3-8b-8192", prompt_tokens, completion_tokens)
                    self.total_cost += cost
                    
                    print(f"💰 API Cost: ${cost:.6f} | Total: ${self.total_cost:.6f} | Tokens: {total_tokens}")
                
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
                    The insights gained from this analysis provide a solid foundation for further exploration.                    """
                    return fallback_content * (config.num_pages // 2 + 1)  # Scale for page count
                
                # Wait before next attempt
                time.sleep(base_delay * (attempt + 1))

    def create_document_image(self, text: str, page_width: int = 800, page_height: int = 1000) -> tuple[Image.Image, dict]:
        """Create a professional-looking document image with detailed layout information"""
        clean_text, formatting = parse_markdown_text(text)

        img = Image.new('RGB', (page_width, page_height), 'white')
        draw = ImageDraw.Draw(img)

        # Load different fonts
        normal_font = font
        bold_font = load_font(size=12, bold=True)
        h1_font = load_font(size=24, bold=True)
        h2_font = load_font(size=20, bold=True)
        h3_font = load_font(size=16, bold=True)

        margin = 60
        line_height = 24  # Slightly more for non-Latin scripts
        max_width = page_width - 2 * margin
        
        lines = clean_text.split('\n')
        y_position = margin 
        text_coordinates = []

        for line_idx, line in enumerate(lines):
            if y_position > page_height - margin - 50:
                break 
            line_formatting = formatting.get(line_idx, {})
            
            if line_formatting.get('type') == 'heading':
                level = line_formatting['level']
                if level == 1:
                    current_font = h1_font
                    y_position += 20  #xtra space before heading
                    heading_height = 32  #larger height for h1
                elif level == 2:
                    current_font = h2_font
                    y_position += 16  #extra space before heading
                    heading_height = 28  #medium height for h2
                else:
                    current_font = h3_font
                    y_position += 12  #extra space before heading
                    heading_height = 24  #normal height for h3

                # Check if heading fits on page
                if y_position + heading_height > page_height - margin - 50:
                    break
                    
                draw.text((margin, y_position), line, fill='black', font=current_font)
                y_position += heading_height + 10  # Extra spacing after heading
                
            elif line_formatting.get('type') in ['equation', 'inline_equation']:
                try:
                    eq_img = self.render_equation(line_formatting['equation'])
                    img.paste(eq_img, (margin, y_position))
                    y_position += eq_img.height + 10
                except Exception as e:
                    print(f"⚠️ Equation rendering error: {e}")
                    draw.text((margin, y_position), f"[Equation: {line_formatting['equation']}]",
                              fill='blue', font=normal_font)
                    y_position += line_height
                    
            elif 'bold_spans' in line_formatting:
                x_offset = margin 
                current_pos = 0
                
                for span in line_formatting['bold_spans']:
                    # Draw normal text before bold span
                    if current_pos < span['start']:
                        normal_text = line[current_pos:span['start']]
                        if normal_text:
                            draw.text((x_offset, y_position), normal_text, fill='black', font=normal_font)
                            if normal_font:
                                try:
                                    bbox = draw.textbbox((0, 0), normal_text, font=normal_font)
                                    x_offset += bbox[2] - bbox[0]
                                except:
                                    x_offset += len(normal_text) * 8
                            else:
                                x_offset += len(normal_text) * 8
                    
                    # Draw bold text
                    bold_text = span['text']
                    draw.text((x_offset, y_position), bold_text, fill='black', font=bold_font)
                    if bold_font:
                        try:
                            bbox = draw.textbbox((0, 0), bold_text, font=bold_font)
                            x_offset += bbox[2] - bbox[0]
                        except:
                            x_offset += len(bold_text) * 8
                    else:
                        x_offset += len(bold_text) * 8
                    
                    current_pos = span['end']
                if current_pos < len(line):
                    remaining_text = line[current_pos:]
                    draw.text((x_offset, y_position), remaining_text, fill='black', font=normal_font)
                
                y_position += line_height
            else:
                
                if y_position + line_height > page_height - margin - 30:
                    break
                    
                draw.text((margin, y_position), line, fill='black', font=normal_font)
                y_position += line_height
        
        # Create detailed word coordinates for layout information
        text_coordinates = []
        word_bboxes = []
        
        # Process each line for word-level coordinates
        current_y = margin
        for line_idx, line in enumerate(lines):
            if current_y > page_height - margin - line_height:
                break
                
            x_offset = margin
            for word_idx, word in enumerate(line.split()):
                if not word.strip():  # Skip empty words
                    continue
                
                # Calculate word dimensions
                try:
                    if normal_font:
                        word_bbox = draw.textbbox((0, 0), word, font=normal_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        word_height = word_bbox[3] - word_bbox[1]
                    else:
                        word_width = len(word) * 8
                        word_height = line_height
                except:
                    word_width = len(word) * 8
                    word_height = line_height
                
                # Store word information
                word_info = {
                    "word": word,
                    "x": x_offset,
                    "y": current_y,
                    "width": word_width,
                    "height": word_height,
                    "line": line_idx,
                    "word_in_line": word_idx,
                    "bbox": [x_offset, current_y, x_offset + word_width, current_y + word_height]
                }
                text_coordinates.append(word_info)
                word_bboxes.append(word_info["bbox"])
                
                x_offset += word_width + 8  # Add space between words
            
            current_y += line_height

        layout_info = {
            "formatting": formatting,
            "text_coordinates": text_coordinates,
            "word_bboxes": word_bboxes,
            "layout_type": "single_column",
            "page_dimensions": {"width": page_width, "height": page_height},
            "margins": {"top": margin, "bottom": margin, "left": margin, "right": margin},
            "line_height": line_height,
            "total_lines": len(lines),
            "total_words": len(text_coordinates),
            "has_equations": any(f.get('type') in ['equation', 'inline_equation'] for f in formatting.values()),
            "has_headings": any(f.get('type') == 'heading' for f in formatting.values()),
            "has_bold": any('bold_spans' in f for f in formatting.values()),
            "text_regions": [
                {
                    "type": "text_block",
                    "bbox": [margin, margin, page_width - margin, current_y],
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

    def render_equation(self, equation: str, font_size: int = 14):
        """Render mathematical equation as image"""
        try:
            import matplotlib.pyplot as plt 
            from matplotlib import rcParams

            # Configure matplotlib for LaTeX rendering
            rcParams['text.usetex'] = False  # Set to False to avoid LaTeX dependency
            rcParams['font.size'] = font_size

            fig = plt.figure(figsize=(6, 1), dpi=120, facecolor='white')
            fig.text(0.5, 0.5, f"${equation}$", horizontalalignment='center', 
                    verticalalignment='center', fontsize=font_size)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',
                       transparent=True, dpi=120, facecolor='white')

            buf.seek(0)
            eq_img = Image.open(buf)
            plt.close(fig)

            return eq_img
        except Exception as e:
            print(f"⚠️ Equation rendering failed: {e}")
            # Fallback: create simple text image
            from PIL import ImageFont
            img = Image.new('RGB', (200, 30), 'white')
            draw = ImageDraw.Draw(img)
            try:
                draw.text((5, 5), f"Equation: {equation}", fill='black')
            except:
                draw.text((5, 5), f"Equation: [LaTeX]", fill='black')
            return img

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
            
            # Extract comprehensive layout information
            layout_info = page_data.get('layout_info', {})
            
            # Create detailed lines structure
            lines_data = []
            if 'text_coordinates' in layout_info:
                for word_info in layout_info['text_coordinates']:
                    lines_data.append({
                        "text": word_info.get('word', ''),
                        "bbox": word_info.get('bbox', [0, 0, 0, 0]),
                        "line_index": word_info.get('line_index', 0),
                        "confidence": word_info.get('score', 1.0)
                    })              # Create images structure for embedded images
            images_data = []
            visual_elements = layout_info.get('visual_elements', {})
            
            
            # Add tracked AI images
            for img_info in visual_elements.get('images', []):
                images_data.append({
                    "type": img_info.get('type', 'image'),
                    "bbox": [
                        img_info.get('position', {}).get('x', 0),
                        img_info.get('position', {}).get('y', 0),
                        img_info.get('position', {}).get('x', 0) + img_info.get('size', {}).get('width', 400),
                        img_info.get('position', {}).get('y', 0) + img_info.get('size', {}).get('height', 300)
                    ],
                    "description": img_info.get('description', f"Image for page {page_data['page_number']}"),
                    "title": img_info.get('title', 'Untitled Image'),
                    "confidence": 0.95
                })
            
            # so we are adding tracked graphs as images too and nothing gets added to graphs for now(since graphs are visual elements)
            for graph_info in visual_elements.get('graphs', []):
                images_data.append({
                    "type": "graph",
                    "bbox": [
                        graph_info.get('position', {}).get('x', 0),
                        graph_info.get('position', {}).get('y', 0),
                        graph_info.get('position', {}).get('x', 0) + graph_info.get('size', {}).get('width', 500),
                        graph_info.get('position', {}).get('y', 0) + graph_info.get('size', {}).get('height', 350)
                    ],
                    "description": graph_info.get('description', f"Graph for page {page_data['page_number']}"),
                    "title": graph_info.get('title', 'Chart'),
                    "confidence": 0.95
                })
            
            print(f"✅ Images data populated: {len(images_data)} items")
              #create equations structure from detected equations
            equations_data = []
            for eq_info in visual_elements.get('equations', []):
                equations_data.append({
                    "type": eq_info.get('type', 'mathematical'),
                    "content": eq_info.get('content', ''),
                    "description": eq_info.get('description', 'Mathematical equation'),
                    "symbol": eq_info.get('symbol', ''),
                    "confidence": 0.95
                })
            
            #adding some sample equations if there are not detected but document has mathematical content
            if not equations_data and ('mathematical' in page_data.get('text', '').lower() or 
                                      'formula' in page_data.get('text', '').lower() or 
                                      'equation' in page_data.get('text', '').lower()):
                equations_data.append({
                    "type": "mathematical",
                    "content": "Sample mathematical equation",
                    "description": "Mathematical equation detected in text",
                    "symbol": "∑",
                    "confidence": 0.85
                })
            
            print(f"✅ Equations data populated: {len(equations_data)} items")              # Create tables structure from tracked tables
            tables_data = []
            for table_info in visual_elements.get('tables', []):
                tables_data.append({
                    "type": table_info.get('type', 'data_table'),
                    "bbox": [
                        table_info.get('position', {}).get('x', 60),
                        table_info.get('position', {}).get('y', 300),
                        table_info.get('position', {}).get('x', 60) + table_info.get('size', {}).get('width', 500),
                        table_info.get('position', {}).get('y', 300) + table_info.get('size', {}).get('height', 150)
                    ],
                    "rows": table_info.get('rows', 0),
                    "columns": table_info.get('columns', 0),
                    "title": table_info.get('title', 'Data Table'),
                    "description": table_info.get('description', 'Table with data'),
                    "confidence": 0.95
                })
            
            print(f"✅ Tables data populated: {len(tables_data)} items")
            
            #connvert to string format as expected by the dataset
            if not images_data and not tables_data and not equations_data:
                print("⚠️ No visual elements detected, adding sample elements")
                # Add at least one element of each type for demonstration
                images_data.append({
                    "type": "sample",
                    "bbox": [100, 100, 400, 300],
                    "description": "Sample visual element",
                    "title": "Generated Content",
                    "confidence": 0.80
                })
                tables_data.append({
                    "type": "data_table",
                    "bbox": [60, 400, 560, 550],
                    "rows": 3,
                    "columns": 2,
                    "title": "Sample Table",
                    "description": "Sample data table",
                    "confidence": 0.80
                })
                equations_data.append({
                    "type": "mathematical",
                    "content": "y = mx + b",
                    "description": "Linear equation",
                    "symbol": "=",
                    "confidence": 0.80
                })
            
            tables_data_str = str(tables_data) if tables_data else "[]"
            
            #create comprehensive layout detection data
            base_layout_detection = {
                "page_layout": {
                    "type": layout_info.get('layout_type', 'single_column'),
                    "columns": layout_info.get('columns', 1),
                    "margin": layout_info.get('margins', {"top": 60, "bottom": 60, "left": 60, "right": 60}),
                    "header_height": layout_info.get('header_height', 80),
                    "footer_height": layout_info.get('footer_height', 40)
                },
                "text_regions": layout_info.get('text_regions', []),                "visual_elements": {
                    "graphs": layout_info.get('visual_elements', {}).get('graphs', []),
                    "tables": layout_info.get('visual_elements', {}).get('tables', []),
                    "images": layout_info.get('visual_elements', {}).get('images', []),
                    "equations": layout_info.get('visual_elements', {}).get('equations', [])
                },
                "reading_order": list(range(len(layout_info.get('text_coordinates', [])))),
                "confidence_scores": {
                    "overall": 0.95,
                    "text_detection": 0.98,
                    "layout_analysis": 0.92
                }
            }
            
            # Create detailed content list
            content_list_data = []
            if 'text_coordinates' in layout_info:
                for i, word_info in enumerate(layout_info['text_coordinates']):
                    content_list_data.append({
                        "id": i + 1,
                        "type": "text",
                        "content": word_info.get('word', ''),
                        "bbox": word_info.get('bbox', [0, 0, 0, 0]),
                        "confidence": 0.95,
                        "reading_order": i + 1
                    })
            
            # Add visual elements to content list
            if layout_info.get('has_graph', False):
                content_list_data.append({
                    "id": len(content_list_data) + 1,
                    "type": "graph",
                    "content": "Data visualization chart",
                    "bbox": layout_info.get('graph_bbox', [200, 150, 600, 400]),
                    "confidence": 0.90,
                    "reading_order": len(content_list_data) + 1
                })
            
            if layout_info.get('has_table', False):
                content_list_data.append({
                    "id": len(content_list_data) + 1,
                    "type": "table",
                    "content": "Data table with structured information",
                    "bbox": layout_info.get('table_bbox', [60, 300, 500, 450]),
                    "confidence": 0.92,
                    "reading_order": len(content_list_data) + 1
                })
            
            # Create PDF info structure
            pdf_info_data = {
                "filename": pdf_name,
                "page_count": config.num_pages,
                "page_size": {"width": layout_info.get('page_dimensions', {}).get('width', 800), 
                              "height": layout_info.get('page_dimensions', {}).get('height', 1000)},
                "creation_date": "2025-06-21",
                "language": config.language,
                "text_extraction_method": "synthetic_generation",
                "layout_analysis_method": "rule_based",
                "quality_score": 0.95,
                "processing_metadata": {
                    "ocr_engine": "synthetic",
                    "layout_detector": "synthdoc_v1",
                    "confidence_threshold": 0.8
                }
            }
            
            doc = {
                'image': page_data['image'],  # PIL Image for HuggingFace
                'imagewidth': layout_info.get('page_dimensions', {}).get('width', 800),
                'pdf_name': pdf_name,
                'page_number': page_data['page_number'] - 1,  # 0-indexed as in example
                'markdown': markdown_content,
                'html': html_content,
                'layout': str(layout_info),  # Convert to string as in example
                'lines': str(lines_data),  # Convert to string as in example  
                'images': str(images_data),  # Convert to string as in example
                'equations': str(equations_data),  # Convert to string as in example
                'tables': tables_data_str,  # String format as in example
                'page_size': f"{layout_info.get('page_dimensions', {}).get('width', 800)}x{layout_info.get('page_dimensions', {}).get('height', 1000)}",
                'content_list': str(content_list_data),  # Convert to string as in example
                'base_layout_detection': str(base_layout_detection),  # Convert to string as in example
                'pdf_info': str(pdf_info_data),  # Convert to string as in example
                'system_prompt': "Generate high-quality synthetic documents for training ML models",  # Fixed as in example
                'response': page_data['text']  # The generated text content
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
        
        # Display cost summary
        cost_summary = self.get_cost_summary()
        print(f"\n💰 COST SUMMARY:")
        print(f"   • Total Cost: ${cost_summary['total_cost_usd']:.6f}")
        print(f"   • API Calls: {cost_summary['total_api_calls']}")
        print(f"   • Total Tokens: {cost_summary['total_tokens']:,}")
        print(f"   • Prompt Tokens: {cost_summary['prompt_tokens']:,}")
        print(f"   • Completion Tokens: {cost_summary['completion_tokens']:,}")
        print(f"   • Input Cost: ${cost_summary['cost_breakdown']['input_cost']:.6f}")
        print(f"   • Output Cost: ${cost_summary['cost_breakdown']['output_cost']:.6f}")
        
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