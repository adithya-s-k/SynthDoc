from diffusers import StableDiffusionPipeline
import torch
from typing import List, Optional, Dict, Any 
from PIL import Image, ImageDraw
from collections import Counter
from utils  import create_text_section
from config import DocumentConfig
import random
import re

sd_pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
if torch.cuda.is_available():
    sd_pipe = sd_pipe.to("cuda")
print("✅ Stable Diffusion loaded successfully")


def extract_keywords(text: str) -> List[str]:
    """Extract key terms from text content"""
    # Simple keyword extraction
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    word_freq = Counter(words)
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'more', 'than', 'about', 'after', 'first', 'never', 'these', 'could', 'where', 'much', 'through', 'before', 'right', 'should', 'those', 'while', 'another', 'being', 'during', 'years', 'around', 'within', 'under', 'between', 'without', 'across', 'since', 'still', 'example'}
    keywords = [word for word, freq in word_freq.most_common(10) if word not in stop_words]
    return keywords[:5]

def categorize_content(keywords: List[str]) -> str:
    """Categorize content based on keywords"""
    tech_words = {'technology', 'artificial', 'intelligence', 'computing', 'digital', 'software', 'system', 'data', 'algorithm', 'machine', 'learning'}
    business_words = {'business', 'market', 'financial', 'economic', 'company', 'organization', 'strategy', 'management', 'industry', 'commercial'}
    research_words = {'research', 'study', 'analysis', 'scientific', 'method', 'theory', 'experiment', 'results', 'findings', 'investigation'}
    
    keyword_set = set(keywords)
    if keyword_set & tech_words:
        return 'tech'
    elif keyword_set & business_words:
        return 'business'
    elif keyword_set & research_words:
        return 'research'
    else:
        return 'general'



def generate_contextual_image(text_content: str, language: str = "en") -> Image.Image:
    """Generate contextual image based on document content"""
    if not sd_pipe:
        print("⚠️ Stable Diffusion not available, skipping image generation")
        return None
            
    try:
        # Extract keywords from text for prompt
        keywords = extract_keywords(text_content)
        
        prompt_templates = {
            'tech': "professional diagram of {}, clean minimalist style, technical illustration, corporate presentation",
            'business': "corporate infographic about {}, modern flat design, blue and gray colors, professional layout",
            'research': "scientific visualization of {}, academic paper style, charts and data visualization, clean background",
            'general': "professional illustration of {}, corporate presentation style, modern design, clean background"
        }
        
        category = categorize_content(keywords)
        prompt = prompt_templates.get(category, prompt_templates['general']).format(', '.join(keywords[:3]))
        
        # Add negative prompt for better quality
        negative_prompt = "text, watermark, signature, blurry, low quality, distorted, ugly, bad anatomy"
        
        print(f"🎨 Generating image with prompt: {prompt[:50]}...")
        
        image = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=384
        ).images[0]
        
        # Resize to fit document layout
        return image.resize((400, 300))
        
    except Exception as e:
        print(f"⚠️ Error generating image: {e}")
        return None
    

def create_document_image_with_content(text: str, config: DocumentConfig, page_num: int = 0, 
                                        used_graph_types: set = None, used_table_types: set = None,
                                        page_width: int = 800, page_height: int = 1000) -> tuple[Image.Image, dict]:
    """Create document image with mixed content types"""
        
    if used_graph_types is None:
        used_graph_types = set()
    if used_table_types is None:
        used_table_types = set()
        # Decide content type for this page
    include_graph = config.include_graphs and (page_num == 1 or random.random() < 0.4)
    include_table = config.include_tables and (page_num == 2 or random.random() < 0.4)
    include_image = page_num > 0 and random.random() < 0.3  # 30% chance for AI-generated images
    
    # Initialize variables
    graph_type = None
    table_type = None
    
    img = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load font
    from font import load_font, load_title_font
    font = load_font()
    title_font = load_title_font()
    
    margin = 60
    current_y = margin
    
    # Add AI-generated contextual image
    if include_image and page_num > 0:
        try:

            ai_image = generate_contextual_image(text, config.language)
            if ai_image:
                # Paste AI-generated image
                img_x = (page_width - ai_image.width) // 2
                img.paste(ai_image, (img_x, current_y))
                current_y += ai_image.height + 20
                
                # Add image caption
                caption = f"Figure {page_num}-A: Generated Illustration"
                draw.text((margin, current_y), caption, fill='black', font=font)
                current_y += 30
        except Exception as e:
            print(f"⚠️ Error adding AI image: {e}")
    
    # Add graph with unique type
    if include_graph and page_num > 0:
        available_graph_types = ["line", "bar", "pie", "scatter", "area", "heatmap"] 
        available_types = [t for t in available_graph_types if t not in used_graph_types]
        
        if not available_types:
            available_types = available_graph_types  # Reset if all used
            used_graph_types.clear()
        
        graph_type = random.choice(available_types)
        used_graph_types.add(graph_type)
        
        # Generate and add graph
        from tables import generate_sample_data_for_graph, create_advanced_graph, generate_diverse_table_data, create_table_image
        graph_data, _ = generate_sample_data_for_graph(graph_type)
        graph_img = create_advanced_graph(graph_data, graph_type)
        
        # Paste graph
        graph_x = (page_width - graph_img.width) // 2
        img.paste(graph_img, (graph_x, current_y))
        current_y += graph_img.height + 20
        
        # Add graph caption
        caption = f"Figure {page_num}: {graph_data['title']}"
        draw.text((margin, current_y), caption, fill='black', font=font)
        current_y += 40
    
    # Add table with unique type
    if include_table and page_num > 0:
        table_types = ["comparison", "timeline", "statistics", "pricing", "features"]
        available_table_types = [t for t in table_types if t not in used_table_types]
        
        if not available_table_types:
            available_table_types = table_types  # Reset if all used
            used_table_types.clear()
        
        table_type = random.choice(available_table_types)
        used_table_types.add(table_type)
        
        # Generate diverse table
        table_data = generate_diverse_table_data(table_type, config.language)
        table_img = create_table_image(table_data, 500, 150)
        
        # Paste table
        table_x = (page_width - table_img.width) // 2
        img.paste(table_img, (table_x, current_y))
        current_y += table_img.height + 20
        
        # Add table caption
        caption = f"Table {page_num}: {table_type.title()} Overview"
        draw.text((margin, current_y), caption, fill='black', font=font)
        current_y += 40
    

    # Add text content (remaining space)
    remaining_height = page_height - current_y - margin
    if remaining_height > 100:  # Only add text if there's enough space
        text_img, text_layout = create_text_section(
            text, page_width, remaining_height, current_y, font, title_font
        )
        
        # Paste text section
        img.paste(text_img, (0, current_y))
    
    layout_info = {
        "page_number": page_num,
        "has_graph": include_graph,
        "layout_type": "single_column",
        "total_lines": "",
        "has_table": include_table,
        "has_ai_image": include_image,
        "content_start_y": current_y,
        "page_dimensions": {"width": page_width, "height": page_height},
        "graph_type": graph_type if include_graph else None,
        "table_type": table_type if include_table else None
    }
    
    return img, layout_info

