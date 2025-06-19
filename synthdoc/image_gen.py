from diffusers import StableDiffusionPipeline
import torch
from typing import List, Optional, Dict, Any 
from PIL import Image, ImageDraw
from collections import Counter
from utils  import create_text_section
from config import DocumentConfig, LayoutType
from layouts import LayoutManager
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


def create_multicolumn_document_image(text: str, config: DocumentConfig, page_num: int = 0, 
                                    used_graph_types: set = None, used_table_types: set = None,
                                    page_width: int = 800, page_height: int = 1000) -> tuple[Image.Image, dict]:
    """Create document image with advanced multi-column layouts"""
    
    # Initialize word coordinates tracking
    if not hasattr(create_multicolumn_document_image, '_word_coords'):
        create_multicolumn_document_image._word_coords = []
    else:
        create_multicolumn_document_image._word_coords = []  # Reset for this page
    
    if used_graph_types is None:
        used_graph_types = set()
    if used_table_types is None:
        used_table_types = set()
    
    # Initialize layout manager
    layout_manager = LayoutManager()
    
    # Calculate layout dimensions
    layout_info = layout_manager.calculate_layout_dimensions(
        config.layout_type, page_width, page_height
    )
    
    # Create image
    img = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    from font import load_font, load_title_font
    font = load_font()
    title_font = load_title_font()
    
    # Add page elements (headers, footers, etc.)
    layout_manager.add_page_elements(draw, layout_info, page_num, title_font, font, 
                                   title=config.prompt)
    
    # Calculate content area
    content_start_y = layout_info['margin'] + layout_info['header_height']
    content_height = (page_height - layout_info['margin'] - layout_info['footer_height'] 
                     - content_start_y)
    
    # Decide what visual elements to include
    include_graph = config.include_graphs and (page_num == 1 or random.random() < 0.4)
    include_table = config.include_tables and (page_num == 2 or random.random() < 0.4)
    include_image = page_num > 0 and random.random() < 0.3
    
    current_y = content_start_y
    
    # Add visual elements first (they take fixed space)
    visual_elements_height = 0
    
    # Add AI-generated image
    if include_image and page_num > 0:
        try:
            ai_image = generate_contextual_image(text, config.language)
            if ai_image:
                # For multi-column, make image span across columns or fit in one
                if config.layout_type in [LayoutType.TWO_COLUMN, LayoutType.THREE_COLUMN]:
                    # Resize to fit single column width
                    target_width = layout_info['column_width']
                    aspect_ratio = ai_image.height / ai_image.width
                    target_height = int(target_width * aspect_ratio)
                    ai_image = ai_image.resize((target_width, target_height))
                
                # Center the image in first column or span across page
                if config.layout_type == LayoutType.SINGLE_COLUMN:
                    img_x = (page_width - ai_image.width) // 2
                else:
                    img_x = layout_info['column_positions'][0]
                
                img.paste(ai_image, (img_x, current_y))
                current_y += ai_image.height + 20
                visual_elements_height += ai_image.height + 20
                
                # Add caption
                caption = f"Figure {page_num}-A: Generated Illustration"
                draw.text((img_x, current_y), caption, fill='black', font=font)
                current_y += 30
                visual_elements_height += 30
        except Exception as e:
            print(f"⚠️ Error adding AI image: {e}")
    
    # Add graph
    if include_graph and page_num > 0:
        available_graph_types = ["line", "bar", "pie", "scatter", "area", "heatmap"] 
        available_types = [t for t in available_graph_types if t not in used_graph_types]
        
        if not available_types:
            available_types = available_graph_types
            used_graph_types.clear()
        
        graph_type = random.choice(available_types)
        used_graph_types.add(graph_type)
        
        try:
            from tables import generate_sample_data_for_graph, create_advanced_graph
            graph_data, _ = generate_sample_data_for_graph(graph_type)
            graph_img = create_advanced_graph(graph_data, graph_type)
            
            # Resize graph for layout
            if config.layout_type in [LayoutType.TWO_COLUMN, LayoutType.THREE_COLUMN]:
                target_width = layout_info['column_width']
                aspect_ratio = graph_img.height / graph_img.width
                target_height = int(target_width * aspect_ratio)
                graph_img = graph_img.resize((target_width, target_height))
            
            # Position graph
            if config.layout_type == LayoutType.SINGLE_COLUMN:
                graph_x = (page_width - graph_img.width) // 2
            else:
                # Place in second column if two-column, or span across if space allows
                graph_x = layout_info['column_positions'][-1] if len(layout_info['column_positions']) > 1 else layout_info['column_positions'][0]
            
            img.paste(graph_img, (graph_x, current_y))
            current_y += graph_img.height + 20
            visual_elements_height += graph_img.height + 20
            
            # Add caption
            caption = f"Figure {page_num}: {graph_data['title']}"
            draw.text((graph_x, current_y), caption, fill='black', font=font)
            current_y += 40
            visual_elements_height += 40
            
        except Exception as e:
            print(f"⚠️ Error adding graph: {e}")
    
    # Calculate remaining space for text
    remaining_height = content_height - visual_elements_height
    
    # Now handle text in columns
    if remaining_height > 100:
        # Wrap text into columns
        columns_data = layout_manager.wrap_text_to_columns(text, layout_info, font, draw)
          # Render text in each column
        for col_idx, col_data in enumerate(columns_data):
            text_y = current_y
            
            # Special styling for sidebar
            if col_data.get('type') == 'sidebar':
                # Draw sidebar background
                sidebar_bg = Image.new('RGBA', (col_data['width'] + 20, remaining_height), (240, 240, 240, 100))
                img.paste(sidebar_bg, (col_data['x_position'] - 10, text_y), sidebar_bg)
              # Render text lines with word-level coordinate tracking
            for line_idx, line in enumerate(col_data['text_lines']):
                if text_y > page_height - layout_info['margin'] - layout_info['footer_height'] - 30:
                    break  # Stop if we reach bottom margin
                    
                try:
                    current_font = font
                    # Make sidebar text smaller
                    if col_data.get('type') == 'sidebar' and font:
                        # Would need smaller font here, for now just use regular
                        pass
                      # Draw the line
                    draw.text((col_data['x_position'], text_y), line, fill='black', font=current_font)
                    
                    # Track word-level coordinates for this line
                    x_offset = col_data['x_position']
                    words = line.split()
                    
                    for word_idx, word in enumerate(words):
                        # Calculate word dimensions
                        if current_font:
                            try:
                                word_bbox = draw.textbbox((0, 0), word, font=current_font)
                                word_width = word_bbox[2] - word_bbox[0]
                                word_height = word_bbox[3] - word_bbox[1]
                            except:
                                word_width = len(word) * 8
                                word_height = 24
                        else:
                            word_width = len(word) * 8
                            word_height = 24
                        
                        # Store word coordinates for later use
                        if not hasattr(create_multicolumn_document_image, '_word_coords'):
                            create_multicolumn_document_image._word_coords = []
                            
                        create_multicolumn_document_image._word_coords.append({
                            "type": "text",
                            "content": word,
                            "coordinates": [x_offset, text_y, x_offset + word_width, text_y + word_height],
                            "score": 1.0,
                            "index": len(create_multicolumn_document_image._word_coords) + 1,
                            "column_index": col_idx,
                            "line_index": line_idx,
                            "word_in_line": word_idx,
                            "page_number": page_num + 1
                        })
                        
                        x_offset += word_width + 8  # Add space between words
                        
                    text_y += 24  # line height
                    
                except Exception as e:
                    print(f"⚠️ Text rendering error: {e}")
                    continue
                    
    # Prepare detailed layout info for return with enhanced structure
    detailed_layout_info = {
        "page_number": page_num + 1,  # 1-indexed
        "layout_type": config.layout_type.value,
        "has_graph": include_graph,
        "has_table": include_table,
        "has_ai_image": include_image,
        "page_dimensions": {"width": page_width, "height": page_height},
        "margins": {
            "top": layout_info['margin'],
            "bottom": layout_info['margin'],
            "left": layout_info['margin'],
            "right": layout_info['margin']
        },
        "header_height": layout_info['header_height'],
        "footer_height": layout_info['footer_height'],
        "content_area": {
            "x": layout_info['margin'],
            "y": content_start_y,
            "width": page_width - 2 * layout_info['margin'],
            "height": content_height
        },
        "columns": [],
        "text_coordinates": [],
        "visual_elements": {
            "graphs": [],
            "tables": [],
            "images": []
        },
        "total_words": 0,
        "visual_elements_height": visual_elements_height
    }
    
    # Add column information
    if config.layout_type == LayoutType.SINGLE_COLUMN:
        detailed_layout_info["columns"] = [{
            "x": layout_info['margin'],
            "y": content_start_y,
            "width": page_width - 2 * layout_info['margin'],
            "height": content_height,
            "column_index": 0
        }]
    elif config.layout_type == LayoutType.TWO_COLUMN:
        column_width = (page_width - 3 * layout_info['margin']) // 2
        column_gap = layout_info['margin']
        detailed_layout_info["columns"] = [
            {
                "x": layout_info['margin'],
                "y": content_start_y,
                "width": column_width,
                "height": content_height,
                "column_index": 0
            },
            {
                "x": layout_info['margin'] + column_width + column_gap,
                "y": content_start_y,
                "width": column_width,
                "height": content_height,
                "column_index": 1
            }
        ]
    elif config.layout_type == LayoutType.THREE_COLUMN:
        column_width = (page_width - 4 * layout_info['margin']) // 3
        column_gap = layout_info['margin']
        detailed_layout_info["columns"] = [
            {
                "x": layout_info['margin'],
                "y": content_start_y,
                "width": column_width,
                "height": content_height,
                "column_index": 0
            },
            {
                "x": layout_info['margin'] + column_width + column_gap,
                "y": content_start_y,
                "width": column_width,
                "height": content_height,
                "column_index": 1
            },
            {
                "x": layout_info['margin'] + 2 * (column_width + column_gap),
                "y": content_start_y,
                "width": column_width,
                "height": content_height,
                "column_index": 2
            }
        ]
      # Add captured word coordinates from the text rendering
    if hasattr(create_multicolumn_document_image, '_word_coords'):
        detailed_layout_info["text_coordinates"] = create_multicolumn_document_image._word_coords
        detailed_layout_info["total_words"] = len(create_multicolumn_document_image._word_coords)
        # Clear for next use
        create_multicolumn_document_image._word_coords = []
    else:
        detailed_layout_info["text_coordinates"] = []
        detailed_layout_info["total_words"] = 0
    
    return img, detailed_layout_info

