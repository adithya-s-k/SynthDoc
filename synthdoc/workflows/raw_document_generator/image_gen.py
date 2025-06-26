import io
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import re
import os

# Helper function to load language-appropriate fonts
def load_language_font(language_code: str, size: int = 12):
    """Load appropriate font for the given language."""
    try:
        # Get the fonts directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_base_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'fonts')
        
        # Language-specific font mapping to local files
        language_font_files = {
            'hi': ['NotoSansDevanagari-Regular.ttf', 'AnnapurnaSIL-Regular.ttf', 'Kalam-Regular.ttf'],
            'sa': ['NotoSansDevanagari-Regular.ttf', 'AnnapurnaSIL-Regular.ttf'],
            'mr': ['NotoSansDevanagari-Regular.ttf'],
            'bn': ['NotoSansBengali-Regular.ttf', 'NotoSerifBengali-Regular.ttf'],
            'gu': ['NotoSansGujarati-Regular.ttf', 'NotoSerifGujarati-Regular.ttf'],
            'kn': ['NotoSansKannada-Regular.ttf', 'NotoSerifKannada-Regular.ttf'],
            'ml': ['NotoSansMalayalam-Regular.ttf', 'NotoSerifMalayalam-Regular.ttf'],
            'or': ['NotoSansOriya-Regular.ttf'],
            'pa': ['NotoSansGurmukhi-Regular.ttf'],
            'ta': ['NotoSansTamil-Regular.ttf'],
            'te': ['NotoSansTelugu-Regular.ttf']
        }
        
        # Try local font files first
        if language_code in language_font_files:
            lang_dir = os.path.join(fonts_base_dir, language_code)
            if os.path.exists(lang_dir):
                for font_file in language_font_files[language_code]:
                    font_path = os.path.join(lang_dir, font_file)
                    if os.path.exists(font_path):
                        try:
                            return ImageFont.truetype(font_path, size)
                        except:
                            continue
        
        # Fallback to system fonts
        fallback_fonts = ["arial.ttf", "Arial", "DejaVu Sans"]
        for font_name in fallback_fonts:
            try:
                return ImageFont.truetype(font_name, size)
            except:
                continue
        
        # Final fallback
        return ImageFont.load_default()
        
    except Exception as e:
        print(f"⚠️ Font loading error for {language_code}: {e}")
        try:
            return ImageFont.load_default()
        except:
            return None

# Advanced AI image generation setup
try:
    from diffusers import StableDiffusionPipeline
    import torch
    
    # Initialize SD pipeline
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        sd_pipe = sd_pipe.to("cuda")
    SD_AVAILABLE = True
    print("✅ Stable Diffusion loaded successfully")
except Exception as e:
    SD_AVAILABLE = False
    sd_pipe = None
    print(f"⚠️ Stable Diffusion not available: {e}")

# Content analysis functions
def extract_keywords(text: str) -> List[str]:
    """Extract key terms from text content"""
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

# Helper functions for advanced rendering
def wrap_text_to_width(text: str, max_width: int, font, draw) -> List[str]:
    """Wrap text to fit within specified width"""
    if not text:
        return []
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Test if adding this word would exceed width
        test_line = ' '.join(current_line + [word])
        
        if font:
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]
            except:
                line_width = len(test_line) * 7  # Fallback
        else:
            line_width = len(test_line) * 7  # Fallback
        
        if line_width <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def create_advanced_chart(chart_type: str = "bar") -> Image.Image:
    """Create advanced, publication-quality charts"""
    # Set professional styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    
    # Professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5A189A', '#2D6A4F']
    
    if chart_type == "bar":
        categories = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024"]
        values = [random.randint(75, 125) for _ in range(5)]
        bars = ax.bar(categories, values, color=colors[:5], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title("Quarterly Performance Analysis", fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Performance Index", fontsize=11)
        ax.set_xlabel("Quarter", fontsize=11)
        
    elif chart_type == "line":
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        revenue = [random.randint(80, 120) + i*5 for i in range(6)]
        costs = [random.randint(60, 90) + i*3 for i in range(6)]
        
        ax.plot(months, revenue, marker='o', linewidth=3, label='Revenue', color=colors[0])
        ax.plot(months, costs, marker='s', linewidth=3, label='Costs', color=colors[1])
        
        ax.fill_between(months, revenue, alpha=0.3, color=colors[0])
        ax.fill_between(months, costs, alpha=0.3, color=colors[1])
        
        ax.set_title("Revenue vs Costs Trend", fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Amount ($K)", fontsize=11)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
    elif chart_type == "pie":
        labels = ["Technology", "Healthcare", "Finance", "Education", "Other"]
        sizes = [30, 25, 20, 15, 10]
        explode = (0.05, 0, 0, 0, 0)  # Slightly separate first slice
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         startangle=90, colors=colors[:5], 
                                         explode=explode, shadow=True)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title("Market Segment Distribution", fontsize=14, fontweight='bold', pad=20)
    
    # Common styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if chart_type != "pie":
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def create_advanced_table() -> Image.Image:
    """Create publication-quality data tables"""
    # Table data
    headers = ["Product", "Revenue ($M)", "Growth (%)", "Market Share", "Status"]
    data = [
        ["Alpha Platform", "125.4", "+12.3", "23.5%", "Growing"],
        ["Beta Services", "89.7", "+8.1", "18.2%", "Stable"], 
        ["Gamma Solutions", "156.2", "+15.7", "28.9%", "Expanding"],
        ["Delta Analytics", "73.8", "+5.4", "14.1%", "Stable"],
        ["Epsilon Tools", "91.3", "+9.8", "15.3%", "Growing"]
    ]
    
    # Calculate dimensions
    cell_width = 120
    cell_height = 35
    header_height = 45
    table_width = len(headers) * cell_width
    table_height = header_height + len(data) * cell_height
    
    # Create image with some padding
    padding = 20
    img = Image.new('RGB', (table_width + 2*padding, table_height + 2*padding), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load fonts - with fallback
    try:
        header_font = ImageFont.truetype("arial.ttf", 12)
        cell_font = ImageFont.truetype("arial.ttf", 10)
    except:
        try:
            header_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
        except:
            header_font = None
            cell_font = None
    
    # Color scheme
    header_color = '#2E86AB'
    alt_row_color = '#F8F9FA'
    border_color = '#DEE2E6'
    text_color = '#212529'
    
    # Draw table
    start_x, start_y = padding, padding
    
    # Draw headers
    for i, header in enumerate(headers):
        x = start_x + i * cell_width
        y = start_y
        
        # Header background
        draw.rectangle([x, y, x + cell_width, y + header_height], 
                      fill=header_color, outline=border_color, width=1)
        
        # Header text (centered)
        text_bbox = draw.textbbox((0, 0), header, font=header_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (cell_width - text_width) // 2
        text_y = y + (header_height - text_height) // 2
        
        draw.text((text_x, text_y), header, fill='white', font=header_font)
    
    # Draw data rows
    for row_idx, row in enumerate(data):
        y = start_y + header_height + row_idx * cell_height
        
        # Alternate row colors
        row_color = alt_row_color if row_idx % 2 == 1 else 'white'
        
        for col_idx, cell_data in enumerate(row):
            x = start_x + col_idx * cell_width
            
            # Cell background
            draw.rectangle([x, y, x + cell_width, y + cell_height], 
                          fill=row_color, outline=border_color, width=1)
            
            # Cell text
            text_bbox = draw.textbbox((0, 0), str(cell_data), font=cell_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Right align numeric columns (1, 2, 3), left align others
            if col_idx in [1, 2, 3]:
                text_x = x + cell_width - text_width - 8
            else:
                text_x = x + 8
            
            text_y = y + (cell_height - text_height) // 2
            
            # Color coding for status
            if col_idx == 4:  # Status column
                if cell_data == "Growing":
                    text_color_cell = '#28A745'
                elif cell_data == "Expanding":
                    text_color_cell = '#007BFF'
                else:
                    text_color_cell = '#6C757D'
            else:
                text_color_cell = text_color
            
            draw.text((text_x, text_y), str(cell_data), fill=text_color_cell, font=cell_font)
    
    return img

def generate_contextual_image(text_content: str, language: str = "en") -> Image.Image:
    """Generate high-quality contextual images using AI or create professional placeholders"""
    if SD_AVAILABLE and sd_pipe:
        return _generate_ai_image(text_content)
    else:
        return _create_professional_placeholder(text_content)

def _generate_ai_image(text_content: str) -> Image.Image:
    """Generate real AI image using Stable Diffusion"""
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
        
        print(f"🎨 Generating AI image with prompt: {prompt[:50]}...")
        
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
        print(f"⚠️ Error generating AI image: {e}")
        return _create_professional_placeholder(text_content)

def _create_professional_placeholder(text_content: str) -> Image.Image:
    """Create professional-quality placeholder image based on content"""
    width, height = 400, 300
    img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    # Load fonts - with better fallback
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        except:
            title_font = None
            text_font = None
    
    # Extract keywords
    keywords = extract_keywords(text_content)
    category = categorize_content(keywords)
    
    # Color scheme based on category
    color_schemes = {
        'tech': {'bg': '#E3F2FD', 'primary': '#1976D2', 'secondary': '#42A5F5'},
        'business': {'bg': '#F3E5F5', 'primary': '#7B1FA2', 'secondary': '#BA68C8'},
        'research': {'bg': '#E8F5E8', 'primary': '#388E3C', 'secondary': '#66BB6A'},
        'general': {'bg': '#FFF3E0', 'primary': '#F57C00', 'secondary': '#FFB74D'}
    }
    
    colors = color_schemes.get(category, color_schemes['general'])
    
    # Background gradient effect
    draw.rectangle([0, 0, width, height], fill=colors['bg'])
    
    # Border
    draw.rectangle([10, 10, width-10, height-10], outline=colors['primary'], width=3)
    
    # Title
    title = f"{category.title()} Visualization"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 25), title, fill=colors['primary'], font=title_font)
    
    # Keywords
    if keywords:
        keywords_text = "Keywords: " + ", ".join(keywords[:3])
        kw_bbox = draw.textbbox((0, 0), keywords_text, font=text_font)
        kw_width = kw_bbox[2] - kw_bbox[0]
        kw_x = (width - kw_width) // 2
        draw.text((kw_x, 55), keywords_text, fill=colors['secondary'], font=text_font)
    
    # Geometric elements based on category
    if category == 'tech':
        # Circuit-like pattern
        for i in range(3):
            y = 100 + i * 40
            draw.line([(30, y), (width-30, y)], fill=colors['primary'], width=2)
            draw.ellipse([50 + i*100, y-5, 60 + i*100, y+5], fill=colors['secondary'])
    
    elif category == 'business':
        # Bar chart representation
        for i in range(4):
            x = 60 + i * 80
            bar_height = random.randint(40, 120)
            draw.rectangle([x, 200-bar_height, x+40, 200], fill=colors['primary'])
            draw.rectangle([x+5, 205-bar_height, x+35, 195], fill=colors['secondary'])
    
    elif category == 'research':
        # Data points and trend line
        points = [(50 + i*60, 150 + random.randint(-30, 30)) for i in range(5)]
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=colors['primary'], width=3)
        for point in points:
            draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill=colors['secondary'])
    
    else:
        # Abstract shapes
        for i in range(3):
            x = 80 + i * 80
            y = 120 + random.randint(-20, 20)
            draw.ellipse([x, y, x+60, y+40], fill=colors['primary'])
            draw.ellipse([x+10, y+10, x+50, y+30], fill=colors['secondary'])
    
    # Footer text
    draw.text((20, height-30), "Professional Document Illustration", fill=colors['primary'], font=text_font)
    
    return img

def create_multicolumn_document_image(text: str, config, page_num: int = 0, 
                                    used_graph_types: set = None, used_table_types: set = None,
                                    page_width: int = 800, page_height: int = 1000) -> Tuple[Image.Image, dict]:
    """Create document image with proper multi-column layouts"""
    
    if used_graph_types is None:
        used_graph_types = set()
    if used_table_types is None:
        used_table_types = set()
    
    # Create image
    img = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load language-appropriate fonts
    try:
        import os
        from ...languages import get_language_fonts
        
        # Get language code from config
        language_code = config.language.value if hasattr(config.language, 'value') else str(config.language)
        language_fonts = get_language_fonts(language_code)
        
        # Try to load appropriate fonts for the language
        font = None
        title_font = None
        header_font = None
        
        # Get the fonts directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_base_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'fonts')
        
        # Language-specific font mapping to local files
        language_font_files = {
            'hi': ['NotoSansDevanagari-Regular.ttf', 'AnnapurnaSIL-Regular.ttf', 'Kalam-Regular.ttf'],
            'sa': ['NotoSansDevanagari-Regular.ttf', 'AnnapurnaSIL-Regular.ttf'],
            'mr': ['NotoSansDevanagari-Regular.ttf'],
            'bn': ['NotoSansBengali-Regular.ttf', 'NotoSerifBengali-Regular.ttf'],
            'gu': ['NotoSansGujarati-Regular.ttf', 'NotoSerifGujarati-Regular.ttf'],
            'kn': ['NotoSansKannada-Regular.ttf', 'NotoSerifKannada-Regular.ttf'],
            'ml': ['NotoSansMalayalam-Regular.ttf', 'NotoSerifMalayalam-Regular.ttf'],
            'or': ['NotoSansOriya-Regular.ttf'],
            'pa': ['NotoSansGurmukhi-Regular.ttf'],
            'ta': ['NotoSansTamil-Regular.ttf'],
            'te': ['NotoSansTelugu-Regular.ttf']
        }
        
        # Try local font files first
        if language_code in language_font_files:
            lang_dir = os.path.join(fonts_base_dir, language_code)
            if os.path.exists(lang_dir):
                for font_file in language_font_files[language_code]:
                    font_path = os.path.join(lang_dir, font_file)
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, 12)
                            title_font = ImageFont.truetype(font_path, 16)
                            header_font = ImageFont.truetype(font_path, 20)
                            print(f"✅ Loaded local font {font_file} for {language_code}")
                            break
                        except Exception as e:
                            print(f"⚠️ Failed to load {font_file}: {e}")
                            continue
        
        # If local fonts fail, try system fonts
        if not font:
            # Try language-specific fonts from languages.py
            for font_name in language_fonts:
                try:
                    font = ImageFont.truetype(font_name, 12)
                    title_font = ImageFont.truetype(font_name, 16)
                    header_font = ImageFont.truetype(font_name, 20)
                    print(f"✅ Loaded system font {font_name} for {language_code}")
                    break
                except:
                    continue
        
        # If still no font, try generic Indic fonts
        if not font and language_code in ['hi', 'sa', 'mr', 'bn', 'gu', 'kn', 'ml', 'or', 'pa', 'ta', 'te']:
            indic_fonts = [
                "NotoSansDevanagari-Regular",
                "mangal.ttf",
                "Mangal",
                "Noto Sans Devanagari",
                "DejaVu Sans",
                "Arial Unicode MS"
            ]
            for font_name in indic_fonts:
                try:
                    font = ImageFont.truetype(font_name, 12)
                    title_font = ImageFont.truetype(font_name, 16)
                    header_font = ImageFont.truetype(font_name, 20)
                    print(f"✅ Loaded generic Indic font {font_name} for {language_code}")
                    break
                except:
                    continue
        
        # Final fallback to default fonts
        if not font:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                title_font = ImageFont.truetype("arial.ttf", 16)
                header_font = ImageFont.truetype("arial.ttf", 20)
                print(f"⚠️ Using Arial fallback for {language_code}")
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                print(f"⚠️ Using default font for {language_code}")
                
    except Exception as e:
        print(f"⚠️ Font loading error: {e}")
        try:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        except:
            font = None
            title_font = None
            header_font = None
    
    # Layout calculations
    margin = 60
    header_height = 80
    footer_height = 60
    content_start_y = margin + header_height
    content_height = page_height - margin - footer_height - content_start_y
    
    # Determine layout type
    layout_type = getattr(config, 'layout_type', 'SINGLE_COLUMN')
    if hasattr(layout_type, 'value'):
        layout_type = layout_type.value
    
    # Calculate columns
    if layout_type == "TWO_COLUMN":
        num_columns = 2
        column_gap = 30
        column_width = (page_width - 2 * margin - column_gap) // 2
    elif layout_type == "THREE_COLUMN":
        num_columns = 3
        column_gap = 20
        column_width = (page_width - 2 * margin - 2 * column_gap) // 3
    elif layout_type == "NEWSLETTER":
        num_columns = 2
        column_gap = 40
        column_width = (page_width - 2 * margin - column_gap) // 2
    else:  # SINGLE_COLUMN
        num_columns = 1
        column_gap = 0
        column_width = page_width - 2 * margin
    
    # Draw header
    if header_font and hasattr(config, 'prompt') and config.prompt:
        draw.text((margin, 20), config.prompt[:50], fill='black', font=header_font)
        draw.line([(margin, 50), (page_width - margin, 50)], fill='black', width=2)
    
    # Draw footer
    if font:
        footer_text = f"Page {page_num + 1}"
        if hasattr(config, 'language'):
            # Handle both string and Language enum
            if hasattr(config.language, 'value'):
                footer_text += f" | {config.language.value.upper()}"
            else:
                footer_text += f" | {str(config.language).upper()}"
        draw.text((margin, page_height - 40), footer_text, fill='gray', font=font)
    
    current_y = content_start_y
    visual_elements = {'graphs': [], 'tables': [], 'images': [], 'equations': []}
    word_coords = []
    
    # Add visual elements strategically - GUARANTEE 1-2 per page with LOTS OF TEXT
    visual_elements_height = 0
    visual_elements_added = 0
    max_visual_elements = 2  # Limit to max 2 visual elements per page
    
    # Add AI-generated image - ALWAYS add one per page if enabled
    include_image = getattr(config, 'include_ai_images', False)
    if include_image and visual_elements_added < max_visual_elements:
        try:
            # Handle Language enum properly
            language_code = getattr(config, 'language', 'en')
            if hasattr(language_code, 'value'):
                language_code = language_code.value
            ai_image = generate_contextual_image(text, language_code)
            if ai_image:
                # Make image smaller to leave MORE room for text
                max_width = min(250, column_width)
                if ai_image.width > max_width:
                    aspect_ratio = ai_image.height / ai_image.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    ai_image = ai_image.resize((target_width, target_height))
                
                # Position image
                img_x = (page_width - ai_image.width) // 2
                img.paste(ai_image, (img_x, current_y))
                current_y += ai_image.height + 15
                visual_elements_height += ai_image.height + 15
                visual_elements_added += 1
                
                visual_elements['images'].append({
                    'type': 'ai_generated',
                    'title': f"Figure {page_num + 1}",
                    'position': {'x': img_x, 'y': current_y - ai_image.height - 15},
                    'size': {'width': ai_image.width, 'height': ai_image.height},
                    'description': f"Illustration for page {page_num + 1}"
                })
                
                # Small caption
                if font:
                    caption = f"Figure {page_num + 1}"
                    draw.text((img_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
        except Exception as e:
            print(f"⚠️ Error adding AI image: {e}")
    
    # Add graph OR table - add one more if we have room
    if visual_elements_added < max_visual_elements:
        # Alternate between graphs and tables
        if page_num % 2 == 0 and getattr(config, 'include_graphs', False):
            # Add graph on even pages
            try:
                available_graph_types = ["line", "bar", "pie"]
                available_types = [t for t in available_graph_types if t not in used_graph_types]
                
                if not available_types:
                    available_types = available_graph_types
                    used_graph_types.clear()
                
                graph_type = random.choice(available_types)
                used_graph_types.add(graph_type)
                
                graph_img = create_advanced_chart(graph_type)
                
                # Make graph smaller to leave MORE room for text
                max_width = min(300, column_width)
                if graph_img.width > max_width:
                    aspect_ratio = graph_img.height / graph_img.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    graph_img = graph_img.resize((target_width, target_height))
                
                # Position graph
                graph_x = (page_width - graph_img.width) // 2
                img.paste(graph_img, (graph_x, current_y))
                current_y += graph_img.height + 15
                visual_elements_height += graph_img.height + 15
                visual_elements_added += 1
                
                visual_elements['graphs'].append({
                    'type': graph_type,
                    'title': f'{graph_type.title()} Chart',
                    'position': {'x': graph_x, 'y': current_y - graph_img.height - 15},
                    'size': {'width': graph_img.width, 'height': graph_img.height},
                    'description': f"Data visualization chart"
                })
                
                # Small caption
                if font:
                    caption = f"Chart {page_num + 1}: {graph_type.title()}"
                    draw.text((graph_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
            except Exception as e:
                print(f"⚠️ Error adding graph: {e}")
                
        elif page_num % 2 == 1 and getattr(config, 'include_tables', False):
            # Add table on odd pages
            try:
                table_img = create_advanced_table()
                
                # Make table smaller to leave MORE room for text
                max_width = min(350, column_width)
                if table_img.width > max_width:
                    aspect_ratio = table_img.height / table_img.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    table_img = table_img.resize((target_width, target_height))
                
                # Position table
                table_x = (page_width - table_img.width) // 2
                img.paste(table_img, (table_x, current_y))
                current_y += table_img.height + 15
                visual_elements_height += table_img.height + 15
                visual_elements_added += 1
                
                visual_elements['tables'].append({
                    'type': 'data_table',
                    'title': 'Data Table',
                    'position': {'x': table_x, 'y': current_y - table_img.height - 15},
                    'size': {'width': table_img.width, 'height': table_img.height},
                    'rows': 4,
                    'columns': 3,
                    'description': "Data overview table"
                })
                
                # Small caption
                if font:
                    caption = f"Table {page_num + 1}: Data"
                    draw.text((table_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
            except Exception as e:
                print(f"⚠️ Error adding table: {e}")
    
    print(f"📊 Page {page_num}: Added {visual_elements_added} visual elements, focusing on text content")

    # Add text content in columns - DENSE TEXT LAYOUT
    remaining_height = content_height - visual_elements_height
    if remaining_height > 50:
        # Split text into columns
        words = text.split()
        words_per_column = len(words) // num_columns if num_columns > 1 else len(words)
        
        for col in range(num_columns):
            start_idx = col * words_per_column
            end_idx = start_idx + words_per_column if col < num_columns - 1 else len(words)
            
            if start_idx < len(words):
                column_text = ' '.join(words[start_idx:end_idx])
                
                # Calculate column position
                col_x = margin + col * (column_width + column_gap)
                
                # Add text with word wrapping - MUCH DENSER
                y_offset = current_y
                lines = wrap_text_to_width(column_text, column_width, font, draw)
                
                line_height = 16  # Reduced from 20 to 16 for denser text
                for i, line in enumerate(lines):
                    line_y = y_offset + i * line_height
                    if line_y + line_height < page_height - footer_height:
                        if font:
                            draw.text((col_x, line_y), line, fill='black', font=font)
                        
                        # Track word coordinates
                        x_offset = col_x
                        for word in line.split():
                            if font:
                                try:
                                    word_bbox = draw.textbbox((0, 0), word, font=font)
                                    word_width = word_bbox[2] - word_bbox[0]
                                except:
                                    word_width = len(word) * 7  # Reduced from 8 to 7
                            else:
                                word_width = len(word) * 7
                            
                            word_coords.append({
                                'word': word,
                                'x': x_offset,
                                'y': line_y,
                                'width': word_width,
                                'height': line_height,
                                'page': page_num,
                                'column': col
                            })
                            x_offset += word_width + 6  # Reduced spacing from 8 to 6
                    else:
                        break
    
    # Prepare metadata
    metadata = {
        'page_num': page_num,
        'layout_type': layout_type,
        'num_columns': num_columns,
        'visual_elements': visual_elements,
        'word_coords': word_coords,
        'layout_info': {
            'margin': margin,
            'header_height': header_height,
            'footer_height': footer_height,
            'column_width': column_width,
            'column_gap': column_gap
        },
        'content_height': content_height,
        'visual_elements_height': visual_elements_height,
        'page_dimensions': {'width': page_width, 'height': page_height}
    }
    
    return img, metadata

# Simple functions for workflow compatibility
def create_simple_chart(chart_type="bar", width=400, height=300):
    """Create simple charts for workflow compatibility"""
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=80)
    
    if chart_type == "bar":
        categories = ["A", "B", "C", "D"]
        values = [random.randint(20, 80) for _ in range(4)]
        ax.bar(categories, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax.set_title("Sample Bar Chart")
    elif chart_type == "line":
        x = list(range(2020, 2025))
        y = [random.randint(50, 150) for _ in range(5)]
        ax.plot(x, y, marker='o', linewidth=2, color='#2E86AB')
        ax.set_title("Trend Analysis")
    else:  # pie
        labels = ["X", "Y", "Z"]
        sizes = [30, 45, 25]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Distribution Chart")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def create_simple_table(width=400, height=200):
    """Create simple table for workflow compatibility"""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    headers = ["Item", "Value", "Status"]
    rows = [["Product A", "$100", "Active"], ["Product B", "$200", "Pending"], ["Product C", "$150", "Active"]]
    
    col_width = width // 3
    row_height = 30
    
    # Draw headers
    for i, header in enumerate(headers):
        x = i * col_width
        draw.rectangle([x, 0, x + col_width, row_height], fill='lightgray', outline='black')
        draw.text((x + 5, 5), header, fill='black', font=font)
    
    # Draw rows
    for row_idx, row in enumerate(rows):
        y = (row_idx + 1) * row_height
        for i, cell in enumerate(row):
            x = i * col_width
            draw.rectangle([x, y, x + col_width, y + row_height], outline='black')
            draw.text((x + 5, y + 5), cell, fill='black', font=font)
    
    return img

def create_placeholder_image(text_content, width=400, height=300):
    """Create placeholder contextual image"""
    img = Image.new('RGB', (width, height), '#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Simple keyword extraction
    words = text_content.lower().split()
    keywords = [w for w in words if len(w) > 4][:3]
    
    # Draw placeholder with keywords
    draw.rectangle([20, 20, width-20, height-20], outline='#333', width=2)
    draw.text((30, 30), "Generated Image", fill='black', font=font)
    if keywords:
        draw.text((30, 60), f"Keywords: {', '.join(keywords)}", fill='#666', font=font)
    
    # Add some simple shapes
    for i in range(3):
        x = 50 + i * 100
        y = 100 + random.randint(-20, 20)
        draw.ellipse([x, y, x+60, y+40], fill=f'#{random.randint(100,255):02x}{random.randint(100,255):02x}{random.randint(100,255):02x}')
    
    return img

def generate_ai_image(text_content, width=400, height=300):
    """Generate AI image - wrapper for compatibility"""
    return generate_contextual_image(text_content)
