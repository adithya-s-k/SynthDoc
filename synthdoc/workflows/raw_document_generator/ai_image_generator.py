import random
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont
from .text_utils import extract_keywords, categorize_content
from .content_analyzer import ContentAnalyzer

# Advanced AI image generation setup - lazy loading
_sd_pipe = None
_sd_available = None

def _initialize_sd_pipeline():
    """Initialize Stable Diffusion pipeline lazily"""
    global _sd_pipe, _sd_available
    
    if _sd_available is None:
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            print("🎨 Loading Stable Diffusion pipeline...")
            _sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                _sd_pipe = _sd_pipe.to("cuda")
            _sd_available = True
            print("Stable Diffusion loaded successfully")
        except Exception as e:
            _sd_available = False
            _sd_pipe = None
            print(f"Stable Diffusion not available: {e}")
    
    return _sd_available, _sd_pipe

    
def generate_contextual_image(text_content: str, language: str = "en", model_type: str = "stable-diffusion", max_retries: int = 3) -> Image.Image:
    """Generate high-quality contextual images using AI or create professional placeholders

    Args:
        text_content: Content to generate image for
        language: Language code for content
        model_type: AI model to use ('stable-diffusion', 'placeholder')
        max_retries: Maximum retry attempts for AI generation
    """
    # Handle edge cases
    if not text_content or len(text_content.strip()) < 10:
        print("Text content too short for meaningful image generation")
        return _create_professional_placeholder("Minimal content provided")

    if len(text_content) > 5000:
        print("Text content too long, truncating for image generation")
        text_content = text_content[:5000] + "..."

    # Try AI generation with retries
    if model_type == "stable-diffusion":
        for attempt in range(max_retries):
            try:
                sd_available, sd_pipe = _initialize_sd_pipeline()
                if sd_available and sd_pipe:
                    return _generate_ai_image_with_validation(text_content, sd_pipe, attempt)
                else:
                    break  # No point retrying if SD not available
            except Exception as e:
                print(f"AI image generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All AI generation attempts failed, using professional placeholder")

    # Fallback to professional placeholder
    return _create_professional_placeholder(text_content)



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



def _generate_ai_image_with_validation(text_content: str, sd_pipe, attempt: int = 0) -> Image.Image:
    """Generate CONTENT-AWARE AI image using Stable Diffusion with validation"""
    try:
#Analyze content for relevance
        analyzer = ContentAnalyzer()
        content_data = analyzer.extract_visual_data(text_content)
        
        # Build prompt based on ACTUAL content
        prompt = _build_content_aware_prompt(content_data, text_content)
        
        print(f"Content-aware prompt (attempt {attempt + 1}): {prompt[:60]}...")

        # Enhanced negative prompt for better quality
        negative_prompt = "text, watermark, signature, blurry, low quality, distorted, ugly, bad anatomy, deformed, nsfw, inappropriate"

        # Validate pipeline before generation
        if not hasattr(sd_pipe, '__call__'):
            raise ValueError("Invalid Stable Diffusion pipeline")

        # Generate with timeout protection
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Image generation timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            image = sd_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=min(20, 30 - attempt * 5),
                guidance_scale=7.5,
                width=512,
                height=384
            ).images[0]
        finally:
            signal.alarm(0)

        if not _validate_generated_image(image):
            raise ValueError("Generated image failed validation")

        resized_image = image.resize((400, 300), Image.Resampling.LANCZOS)
        print(f"Successfully generated RELEVANT AI image on attempt {attempt + 1}")
        return resized_image

    except Exception as e:
        print(f"Error generating content-aware AI image (attempt {attempt + 1}): {e}")
        if attempt < 2:
            raise e
        else:
            return _create_professional_placeholder(text_content)

def _build_content_aware_prompt(content_data: Dict[str, Any], text_content: str) -> str:
    """Build AI image prompt based on actual content analysis"""
    
    entities = content_data.get('entities', [])
    technical_concepts = content_data.get('technical_concepts', [])
    numbers = content_data.get('numbers', [])
    categories = content_data.get('categories', [])
    
    # Base prompt components
    main_subject = ""
    style_modifier = "professional, clean, corporate presentation style"
    
    # Primary subject from entities or technical concepts
    if technical_concepts:
        main_subject = f"diagram showing {technical_concepts[0]}"
        if len(technical_concepts) > 1:
            main_subject += f" and {technical_concepts[1]}"
    elif entities:
        main_subject = f"infographic about {entities[0]}"
        if len(entities) > 1:
            main_subject += f" compared to {entities[1]}"
    else:
        # Fallback to keyword extraction
        keywords = extract_keywords(text_content)
        if keywords:
            main_subject = f"illustration of {', '.join(keywords[:2])}"
        else:
            main_subject = "professional business diagram"
    
    # Add data visualization elements if numbers are present
    if numbers and len(numbers) >= 2:
        main_subject += f", showing data trends and metrics"
        style_modifier += ", data visualization, charts and graphs"
    
    # Add category-specific styling
    category = categorize_content(extract_keywords(text_content))
    category_styles = {
        'tech': ", modern tech design, blue and white color scheme, minimalist",
        'business': ", corporate design, professional color palette, clean layout",
        'research': ", academic style, scientific visualization, data-focused",
        'general': ", versatile professional design, balanced composition"
    }
    
    style_modifier += category_styles.get(category, category_styles['general'])
    
    # Construct final prompt
    final_prompt = f"{main_subject}, {style_modifier}, high quality, detailed illustration"
    
    return final_prompt


def _validate_generated_image(image: Image.Image) -> bool:
    """Validate that the generated image meets quality standards"""
    try:
        # Check image dimensions
        if image.width < 100 or image.height < 100:
            print(" Generated image too small")
            return False

        # Check if image is not completely black or white
        import numpy as np
        img_array = np.array(image)

        # Check for completely black image
        if np.all(img_array == 0):
            print("Generated image is completely black")
            return False

        # Check for completely white image
        if np.all(img_array == 255):
            print("Generated image is completely white")
            return False

        # Check for reasonable color variance
        variance = np.var(img_array)
        if variance < 100:  # Very low variance indicates poor quality
            print("Generated image has very low color variance")
            return False

        return True

    except Exception as e:
        print(f"Image validation error: {e}")
        return False


def _generate_ai_image(text_content: str, sd_pipe) -> Image.Image:
    """Generate real AI image using Stable Diffusion - legacy function"""
    return _generate_ai_image_with_validation(text_content, sd_pipe, 0)

def _create_professional_placeholder(text_content: str) -> Image.Image:
    """Create CONTENT-AWARE professional placeholder image"""
    width, height = 400, 300
    img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # NEW: Analyze content for placeholder relevance
    analyzer = ContentAnalyzer()
    content_data = analyzer.extract_visual_data(text_content)
    
    entities = content_data.get('entities', [])
    technical_concepts = content_data.get('technical_concepts', [])
    numbers = content_data.get('numbers', [])
    
    # Determine main theme from actual content
    if technical_concepts:
        main_theme = technical_concepts[0]
        category = 'tech'
    elif entities:
        main_theme = entities[0]
        category = 'business'
    else:
        keywords = extract_keywords(text_content)
        main_theme = keywords[0] if keywords else "Content"
        category = categorize_content(keywords)
    
    # Color scheme based on content
    color_schemes = {
        'tech': {'bg': '#E3F2FD', 'primary': '#1976D2', 'secondary': '#42A5F5'},
        'business': {'bg': '#F3E5F5', 'primary': '#7B1FA2', 'secondary': '#BA68C8'},
        'research': {'bg': '#E8F5E8', 'primary': '#388E3C', 'secondary': '#66BB6A'},
        'general': {'bg': '#FFF3E0', 'primary': '#F57C00', 'secondary': '#FFB74D'}
    }
    
    colors = color_schemes.get(category, color_schemes['general'])
    
    # Background and border
    draw.rectangle([0, 0, width, height], fill=colors['bg'])
    draw.rectangle([10, 10, width-10, height-10], outline=colors['primary'], width=3)
    
    # Title based on actual content
    title = f"{main_theme} Visualization"
    if len(title) > 25:
        title = title[:22] + "..."
    
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 25), title, fill=colors['primary'], font=title_font)
    
    # Show actual extracted data
    info_lines = []
    if entities:
        info_lines.append(f"Entities: {', '.join(entities[:2])}")
    if technical_concepts:
        info_lines.append(f"Concepts: {', '.join(technical_concepts[:2])}")
    if numbers:
        info_lines.append(f"Data Points: {len(numbers)} values")
    
    y_offset = 55
    for line in info_lines[:2]:  # Show max 2 lines
        if len(line) > 35:
            line = line[:32] + "..."
        line_bbox = draw.textbbox((0, 0), line, font=text_font)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (width - line_width) // 2
        draw.text((line_x, y_offset), line, fill=colors['secondary'], font=text_font)
        y_offset += 20
    
    # Visual elements based on actual data
    if numbers and len(numbers) >= 2:
        # Draw actual data as simple bars
        bar_values = [num['value'] for num in numbers[:4]]
        max_val = max(bar_values) if bar_values else 100
        
        for i, val in enumerate(bar_values):
            x = 60 + i * 70
            bar_height = int((val / max_val) * 80)
            draw.rectangle([x, 200-bar_height, x+30, 200], fill=colors['primary'])
            # Show actual value
            draw.text((x, 205), str(int(val)), fill=colors['primary'], font=text_font)
    
    elif entities and len(entities) >= 2:
        # Draw network/relationship diagram
        positions = [(120, 140), (280, 140), (200, 180)]
        for i, pos in enumerate(positions[:len(entities)]):
            draw.ellipse([pos[0]-25, pos[1]-15, pos[0]+25, pos[1]+15], fill=colors['primary'])
            entity_name = entities[i][:8] + "..." if len(entities[i]) > 8 else entities[i]
            text_bbox = draw.textbbox((0, 0), entity_name, font=text_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((pos[0] - text_width//2, pos[1] - 5), entity_name, fill='white', font=text_font)
        
        # Draw connections
        if len(positions) >= 2:
            draw.line([positions[0], positions[1]], fill=colors['secondary'], width=2)
            if len(positions) >= 3:
                draw.line([positions[1], positions[2]], fill=colors['secondary'], width=2)
    
    else:
        # Default geometric pattern
        for i in range(3):
            x = 80 + i * 80
            y = 140
            draw.ellipse([x, y, x+40, y+30], fill=colors['primary'])
    
    # Footer with content info
    footer_text = f"Content-Aware Visualization | Theme: {category.title()}"
    draw.text((15, height-25), footer_text, fill=colors['primary'], font=text_font)
    
    return img

def generate_ai_image(text_content, width=400, height=300):
    """Generate AI image - wrapper for compatibility"""
    return generate_contextual_image(text_content)