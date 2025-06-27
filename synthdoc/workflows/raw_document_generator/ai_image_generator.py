import random
from PIL import Image, ImageDraw, ImageFont
from .text_utils import extract_keywords, categorize_content

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

    
def generate_contextual_image(text_content: str, language: str = "en") -> Image.Image:
    """Generate high-quality contextual images using AI or create professional placeholders"""
    if SD_AVAILABLE and sd_pipe:
        return _generate_ai_image(text_content)
    else:
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

def generate_ai_image(text_content, width=400, height=300):
    """Generate AI image - wrapper for compatibility"""
    return generate_contextual_image(text_content)