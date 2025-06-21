import os
from PIL import ImageFont, Image, ImageDraw
import random

font = None
title_font = None

# Realistic font selections for different document types
realistic_fonts = {
    'academic': [
        "C:/Windows/Fonts/times.ttf",      # Times New Roman - standard academic
        "C:/Windows/Fonts/timesbd.ttf",    # Times Bold
        "C:/Windows/Fonts/georgia.ttf",    # Georgia - readable serif
        "C:/Windows/Fonts/calibri.ttf",    # Calibri - modern academic
    ],
    'business': [
        "C:/Windows/Fonts/arial.ttf",      # Arial - business standard
        "C:/Windows/Fonts/calibri.ttf",    # Calibri - modern business
        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI - Microsoft standard
        "C:/Windows/Fonts/tahoma.ttf",     # Tahoma - clean business
    ],
    'technical': [
        "C:/Windows/Fonts/consola.ttf",    # Consolas - technical/code
        "C:/Windows/Fonts/courier.ttf",    # Courier - technical reports
        "C:/Windows/Fonts/calibri.ttf",    # Calibri - technical documentation
        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI - modern technical
    ],
    'newsletter': [
        "C:/Windows/Fonts/arial.ttf",      # Arial - newsletter standard
        "C:/Windows/Fonts/verdana.ttf",    # Verdana - web/newsletter friendly
        "C:/Windows/Fonts/trebuc.ttf",     # Trebuchet MS - modern newsletter
        "C:/Windows/Fonts/tahoma.ttf",     # Tahoma - compact newsletter
    ]
}

# Fallback fonts for cross-platform compatibility
fallback_fonts = [
    "C:/Windows/Fonts/arial.ttf",   # Windows fallback
    "C:/Windows/Fonts/calibri.ttf", # Windows alternative
    "/System/Library/Fonts/Arial.ttf",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alternative
]

def load_realistic_font(document_type: str = 'academic', font_size: int = 14):
    """Load a realistic font based on document type"""
    
    fonts_to_try = realistic_fonts.get(document_type, realistic_fonts['academic'])
    
    # Try document-appropriate fonts first
    for font_path in fonts_to_try:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"✅ Loaded realistic {document_type} font: {font_path} (size: {font_size})")
                return font
        except Exception as e:
            continue
    
    # Fallback to general fonts
    for font_path in fallback_fonts:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"✅ Loaded fallback font: {font_path} (size: {font_size})")
                return font
        except Exception as e:
            continue
    
    # Ultimate fallback
    print(f"⚠️ Using default font (size: {font_size})")
    return ImageFont.load_default()

# Initialize with default academic fonts
for font_path in realistic_fonts['academic']:
    try:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 14)
            title_font = ImageFont.truetype(font_path, 18)
            print(f"✅ Successfully loaded font: {font_path} (size: 14, bold: False)")
            break
    except Exception as e:
        print(f"⚠️ Failed to load font {font_path}: {e}")
        continue
        
# Fallback to default fonts if none found
if not font:
    for font_path in fallback_fonts:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 14)
                title_font = ImageFont.truetype(font_path, 18)
                print(f"✅ Loaded fallback font: {font_path}")
                break
        except Exception as e:
            continue
            
# Ultimate fallback
if not font:
    try:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        print("⚠️ Using default font - may not display all characters correctly")
    except:
        font = None
        title_font = None
        print("⚠️ No font available - using basic rendering")



#custom func, when hindi doesnt work etc. 
def load_font(size: int = 14, bold: bool = False):
    """Load appropriate font with better Hindi support"""
    # Extended font paths with better Hindi support
    font_paths = [
        # Windows Hindi fonts (better support)
        "C:/Windows/Fonts/mangal.ttf",      # Primary Hindi font
        "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",  # Google Noto
    ]
    
    # Add bold versions if requested
    if bold:
        font_paths.extend([
            "C:/Windows/Fonts/mangalb.ttf",     # Bold Hindi font
            "C:/Windows/Fonts/arialbd.ttf",     # Bold Arial
            "C:/Windows/Fonts/calibrib.ttf",    # Bold Calibri
        ])
    
    # Add regular fallbacks
    font_paths.extend([
        "C:/Windows/Fonts/arial.ttf",       # Fallback
        "C:/Windows/Fonts/calibri.ttf",     # Alternative
        # macOS
        "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ])
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font_obj = ImageFont.truetype(font_path, size)
                # Test if font can render Hindi text
                test_text = "नमस्ते"
                try:
                    # Create a test image to verify font works
                    test_img = Image.new('RGB', (100, 50), 'white')
                    test_draw = ImageDraw.Draw(test_img)
                    test_draw.text((10, 10), test_text, fill='black', font=font_obj)
                    print(f"✅ Successfully loaded font: {font_path} (size: {size}, bold: {bold})")
                    return font_obj
                except Exception as e:
                    print(f"⚠️ Font works but Hindi test failed for {font_path}: {e}")
                    return font_obj  # Still return the font
            else:
                print(f"⚠️ Font not found: {font_path}")
        except Exception as e:
            print(f"⚠️ Failed to load font {font_path}: {e}")
            continue
      # Ultimate fallback
    try:
        print(f"⚠️ Using default font (size: {size})")
        return ImageFont.load_default()
    except:
        print("⚠️ No font available")
        return None

def load_title_font():
    return load_font(18)