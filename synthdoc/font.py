import os
from PIL import ImageFont, Image, ImageDraw

font = None
title_font = None

# Try different font approaches for multi-language support
font_paths = [
    "C:/Windows/Fonts/mangal.ttf",  # Windows Hindi font
    "C:/Windows/Fonts/arial.ttf",   # Windows fallback
    "C:/Windows/Fonts/calibri.ttf", # Windows alternative
    "/System/Library/Fonts/Arial.ttf",  # macOS
    "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alternative
]

for font_path in font_paths:
    try:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 14)
            title_font = ImageFont.truetype(font_path, 18)
            print(f"✅ Loaded font: {font_path}")
            break
    except Exception as e:
        print(f"⚠️ Failed to load font {font_path}: {e}")
        continue
        
# Fallback to default
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
def load_font(size: int = 14):
        """Load appropriate font with better Hindi support"""
        # Extended font paths with better Hindi support
        font_paths = [
            # Windows Hindi fonts (better support)
            "C:/Windows/Fonts/mangal.ttf",      # Primary Hindi font
            "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",  # Google Noto
            "C:/Windows/Fonts/arial.ttf",       # Fallback
            "C:/Windows/Fonts/calibri.ttf",
            # macOS
            "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            # Linux
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size)
                    # Test if font can render Hindi text
                    test_text = "नमस्ते"
                    try:
                        # Create a test image to verify font works
                        test_img = Image.new('RGB', (100, 50), 'white')
                        test_draw = ImageDraw.Draw(test_img)
                        test_draw.text((10, 10), test_text, fill='black', font=font)
                        print(f"✅ Successfully loaded Hindi-capable font: {font_path}")
                        return font
                    except Exception as e:
                        print(f"⚠️ Font {font_path} loaded but failed Hindi test: {e}")
                        continue
            except Exception as e:
                print(f"⚠️ Failed to load font {font_path}: {e}")
                continue
        
        print("⚠️ No Hindi-capable font found, using default (may not display Hindi correctly)")
        return ImageFont.load_default()

def load_title_font():
    return load_font(18)