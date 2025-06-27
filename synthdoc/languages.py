"""
Simple language support for SynthDoc using enum.
"""

from enum import Enum
from typing import Dict, List
import os
from PIL import ImageFont

class Language(Enum):
    """Supported languages enum."""
    EN = "en"  # English
    HI = "hi"  # Hindi
    SA = "sa"  # Sanskrit
    BN = "bn"  # Bengali
    GU = "gu"  # Gujarati
    KN = "kn"  # Kannada
    ML = "ml"  # Malayalam
    MR = "mr"  # Marathi
    OR = "or"  # Odia
    PA = "pa"  # Punjabi
    TA = "ta"  # Tamil
    TE = "te"  # Telugu
    ZH = "zh"  # Chinese
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    DE = "de"  # German
    FR = "fr"  # French
    IT = "it"  # Italian
    RU = "ru"  # Russian
    AR = "ar"  # Arabic
    ES = "es"  # Spanish
    TH = "th"  # Thai


# Font mapping for each language
LANGUAGE_FONTS: Dict[str, List[str]] = {
    "en": ["Arial", "Times New Roman", "Helvetica"],
    "hi": ["Mangal", "Noto Sans Devanagari", "AnnapurnaSIL-Regular"],
    "sa": ["Mangal", "Noto Sans Devanagari"],
    "bn": ["Vrinda", "Noto Sans Bengali", "NotoSansBengali-Regular"],
    "gu": ["Shruti", "Noto Sans Gujarati", "NotoSansGujarati-Regular"],
    "kn": ["Tunga", "Noto Sans Kannada", "NotoSansKannada-Regular"],
    "ml": ["Kartika", "Noto Sans Malayalam", "NotoSansMalayalam-Regular"],
    "mr": ["Mangal", "Noto Sans Devanagari", "Mukta-Regular"],
    "or": ["Kalinga", "Noto Sans Oriya", "NotoSansOriya-Regular"],
    "pa": ["Raavi", "Noto Sans Gurmukhi"],
    "ta": ["Latha", "Noto Sans Tamil"],
    "te": ["Gautami", "Noto Sans Telugu", "NotoSansTelugu-Regular"],
    "zh": ["SimSun", "Noto Sans SC"],
    "ja": ["MS Gothic", "Noto Sans JP"],
    "ko": ["Malgun Gothic", "Noto Sans KR"],
    "de": ["Arial", "Times New Roman"],
    "fr": ["Arial", "Times New Roman"],
    "it": ["Arial", "Times New Roman"],
    "ru": ["Times New Roman", "Arial Unicode MS"],
    "ar": ["Traditional Arabic", "Noto Sans Arabic"],
    "es": ["Arial", "Times New Roman"],
    "th": ["Tahoma", "Noto Sans Thai"],
}

# Language display names
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "sa": "Sanskrit", 
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "es": "Spanish",
    "th": "Thai",
}


def get_language_fonts(language_code: str) -> List[str]:
    """Get fonts for a language code."""
    return LANGUAGE_FONTS.get(language_code, ["Arial"])


def get_language_name(language_code: str) -> str:
    """Get display name for a language code."""
    return LANGUAGE_NAMES.get(language_code, "Unknown")

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