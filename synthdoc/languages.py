"""
Simple language support for SynthDoc using enum.
"""

from enum import Enum
from typing import Dict, List


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
