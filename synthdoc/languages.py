"""
Language support configuration for SynthDoc.

This module provides comprehensive language and script support for the library.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum
from PIL import ImageFont, Image, ImageDraw
import os
import platform


class Language(Enum):
    """Supported languages enum for easy access."""
    
    EN = "en"
    HI = "hi"
    KN = "kn"
    TA = "ta"
    TE = "te"
    MR = "mr"
    PA = "pa"
    BN = "bn"
    OR = "or"
    ML = "ml"
    GU = "gu"
    SA = "sa"
    JA = "ja"
    KO = "ko"
    ZH = "zh"
    DE = "de"
    FR = "fr"
    IT = "it"
    RU = "ru"
    AR = "ar"
    ES = "es"
    TH = "th"


class ScriptType(Enum):
    """Supported script types."""

    LATIN = "Latin"
    DEVANAGARI = "Devanagari"
    KANNADA = "Kannada"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    GURMUKHI = "Gurmukhi"
    BENGALI = "Bengali"
    ODIA = "Odia"
    MALAYALAM = "Malayalam"
    GUJARATI = "Gujarati"
    KANJI_KANA = "Kanji/Kana"
    HANGUL = "Hangul"
    SIMPLIFIED_CHINESE = "Simplified"
    CYRILLIC = "Cyrillic"
    ARABIC = "Arabic"
    THAI = "Thai"


@dataclass
class LanguageInfo:
    """Language information structure."""

    code: str
    name: str
    script: ScriptType
    category: str
    rtl: bool = False  # Right-to-left reading direction
    font_families: List[str] = None

    def __post_init__(self):
        if self.font_families is None:
            self.font_families = []


class LanguageSupport:
    """Central language support configuration."""

    # Comprehensive language mapping
    LANGUAGES: Dict[str, LanguageInfo] = {
        # Base Languages
        "en": LanguageInfo(
            "en",
            "English",
            ScriptType.LATIN,
            "Base",
            font_families=["Arial", "Times New Roman", "Helvetica"],
        ),
        # Indic Languages
        "hi": LanguageInfo(
            "hi",
            "Hindi",
            ScriptType.DEVANAGARI,
            "Indic",
            font_families=["Mangal", "Noto Sans Devanagari", "Devanagari Sangam MN", "Kokila", "Utsaah", "Aparajita"],
        ),
        "kn": LanguageInfo(
            "kn",
            "Kannada",
            ScriptType.KANNADA,
            "Indic",
            font_families=["Tunga", "Noto Sans Kannada", "Kannada Sangam MN", "Kedage", "Akshar Unicode"],
        ),
        "ta": LanguageInfo(
            "ta",
            "Tamil",
            ScriptType.TAMIL,
            "Indic",
            font_families=["Latha", "Noto Sans Tamil", "Tamil Sangam MN", "Vijaya", "Bamini"],
        ),
        "te": LanguageInfo(
            "te",
            "Telugu",
            ScriptType.TELUGU,
            "Indic",
            font_families=["Gautami", "Noto Sans Telugu", "Telugu Sangam MN", "Vani", "Suranna"],
        ),
        "mr": LanguageInfo(
            "mr",
            "Marathi",
            ScriptType.DEVANAGARI,
            "Indic",
            font_families=["Mangal", "Noto Sans Devanagari"],
        ),
        "pa": LanguageInfo(
            "pa",
            "Punjabi",
            ScriptType.GURMUKHI,
            "Indic",
            font_families=["Raavi", "Noto Sans Gurmukhi"],
        ),
        "bn": LanguageInfo(
            "bn",
            "Bengali",
            ScriptType.BENGALI,
            "Indic",
            font_families=["Vrinda", "Noto Sans Bengali", "Bengali Sangam MN", "Shonar Bangla", "Kalpurush"],
        ),
        "or": LanguageInfo(
            "or",
            "Odia",
            ScriptType.ODIA,
            "Indic",
            font_families=["Kalinga", "Noto Sans Oriya"],
        ),
        "ml": LanguageInfo(
            "ml",
            "Malayalam",
            ScriptType.MALAYALAM,
            "Indic",
            font_families=["Kartika", "Noto Sans Malayalam"],
        ),
        "gu": LanguageInfo(
            "gu",
            "Gujarati",
            ScriptType.GUJARATI,
            "Indic",
            font_families=["Shruti", "Noto Sans Gujarati"],
        ),
        "sa": LanguageInfo(
            "sa",
            "Sanskrit",
            ScriptType.DEVANAGARI,
            "Indic",
            font_families=["Mangal", "Noto Sans Devanagari"],
        ),
        # Other Languages
        "ja": LanguageInfo(
            "ja",
            "Japanese",
            ScriptType.KANJI_KANA,
            "Other",
            font_families=["MS Gothic", "Noto Sans JP"],
        ),
        "ko": LanguageInfo(
            "ko",
            "Korean",
            ScriptType.HANGUL,
            "Other",
            font_families=["Malgun Gothic", "Noto Sans KR"],
        ),
        "zh": LanguageInfo(
            "zh",
            "Chinese",
            ScriptType.SIMPLIFIED_CHINESE,
            "Other",
            font_families=["SimSun", "Noto Sans SC", "Microsoft YaHei", "SimHei", "NSimSun", "FangSong"],
        ),
        "de": LanguageInfo(
            "de",
            "German",
            ScriptType.LATIN,
            "Other",
            font_families=["Arial", "Times New Roman"],
        ),
        "fr": LanguageInfo(
            "fr",
            "French",
            ScriptType.LATIN,
            "Other",
            font_families=["Arial", "Times New Roman"],
        ),
        "it": LanguageInfo(
            "it",
            "Italian",
            ScriptType.LATIN,
            "Other",
            font_families=["Arial", "Times New Roman"],
        ),
        "ru": LanguageInfo(
            "ru",
            "Russian",
            ScriptType.CYRILLIC,
            "Other",
            font_families=["Times New Roman", "Arial Unicode MS"],
        ),
        "ar": LanguageInfo(
            "ar",
            "Arabic",
            ScriptType.ARABIC,
            "Other",
            rtl=True,
            font_families=["Traditional Arabic", "Noto Sans Arabic"],
        ),
        "es": LanguageInfo(
            "es",
            "Spanish",
            ScriptType.LATIN,
            "Other",
            font_families=["Arial", "Times New Roman"],
        ),
        "th": LanguageInfo(
            "th",
            "Thai",
            ScriptType.THAI,
            "Other",
            font_families=["Tahoma", "Noto Sans Thai"],
        ),
    }

    @classmethod
    def get_language(cls, code: str) -> Optional[LanguageInfo]:
        """Get language information by code."""
        return cls.LANGUAGES.get(code)

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of all supported language codes."""
        return list(cls.LANGUAGES.keys())

    @classmethod
    def get_languages_by_category(cls, category: str) -> Dict[str, LanguageInfo]:
        """Get languages filtered by category."""
        return {
            code: lang
            for code, lang in cls.LANGUAGES.items()
            if lang.category == category
        }

    @classmethod
    def get_languages_by_script(cls, script: ScriptType) -> Dict[str, LanguageInfo]:
        """Get languages filtered by script type."""
        return {
            code: lang for code, lang in cls.LANGUAGES.items() if lang.script == script
        }

    @classmethod
    def is_rtl_language(cls, code: str) -> bool:
        """Check if language is right-to-left."""
        lang = cls.get_language(code)
        return lang.rtl if lang else False

    @classmethod
    def get_default_fonts(cls, code: str) -> List[str]:
        """Get default fonts for a language."""
        lang = cls.get_language(code)
        return lang.font_families if lang else ["Arial"]


# Simple font loading from local fonts folder
def get_local_fonts_path():
    """Get the path to the local fonts directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(current_dir, "fonts")
    return fonts_dir


def find_local_font(language_code: str) -> Optional[str]:
    """Find font file in local fonts directory."""
    fonts_dir = get_local_fonts_path()
    lang_fonts_dir = os.path.join(fonts_dir, language_code)

    if not os.path.exists(lang_fonts_dir):
        return None

    # Look for any .ttf file in the language directory
    try:
        for file in os.listdir(lang_fonts_dir):
            if file.endswith('.ttf'):
                font_path = os.path.join(lang_fonts_dir, file)
                if os.path.exists(font_path):
                    return font_path
    except (PermissionError, OSError):
        pass

    return None


# Simple font loading using local fonts folder
def load_language_font(language_code: str, size: int = 12, style: str = "regular"):
    """Load appropriate font for a language using local fonts folder."""
    try:
        # First try to load from local fonts folder
        local_font_path = find_local_font(language_code)
        if local_font_path:
            try:
                font = ImageFont.truetype(local_font_path, size)
                print(f"Loaded local font for {language_code}: {os.path.basename(local_font_path)}")
                return font
            except Exception:
                pass

        # Fallback to system fonts
        lang_support = LanguageSupport()
        fonts = lang_support.get_default_fonts(language_code)

        for font_name in fonts:
            try:
                font = ImageFont.truetype(font_name, size)
                print(f"Loaded system font: {font_name}")
                return font
            except Exception:
                continue

        # Final fallback
        print(f"Using default font for {language_code}")
        return ImageFont.load_default()

    except Exception as e:
        print(f"Font loading error for {language_code}: {e}")
        return ImageFont.load_default()


def get_language_name(language_code: Union[str, Language]) -> str:
    """Get language name from code."""
    # Handle Language enum
    if isinstance(language_code, Language):
        language_code = language_code.value
    
    lang_support = LanguageSupport()
    lang = lang_support.get_language(language_code)
    return lang.name if lang else "English"


def get_language_fonts(language_code: str) -> List[str]:
    """Get font families for a language."""
    lang_support = LanguageSupport()
    return lang_support.get_default_fonts(language_code)


def check_font_availability(language_code: str) -> Dict[str, bool]:
    """Check which fonts are available for a given language."""
    # Check if local font exists
    local_font = find_local_font(language_code)
    if local_font:
        return {"local_font": True}

    # Check system fonts
    lang_support = LanguageSupport()
    fonts = lang_support.get_default_fonts(language_code)
    availability = {}

    for font_name in fonts:
        try:
            ImageFont.truetype(font_name, 12)
            availability[font_name] = True
        except Exception:
            availability[font_name] = False

    return availability


def get_available_languages_with_fonts() -> List[str]:
    """Get list of languages that have at least one available font."""
    available_languages = []

    for lang_code in LanguageSupport.get_supported_languages():
        font_availability = check_font_availability(lang_code)
        if any(font_availability.values()):
            available_languages.append(lang_code)

    return available_languages





