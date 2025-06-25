"""
Language support configuration for SynthDoc.

This module provides comprehensive language and script support for the library.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


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
        "zh": LanguageInfo(
            "zh",
            "Chinese",
            ScriptType.SIMPLIFIED_CHINESE,
            "Base",
            font_families=["SimSun", "Noto Sans SC"],
        ),
        # Indic Languages
        "hi": LanguageInfo(
            "hi",
            "Hindi",
            ScriptType.DEVANAGARI,
            "Indic",
            font_families=["Mangal", "Noto Sans Devanagari"],
        ),
        "kn": LanguageInfo(
            "kn",
            "Kannada",
            ScriptType.KANNADA,
            "Indic",
            font_families=["Tunga", "Noto Sans Kannada"],
        ),
        "ta": LanguageInfo(
            "ta",
            "Tamil",
            ScriptType.TAMIL,
            "Indic",
            font_families=["Latha", "Noto Sans Tamil"],
        ),
        "te": LanguageInfo(
            "te",
            "Telugu",
            ScriptType.TELUGU,
            "Indic",
            font_families=["Gautami", "Noto Sans Telugu"],
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
            font_families=["Vrinda", "Noto Sans Bengali"],
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
