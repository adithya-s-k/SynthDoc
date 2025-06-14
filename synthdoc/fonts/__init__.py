"""
Font management and handling utilities for SynthDoc.

This module provides font discovery, validation, and management capabilities
for different languages and scripts.
"""

import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

from ..languages import LanguageSupport, ScriptType

logger = logging.getLogger(__name__)


class FontManager:
    """Manages fonts for different languages and scripts."""

    def __init__(self):
        self.system_fonts = self._discover_system_fonts()
        self.font_cache = {}
        self._validate_language_fonts()

    def _discover_system_fonts(self) -> Dict[str, List[str]]:
        """Discover available system fonts."""
        system = platform.system()
        fonts = {}

        if system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ]
        elif system == "Windows":
            font_paths = [
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
            ]
        elif system == "Linux":
            font_paths = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.local/share/fonts"),
                os.path.expanduser("~/.fonts"),
            ]
        else:
            font_paths = []

        for font_path in font_paths:
            if os.path.exists(font_path):
                fonts.update(self._scan_font_directory(font_path))

        return fonts

    def _scan_font_directory(self, directory: str) -> Dict[str, List[str]]:
        """Scan a directory for font files."""
        fonts = {}
        font_extensions = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in font_extensions):
                        font_name = os.path.splitext(file)[0]
                        full_path = os.path.join(root, file)

                        if font_name not in fonts:
                            fonts[font_name] = []
                        fonts[font_name].append(full_path)
        except (OSError, PermissionError):
            logger.warning(f"Could not scan font directory: {directory}")

        return fonts

    def _validate_language_fonts(self):
        """Validate that required fonts are available for each language."""
        missing_fonts = {}

        for lang_code, lang_info in LanguageSupport.LANGUAGES.items():
            available_fonts = []
            missing_for_lang = []

            for font_name in lang_info.font_families:
                if self._is_font_available(font_name):
                    available_fonts.append(font_name)
                else:
                    missing_for_lang.append(font_name)

            if missing_for_lang:
                missing_fonts[lang_code] = missing_for_lang
                logger.warning(f"Missing fonts for {lang_code}: {missing_for_lang}")

        if missing_fonts:
            logger.info("Consider installing missing fonts for better language support")

    def _is_font_available(self, font_name: str) -> bool:
        """Check if a specific font is available."""
        # Simple check - look for fonts with similar names
        font_name_lower = font_name.lower()
        for available_font in self.system_fonts.keys():
            if font_name_lower in available_font.lower():
                return True
        return False

    def get_available_fonts_for_language(self, lang_code: str) -> List[str]:
        """Get available fonts for a specific language."""
        lang_info = LanguageSupport.get_language(lang_code)
        if not lang_info:
            return ["Arial", "Times New Roman"]  # Fallback fonts

        available_fonts = []
        for font_name in lang_info.font_families:
            if self._is_font_available(font_name):
                available_fonts.append(font_name)

        # Add fallback fonts if none are available
        if not available_fonts:
            available_fonts = self._get_fallback_fonts_for_script(lang_info.script)

        return available_fonts

    def _get_fallback_fonts_for_script(self, script: ScriptType) -> List[str]:
        """Get fallback fonts for a script type."""
        fallback_map = {
            ScriptType.LATIN: ["Arial", "Times New Roman", "Helvetica"],
            ScriptType.DEVANAGARI: ["Arial Unicode MS", "Noto Sans Devanagari"],
            ScriptType.ARABIC: ["Arial Unicode MS", "Tahoma"],
            ScriptType.CYRILLIC: ["Arial Unicode MS", "Times New Roman"],
            ScriptType.KANJI_KANA: ["Arial Unicode MS", "MS Gothic"],
            ScriptType.HANGUL: ["Arial Unicode MS", "Malgun Gothic"],
            ScriptType.SIMPLIFIED_CHINESE: ["Arial Unicode MS", "SimSun"],
            ScriptType.THAI: ["Tahoma", "Arial Unicode MS"],
        }

        return fallback_map.get(script, ["Arial", "Times New Roman"])

    def get_font_path(self, font_name: str) -> Optional[str]:
        """Get the file path for a specific font."""
        for name, paths in self.system_fonts.items():
            if font_name.lower() in name.lower():
                return paths[0] if paths else None
        return None

    def list_all_fonts(self) -> List[str]:
        """List all available fonts."""
        return list(self.system_fonts.keys())

    def get_script_fonts(self, script: ScriptType) -> List[str]:
        """Get fonts suitable for a specific script."""
        fonts = []
        for lang_code, lang_info in LanguageSupport.LANGUAGES.items():
            if lang_info.script == script:
                fonts.extend(self.get_available_fonts_for_language(lang_code))

        # Remove duplicates and return
        return list(set(fonts))
