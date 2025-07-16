"""
Translation interface and implementations for SynthDoc.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from .languages import Language

# Try to import deep_translator, but don't fail if it's not installed
try:
    from deep_translator import (
        GoogleTranslator,
        BaiduTranslator,
        ChatGptTranslator,
        LibreTranslator,
    )

    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False


class TranslationError(Exception):
    """Custom exception for translation errors."""

    pass


class DeepTranslatorType(str, Enum):
    """Supported translator types from deep_translator."""

    GOOGLE = "google"
    BAIDU = "baidu"
    CHATGPT = "chatgpt"
    LIBRE = "libre"


class Translator(ABC):
    """
    Abstract base class for a translator.
    """

    @abstractmethod
    def translate(
        self,
        texts: List[str],
        target_language: Language,
        source_language: Optional[Language] = None,
    ) -> List[str]:
        """
        Translate a list of texts to the target language.

        Args:
            texts: A list of strings to be translated.
            target_language: The target language for translation.
            source_language: The source language of the text. Auto-detect if None.

        Returns:
            A list of translated strings.

        Raises:
            TranslationError: If translation fails.
        """
        pass


class DeepTranslator(Translator):
    """
    Generic translator using the deep_translator library.
    """

    def __init__(
        self,
        translator_type: DeepTranslatorType = DeepTranslatorType.GOOGLE,
        **kwargs,
    ):
        if not DEEP_TRANSLATOR_AVAILABLE:
            raise ImportError(
                "deep_translator is not installed. Please install it with 'pip install deep-translator'"
            )

        self.translator_type = translator_type
        self.translator_kwargs = kwargs
        self._translator_class = self._get_translator_class()

    def _get_translator_class(self):
        if self.translator_type == DeepTranslatorType.GOOGLE:
            return GoogleTranslator
        elif self.translator_type == DeepTranslatorType.BAIDU:
            return BaiduTranslator
        elif self.translator_type == DeepTranslatorType.CHATGPT:
            return ChatGptTranslator
        elif self.translator_type == DeepTranslatorType.LIBRE:
            return LibreTranslator
        else:
            raise ValueError(f"Unsupported translator type: {self.translator_type}")

    def translate(
        self,
        texts: List[str],
        target_language: Language,
        source_language: Optional[Language] = None,
    ) -> List[str]:
        """
        Translate a list of texts using the configured translator.

        Args:
            texts: A list of strings to be translated.
            target_language: The target language for translation.
            source_language: The source language of the text. Auto-detect if None.

        Returns:
            A list of translated strings.

        Raises:
            TranslationError: If translation fails.
        """
        source_lang_code = source_language.value if source_language else "auto"
        target_lang_code = target_language.value

        try:
            if not texts:
                return []

            translator = self._translator_class(
                source=source_lang_code,
                target=target_lang_code,
                **self.translator_kwargs,
            )

            # Use translate_batch for multiple texts, and translate for a single text
            if len(texts) > 1:
                return translator.translate_batch(texts)
            elif texts:
                return [translator.translate(texts[0])]
            return []
        except Exception as e:
            raise TranslationError(
                f"Failed to translate using {self.translator_type.value}: {e}"
            ) from e
