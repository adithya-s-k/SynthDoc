"""
SynthDoc - A library for generating synthetic documents for ML training.

A comprehensive library for generating synthetic documents designed for training
and evaluating models in document understanding tasks.
"""

from .core import SynthDoc
from .languages import LanguageSupport
from .generators import DocumentGenerator, LayoutGenerator, VQAGenerator
from .augmentations import Augmentor
from .config import SynthDocConfig, DEFAULT_CONFIG
from .fonts import FontManager

__version__ = "0.1.0"
__author__ = "Adithya Skolavi"

__all__ = [
    "SynthDoc",
    "LanguageSupport",
    "DocumentGenerator",
    "LayoutGenerator",
    "VQAGenerator",
    "Augmentor",
    "SynthDocConfig",
    "DEFAULT_CONFIG",
    "FontManager",
]
