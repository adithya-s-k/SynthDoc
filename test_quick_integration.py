#!/usr/bin/env python3
"""
Quick integration test for SynthDoc that avoids model downloads.
"""

import os
import sys
from pathlib import Path

# Add SynthDoc to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("📦 Testing Imports...")
    
    try:
        # Test core imports
        from synthdoc import (
            Language, LanguageSupport, SynthDoc,
            RawDocumentGenerationConfig, VQAGenerationConfig, 
            HandwritingGenerationConfig, LayoutAugmentationConfig,
            LayoutType, AugmentationType, OutputFormat,
            VQAGenerator, HandwritingGenerator, LayoutAugmenter
        )
        print("  ✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_language_enum():
    """Test the Language enum functionality."""
    print("🌐 Testing Language Enum...")
    
    try:
        from synthdoc import Language, LanguageSupport
        
        # Test enum values
        assert Language.EN.value == "en"
        assert Language.HI.value == "hi"
        assert Language.ZH.value == "zh"
        print("  ✅ Language enum values correct")
        
        # Test LanguageSupport
        lang_support = LanguageSupport()
        supported = lang_support.get_supported_languages()
        assert len(supported) > 10
        print(f"  ✅ LanguageSupport: {len(supported)} languages supported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Language enum test failed: {e}")
        return False


def test_pydantic_models():
    """Test the Pydantic model configurations."""
    print("📋 Testing Pydantic Models...")
    
    try:
        from synthdoc import (
            RawDocumentGenerationConfig, VQAGenerationConfig,
            HandwritingGenerationConfig, LayoutAugmentationConfig,
            Language, LayoutType, AugmentationType
        )
        
        # Test HandwritingGenerationConfig
        hw_config = HandwritingGenerationConfig(
            text_content="Test content",
            language=Language.EN,
            handwriting_style="cursive",
            num_samples=2
        )
        assert hw_config.language == Language.EN
        assert hw_config.num_samples == 2
        print("  ✅ HandwritingGenerationConfig created successfully")
        
        # Test VQAGenerationConfig
        vqa_config = VQAGenerationConfig(
            documents=["test.txt"],
            num_questions_per_doc=3,
            include_hard_negatives=True
        )
        assert vqa_config.num_questions_per_doc == 3
        print("  ✅ VQAGenerationConfig created successfully")
        
        # Test LayoutAugmentationConfig
        layout_config = LayoutAugmentationConfig(
            documents=["test1.txt"],
            languages=[Language.EN],
            fonts=["Arial"],
            augmentations=[AugmentationType.ROTATION]
        )
        assert len(layout_config.languages) == 1
        print("  ✅ LayoutAugmentationConfig created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pydantic models test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility without model downloads."""
    print("🔄 Testing Backward Compatibility...")
    
    try:
        from synthdoc import SynthDoc
        
        # Initialize SynthDoc
        synth = SynthDoc()
        
        # Test that methods exist (don't actually call them to avoid downloads)
        assert hasattr(synth, 'generate_raw_docs')
        assert hasattr(synth, 'generate_handwriting')
        assert hasattr(synth, 'generate_vqa')
        print("  ✅ SynthDoc methods exist")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False


def test_workflow_instantiation():
    """Test that workflow classes can be instantiated."""
    print("🚀 Testing Workflow Instantiation...")
    
    try:
        from synthdoc import HandwritingGenerator, VQAGenerator, LayoutAugmenter
        
        # Test instantiation (don't process to avoid model downloads)
        hw_gen = HandwritingGenerator(save_dir="./test_quick")
        assert hw_gen is not None
        print("  ✅ HandwritingGenerator instantiated")
        
        vqa_gen = VQAGenerator()  # No API key
        assert vqa_gen is not None
        print("  ✅ VQAGenerator instantiated")
        
        layout_gen = LayoutAugmenter(save_dir="./test_quick_layout")
        assert layout_gen is not None
        print("  ✅ LayoutAugmenter instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow instantiation test failed: {e}")
        return False


def main():
    """Run quick integration tests."""
    print("⚡ SynthDoc Quick Integration Test")
    print("=" * 45)
    
    tests = [
        ("Imports", test_imports),
        ("Language Enum", test_language_enum),
        ("Pydantic Models", test_pydantic_models),
        ("Backward Compatibility", test_backward_compatibility),
        ("Workflow Instantiation", test_workflow_instantiation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Quick Test Results")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Quick integration tests passed!")
        print("📝 The unified SynthDoc implementation is working correctly.")
        print("💡 For full testing with model downloads, run the complete test suite.")
    else:
        print("\n⚠️ Some quick tests failed. Check output above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 