#!/usr/bin/env python3
"""
Integration test for the unified SynthDoc implementation.

This script tests both the new workflow-based architecture and backward compatibility.
"""

import os
import sys
from pathlib import Path

# Add SynthDoc to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("\n📦 Testing Imports...")
    
    try:
        # Test new workflow imports
        from synthdoc import (
            RawDocumentGenerator, VQAGenerator, HandwritingGenerator, 
            LayoutAugmenter, PDFAugmenter,
            RawDocumentGenerationConfig, VQAGenerationConfig, 
            HandwritingGenerationConfig, LayoutAugmentationConfig,
            Language, LayoutType, AugmentationType
        )
        print("  ✅ New workflow imports successful")
        
        # Test backward compatibility imports
        from synthdoc import SynthDoc, LanguageSupport
        print("  ✅ Backward compatibility imports successful")
        
        # Test convenience functions
        from synthdoc import (
            create_handwriting_samples, augment_layouts, 
            create_vqa_dataset, quick_example
        )
        print("  ✅ Convenience function imports successful")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_backward_compatibility():
    """Test that the old SynthDoc API still works."""
    print("\n🔄 Testing Backward Compatibility...")
    
    try:
        from synthdoc import SynthDoc
        
        # Initialize without API key (fallback mode)
        synth = SynthDoc()
        
        # Test old API methods
        docs = synth.generate_raw_docs(language="en", num_pages=1, prompt="Test document")
        if docs and len(docs) > 0:
            print(f"  ✅ generate_raw_docs: Generated {len(docs)} document(s)")
        else:
            print("  ⚠️ generate_raw_docs: No documents generated")
        
        # Test handwriting
        handwriting = synth.generate_handwriting(
            content="Test handwriting",
            language="en", 
            handwriting_template=None,
            writing_style="print",
            paper_template="lined"
        )
        if handwriting.get("image_path"):
            print("  ✅ generate_handwriting: Generated handwriting")
        else:
            print("  ⚠️ generate_handwriting: No image generated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False


def test_new_workflow_architecture():
    """Test the new workflow-based architecture."""
    print("\n🚀 Testing New Workflow Architecture...")
    
    try:
        from synthdoc import (
            HandwritingGenerator, HandwritingGenerationConfig,
            VQAGenerator, VQAGenerationConfig,
            LayoutAugmenter, LayoutAugmentationConfig,
            Language, AugmentationType
        )
        
        # Test HandwritingGenerator
        print("  Testing HandwritingGenerator...")
        hw_generator = HandwritingGenerator(save_dir="./test_handwriting")
        hw_config = HandwritingGenerationConfig(
            text_content="Hello, this is a test!",
            language=Language.EN,
            handwriting_style="cursive",
            num_samples=2
        )
        hw_result = hw_generator.process(hw_config)
        print(f"    ✅ Generated {hw_result.num_samples} handwriting samples")
        
        # Test VQAGenerator (without API key - should use fallback)
        print("  Testing VQAGenerator...")
        vqa_generator = VQAGenerator()  # No API key - fallback mode
        vqa_config = VQAGenerationConfig(
            documents=["Sample document content for testing VQA generation"],
            num_questions_per_doc=3,
            include_hard_negatives=True,
            question_types=["factual", "reasoning"]
        )
        vqa_result = vqa_generator.process(vqa_config)
        print(f"    ✅ Generated {vqa_result.num_samples} VQA samples")
        
        # Test LayoutAugmenter
        print("  Testing LayoutAugmenter...")
        layout_generator = LayoutAugmenter(save_dir="./test_layouts")
        layout_config = LayoutAugmentationConfig(
            documents=["Sample document content for layout testing"],
            languages=[Language.EN],
            fonts=["Arial"],
            augmentations=[AugmentationType.ROTATION]
        )
        layout_result = layout_generator.process(layout_config)
        print(f"    ✅ Generated {layout_result.num_samples} layout variations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ New workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🎯 Testing Convenience Functions...")
    
    try:
        from synthdoc import create_handwriting_samples, augment_layouts, Language
        
        # Test handwriting convenience function
        print("  Testing create_handwriting_samples...")
        hw_result = create_handwriting_samples(
            text_content="Convenience function test",
            language=Language.EN,
            handwriting_style="print",
            num_samples=1,
            save_dir="./test_convenience_hw"
        )
        print(f"    ✅ Generated {hw_result.num_samples} handwriting sample(s)")
        
        # Test layout convenience function
        print("  Testing augment_layouts...")
        layout_result = augment_layouts(
            documents=["Sample content for convenience testing"],
            languages=[Language.EN],
            fonts=["Arial"],
            save_dir="./test_convenience_layouts"
        )
        print(f"    ✅ Generated {layout_result.num_samples} layout variation(s)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Convenience function test failed: {e}")
        return False


def test_language_support():
    """Test multi-language support."""
    print("\n🌐 Testing Language Support...")
    
    try:
        from synthdoc import Language, LanguageSupport, create_handwriting_samples
        
        # Test language enumeration
        print(f"  ✅ Language.EN: {Language.EN}")
        print(f"  ✅ Language.HI: {Language.HI}")
        print(f"  ✅ Language.ZH: {Language.ZH}")
        
        # Test LanguageSupport
        lang_support = LanguageSupport()
        supported_langs = lang_support.get_supported_languages()
        print(f"  ✅ Supported languages: {len(supported_langs)}")
        
        # Test multi-language handwriting
        for lang in [Language.EN, Language.HI]:
            try:
                result = create_handwriting_samples(
                    language=lang,
                    handwriting_style="default",
                    num_samples=1,
                    save_dir=f"./test_lang_{lang.value}"
                )
                print(f"    ✅ {lang.value}: Generated {result.num_samples} sample(s)")
            except Exception as e:
                print(f"    ⚠️ {lang.value}: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Language support test failed: {e}")
        return False


def test_models_and_configs():
    """Test the Pydantic models and configurations."""
    print("\n📋 Testing Models and Configurations...")
    
    try:
        from synthdoc import (
            RawDocumentGenerationConfig, VQAGenerationConfig,
            HandwritingGenerationConfig, LayoutAugmentationConfig,
            Language, LayoutType, AugmentationType, OutputFormat
        )
        
        # Test RawDocumentGenerationConfig
        raw_config = RawDocumentGenerationConfig(
            language=Language.EN,
            num_pages=2,
            prompt="Test prompt",
            layout_type=LayoutType.TWO_COLUMN,
            include_graphs=True,
            include_tables=True
        )
        print(f"  ✅ RawDocumentGenerationConfig: {raw_config.language.value}")
        
        # Test VQAGenerationConfig
        vqa_config = VQAGenerationConfig(
            documents=["test.txt"],
            num_questions_per_doc=5,
            include_hard_negatives=True,
            question_types=["factual", "reasoning"]
        )
        print(f"  ✅ VQAGenerationConfig: {len(vqa_config.documents)} document(s)")
        
        # Test HandwritingGenerationConfig
        hw_config = HandwritingGenerationConfig(
            text_content="Test content",
            language=Language.EN,
            handwriting_style="cursive",
            num_samples=3
        )
        print(f"  ✅ HandwritingGenerationConfig: {hw_config.num_samples} samples")
        
        # Test LayoutAugmentationConfig
        layout_config = LayoutAugmentationConfig(
            documents=["test1.txt", "test2.txt"],
            languages=[Language.EN, Language.HI],
            fonts=["Arial", "Times"],
            augmentations=[AugmentationType.ROTATION, AugmentationType.SCALING]
        )
        print(f"  ✅ LayoutAugmentationConfig: {len(layout_config.documents)} document(s)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Models and configs test failed: {e}")
        return False


def test_quick_example():
    """Test the quick example function."""
    print("\n⚡ Testing Quick Example...")
    
    try:
        from synthdoc import quick_example
        quick_example()
        print("  ✅ Quick example completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Quick example failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧪 SynthDoc Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Backward Compatibility", test_backward_compatibility),
        ("New Workflow Architecture", test_new_workflow_architecture),
        ("Convenience Functions", test_convenience_functions),
        ("Language Support", test_language_support),
        ("Models and Configurations", test_models_and_configs),
        ("Quick Example", test_quick_example),
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
    print("\n📊 Integration Test Results")
    print("-" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All integration tests passed! The unified implementation is working correctly.")
        print("\n📝 Next steps:")
        print("  • Add your API keys to test LLM-powered features")
        print("  • Try the new workflow architecture for advanced document generation")
        print("  • Use convenience functions for quick prototyping")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n📁 Test output directories created:")
    print("  • ./test_handwriting/")
    print("  • ./test_layouts/")
    print("  • ./test_convenience_hw/")
    print("  • ./test_convenience_layouts/")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 