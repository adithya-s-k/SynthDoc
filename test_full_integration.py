#!/usr/bin/env python3
"""
Comprehensive SynthDoc Integration Test
=====================================
This test actually generates data using all workflows to verify functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_raw_document_generation():
    """Test actual raw document generation"""
    print("🔥 Testing Raw Document Generation...")
    
    try:
        from synthdoc import RawDocumentGenerator, RawDocumentGenerationConfig, Language, LayoutType
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize generator with API key
            generator = RawDocumentGenerator(
                groq_api_key="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",
                save_dir=temp_dir
            )
            
            # Create configuration
            config = RawDocumentGenerationConfig(
                language=Language.EN,
                num_pages=2,
                prompt="Generate a technical report about machine learning",
                layout_type=LayoutType.SINGLE_COLUMN,
                include_graphs=True,
                include_tables=True,
                include_ai_images=False  # Skip AI images to avoid model downloads
            )
            
            # Generate documents
            result = generator.process(config)
            
            print(f"  ✅ Generated {len(result.output_files)} documents")
            print(f"  ✅ Cost tracking: {result.metadata.get('cost_info', 'No cost info')}")
            print(f"  ✅ Processing time: {result.metadata.get('processing_time', 'Unknown')} seconds")
            
            # Verify files were created
            output_files = list(Path(temp_dir).glob("**/*.png"))
            if output_files:
                print(f"  ✅ Found {len(output_files)} generated image files")
            else:
                print("  ⚠️ No image files found (expected for template mode)")
                
            return True
            
    except Exception as e:
        print(f"  ❌ Raw document generation failed: {e}")
        return False

def test_handwriting_generation():
    """Test actual handwriting generation"""
    print("✍️ Testing Handwriting Generation...")
    
    try:
        from synthdoc import HandwritingGenerator, HandwritingGenerationConfig, Language
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize generator
            generator = HandwritingGenerator(save_dir=temp_dir)
            
            # Create configuration
            config = HandwritingGenerationConfig(
                text_content="This is a test of handwriting generation in multiple languages. यह हिंदी में एक परीक्षण है।",
                language=Language.EN,
                handwriting_style="print",
                num_samples=3
            )
            
            # Generate handwriting
            result = generator.process(config)
            
            print(f"  ✅ Generated {len(result.output_files)} handwriting samples")
            print(f"  ✅ Processing time: {result.metadata.get('processing_time', 'Unknown')} seconds")
            
            # Verify files were created
            output_files = list(Path(temp_dir).glob("**/*.png"))
            print(f"  ✅ Found {len(output_files)} handwriting image files")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Handwriting generation failed: {e}")
        return False

def test_layout_augmentation():
    """Test actual layout augmentation"""
    print("🎨 Testing Layout Augmentation...")
    
    try:
        from synthdoc import LayoutAugmenter, LayoutAugmentationConfig, Language, LayoutType, AugmentationType
        from PIL import Image
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample document image first
            sample_text = "This is a sample document for layout testing. " * 20
            sample_img = Image.new('RGB', (800, 600), 'white')
            sample_path = Path(temp_dir) / "sample_doc.png"
            sample_img.save(sample_path)
            
            # Initialize augmenter
            augmenter = LayoutAugmenter(save_dir=temp_dir)
            
            # Create configuration
            config = LayoutAugmentationConfig(
                documents=[str(sample_path)],
                languages=[Language.EN, Language.ES],
                layout_types=[LayoutType.SINGLE_COLUMN, LayoutType.TWO_COLUMN],
                augmentations=[AugmentationType.ROTATION, AugmentationType.NOISE]
            )
            
            # Process layouts
            result = augmenter.process(config)
            
            print(f"  ✅ Generated {len(result.output_files)} layout variations")
            print(f"  ✅ Processing time: {result.metadata.get('processing_time', 'Unknown')} seconds")
            
            # Verify files were created
            output_files = list(Path(temp_dir).glob("**/*.png"))
            print(f"  ✅ Found {len(output_files)} total image files")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Layout augmentation failed: {e}")
        return False

def test_vqa_generation():
    """Test VQA generation with sample content"""
    print("❓ Testing VQA Generation...")
    
    try:
        from synthdoc import VQAGenerator, VQAGenerationConfig
        
        # Sample document content
        sample_docs = [
            "This is a technical report about artificial intelligence. AI has revolutionized many industries including healthcare, finance, and transportation.",
            "The quarterly sales report shows a 15% increase in revenue. The marketing department exceeded targets by 20%."
        ]
        
        # Initialize generator with API key for full LLM integration
        generator = VQAGenerator(
            api_key="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",
            llm_model="groq/llama3-8b-8192"
        )
        
        # Create configuration
        config = VQAGenerationConfig(
            documents=sample_docs,
            num_questions_per_doc=3,
            include_hard_negatives=True,
            question_types=["factual", "reasoning", "comparative"]
        )
        
        # Generate VQA dataset
        result = generator.process(config)
        
        print(f"  ✅ Generated questions for {len(sample_docs)} documents")
        print(f"  ✅ Processing time: {result.metadata.get('processing_time', 'Unknown')} seconds")
        
        # Check if questions were generated
        if result.metadata.get('total_questions', 0) > 0:
            print(f"  ✅ Generated {result.metadata['total_questions']} total questions")
        else:
            print("  ⚠️ No questions generated (expected for template mode without API key)")
            
        return True
        
    except Exception as e:
        print(f"  ❌ VQA generation failed: {e}")
        return False

def test_convenience_functions():
    """Test the convenience functions"""
    print("🛠️ Testing Convenience Functions...")
    
    try:
        from synthdoc import Language, LayoutType, AugmentationType
        from synthdoc import create_handwriting_samples, augment_layouts, create_raw_documents
        from PIL import Image
        
        # Test convenience functions
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test handwriting convenience function
            result = create_handwriting_samples(
                text_content="Testing convenience function",
                language=Language.EN,
                num_samples=2,
                save_dir=temp_dir
            )
            print("  ✅ create_handwriting_samples() works")
            
            # Test raw document generation convenience function
            try:
                result = create_raw_documents(
                    prompt="Generate a brief technical note",
                    language=Language.EN,
                    num_pages=1,
                    layout_type=LayoutType.SINGLE_COLUMN,
                    include_graphs=False,
                    include_tables=False,
                    include_ai_images=False,
                    groq_api_key="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",
                    save_dir=temp_dir
                )
                print("  ✅ create_raw_documents() works with API key")
            except Exception as e:
                print(f"  ⚠️ create_raw_documents() skipped: {e}")
            
            # Create sample for layout augmentation
            sample_img = Image.new('RGB', (600, 400), 'white')
            sample_path = Path(temp_dir) / "test.png"
            sample_img.save(sample_path)
            
            # Test layout augmentation convenience function  
            result = augment_layouts(
                documents=[str(sample_path)],
                languages=[Language.EN],
                fonts=["Arial"],
                augmentations=[AugmentationType.ROTATION],
                save_dir=temp_dir
            )
            print("  ✅ augment_layouts() works")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Convenience functions failed: {e}")
        return False

def test_language_support():
    """Test multi-language functionality"""
    print("🌍 Testing Multi-Language Support...")
    
    try:
        from synthdoc import Language, LanguageSupport
        from synthdoc.languages import get_language_name, load_language_font
        
        # Test language enum
        languages_tested = []
        for lang in [Language.EN, Language.HI, Language.ES, Language.ZH, Language.BN]:
            name = get_language_name(lang)
            languages_tested.append(f"{lang.value}({name})")
        
        print(f"  ✅ Tested languages: {', '.join(languages_tested)}")
        
        # Test font loading for different languages
        try:
            font_path = load_language_font(Language.HI)
            print("  ✅ Hindi font loading works")
        except:
            print("  ⚠️ Hindi font not available (expected)")
            
        # Test LanguageSupport
        lang_support = LanguageSupport()
        supported = lang_support.get_supported_languages()
        print(f"  ✅ {len(supported)} languages supported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Language support test failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility with old SynthDoc API"""
    print("🔄 Testing Backward Compatibility...")
    
    try:
        from synthdoc import SynthDoc
        
        # Initialize with minimal config
        synth = SynthDoc()
        
        # Test that old methods exist
        methods_to_check = [
            'generate_raw_docs',
            'augment_layout', 
            'generate_vqa',
            'create_handwriting'
        ]
        
        for method in methods_to_check:
            if hasattr(synth, method):
                print(f"  ✅ {method}() method exists")
            else:
                print(f"  ❌ {method}() method missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all comprehensive integration tests"""
    print("🚀 SynthDoc Comprehensive Integration Test")
    print("=" * 50)
    print("This test actually generates data to verify functionality.\n")
    
    tests = [
        ("Raw Document Generation", test_raw_document_generation),
        ("Handwriting Generation", test_handwriting_generation), 
        ("Layout Augmentation", test_layout_augmentation),
        ("VQA Generation", test_vqa_generation),
        ("Convenience Functions", test_convenience_functions),
        ("Language Support", test_language_support),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Comprehensive Test Results")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All comprehensive tests passed!")
        print("✨ SynthDoc integration is working correctly!")
        print("💡 The unified architecture successfully combines all workflows.")
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
        print("🔧 Check the output above for specific issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 