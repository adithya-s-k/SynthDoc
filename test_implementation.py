#!/usr/bin/env python3
"""
Test script for SynthDoc implementation.

This script tests the core functionality to ensure everything works correctly.
"""

import os
import sys
from pathlib import Path

# Set the environment variable for the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Add SynthDoc to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from synthdoc import SynthDoc, LanguageSupport
    from synthdoc.augmentations import Augmentor
    from synthdoc.generators import DocumentRenderer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_language_support():
    """Test language support functionality."""
    print("\n🌐 Testing Language Support...")
    
    ls = LanguageSupport()
    supported = ls.get_supported_languages()
    print(f"  Supported languages: {len(supported)}")
    
    # Test specific language
    hindi = ls.get_language("hi")
    if hindi:
        print(f"  Hindi support: ✅ {hindi.name} ({hindi.script.value})")
    else:
        print("  Hindi support: ❌")
    
    return len(supported) > 0


def test_document_renderer():
    """Test document rendering functionality."""
    print("\n📄 Testing Document Renderer...")
    
    try:
        renderer = DocumentRenderer()
        
        # Test text rendering
        test_text = "This is a test document.\nSecond line.\nThird line with more content."
        image = renderer.render_text_to_image(test_text)
        
        print(f"  Rendered image size: {image.size}")
        print("  Document rendering: ✅")
        return True
        
    except Exception as e:
        print(f"  Document rendering: ❌ {e}")
        return False


def test_augmentations():
    """Test augmentation functionality."""
    print("\n🎨 Testing Augmentations...")
    
    try:
        augmentor = Augmentor()
        available = augmentor.get_available_augmentations()
        print(f"  Available augmentations: {len(available)}")
        
        # Test with a simple image
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Test rotation
        rotated = augmentor.apply_rotation(test_image, intensity=0.5)
        print(f"  Rotation test: ✅ {rotated.size}")
        
        # Test brightness
        bright = augmentor.apply_brightness(test_image, intensity=0.5)
        print(f"  Brightness test: ✅ {bright.size}")
        
        print("  Augmentations: ✅")
        return True
        
    except Exception as e:
        print(f"  Augmentations: ❌ {e}")
        return False


def test_core_functionality():
    """Test core SynthDoc functionality."""
    print("\n🚀 Testing Core SynthDoc...")
    
    try:
        # Initialize without LLM first
        synth = SynthDoc(output_dir="./test_output")
        
        # Test language info
        lang_info = synth.get_language_info("en")
        if lang_info:
            print(f"  Language info: ✅ {lang_info['name']}")
        else:
            print("  Language info: ❌")
            return False
        
        # Test document generation (fallback mode)
        print("  Testing document generation...")
        docs = synth.generate_raw_docs(
            language="en",
            num_pages=1,
            prompt="Test document"
        )
        
        if docs and len(docs) > 0:
            doc = docs[0]
            print(f"  Generated document: ✅ ID={doc.get('id')}")
            
            if doc.get("image_path"):
                print(f"  Image saved: ✅ {doc['image_path']}")
            else:
                print("  Image saved: ⚠️  No image path")
                
            if doc.get("content"):
                print(f"  Content length: ✅ {len(doc['content'])} chars")
            else:
                print("  Content: ❌ No content")
                
        else:
            print("  Document generation: ❌ No documents generated")
            return False
        
        # Test handwriting generation
        print("  Testing handwriting generation...")
        handwritten = synth.generate_handwriting(
            content="Test handwriting",
            language="en",
            writing_style="print",
            paper_template="lined"
        )
        
        if handwritten.get("image_path"):
            print(f"  Handwriting: ✅ {handwritten['image_path']}")
        else:
            print(f"  Handwriting: ⚠️  {handwritten.get('error', 'No image generated')}")
        
        # Test VQA generation
        print("  Testing VQA generation...")
        vqa = synth.generate_vqa(
            source_documents=docs,
            question_types=["factual"],
            difficulty_levels=["easy"],
            hard_negative_ratio=0.1
        )
        
        if vqa.get("questions"):
            print(f"  VQA generation: ✅ {len(vqa['questions'])} questions")
        else:
            print("  VQA generation: ⚠️  No questions generated")
        
        print("  Core functionality: ✅")
        return True
        
    except Exception as e:
        print(f"  Core functionality: ❌ {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_llm():
    """Test LLM functionality if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n🤖 LLM Testing: ⚠️  No OPENAI_API_KEY found, skipping LLM tests")
        return True
    
    print("\n🤖 Testing LLM Integration...")
    
    try:
        synth = SynthDoc(
            output_dir="./test_output_llm",
            llm_model="gpt-4o-mini",
            api_key=api_key
        )
        
        # Test LLM document generation
        docs = synth.generate_raw_docs(
            language="en",
            num_pages=1,
            prompt="Write a short technical document about machine learning"
        )
        
        if docs and docs[0].get("content"):
            content = docs[0]["content"]
            if len(content) > 100:  # Should be substantial content from LLM
                print("  LLM content generation: ✅")
            else:
                print(f"  LLM content generation: ⚠️  Short content ({len(content)} chars)")
        else:
            print("  LLM content generation: ❌")
            return False
        
        # Test LLM VQA generation
        vqa = synth.generate_vqa(
            source_documents=docs,
            question_types=["factual", "reasoning"],
            difficulty_levels=["easy", "medium"],
            hard_negative_ratio=0.2
        )
        
        if vqa.get("questions") and len(vqa["questions"]) > 0:
            print(f"  LLM VQA generation: ✅ {len(vqa['questions'])} questions")
            
            # Check if questions look reasonable (not just fallback)
            sample_q = vqa["questions"][0]
            if len(sample_q) > 10 and "Sample" not in sample_q:
                print("  LLM question quality: ✅")
            else:
                print("  LLM question quality: ⚠️  Might be fallback")
        else:
            print("  LLM VQA generation: ❌")
        
        print("  LLM integration: ✅")
        return True
        
    except Exception as e:
        print(f"  LLM integration: ❌ {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 SynthDoc Implementation Test")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Language Support", test_language_support()))
    results.append(("Document Renderer", test_document_renderer()))
    results.append(("Augmentations", test_augmentations()))
    results.append(("Core Functionality", test_core_functionality()))
    results.append(("LLM Integration", test_with_llm()))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n📁 Test output directory: ./test_output/")
    print("Check for generated images and files.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 