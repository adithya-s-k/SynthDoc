"""
Example usage of SynthDoc library with .env configuration.

This script demonstrates various ways to use SynthDoc for document generation
with the newly implemented features including actual document rendering,
image processing augmentations, and VQA generation.

Setup:
1. Copy env.template to .env
2. Add your API keys to .env file
3. Run this script
"""

from synthdoc import SynthDoc, LanguageSupport, print_environment_status


def main():
    """Demonstrate SynthDoc usage with automatic .env loading."""
    print("🚀 SynthDoc Basic Usage Example with .env Configuration")
    print("=" * 60)

    # Show environment configuration status
    print("\n🔧 Environment Configuration:")
    print_environment_status()
    
    print("\n" + "=" * 60)

    # Initialize SynthDoc with automatic .env loading
    print("\n📚 Initializing SynthDoc...")
    synth = SynthDoc(output_dir="./examples_output")  # Automatically loads from .env
    
    if synth.api_key:
        print(f"✅ Using LLM for content generation: {synth.llm_model}")
    else:
        print("⚠️  No API key found - using fallback content generation")
        print("💡 Add API keys to .env file for full LLM features")

    # 1. Show supported languages
    print("\n📋 Supported Languages:")
    languages = synth.get_supported_languages()
    print(f"Total: {len(languages)} languages")

    # Show by category
    for category in ["Base", "Indic", "Other"]:
        category_langs = LanguageSupport.get_languages_by_category(category)
        print(f"\n{category} Languages:")
        for code, lang_info in category_langs.items():
            print(f"  {code}: {lang_info.name} ({lang_info.script.value})")

    # 2. Generate raw documents in English with actual rendering
    print("\n📄 Generating English Documents...")
    en_docs = synth.generate_raw_docs(
        language="en",
        num_pages=2,
        prompt="Generate technical documentation about machine learning",
        augmentations=["rotation", "noise"],
    )
    print(f"Generated {len(en_docs)} English documents")
    
    # Show details of first document
    if en_docs and en_docs[0].get("image_path"):
        print(f"  First document saved to: {en_docs[0]['image_path']}")
        print(f"  Image size: {en_docs[0]['metadata']['image_width']}x{en_docs[0]['metadata']['image_height']}")

    # 3. Generate documents in Hindi with actual rendering
    print("\n📄 Generating Hindi Documents...")
    hi_docs = synth.generate_raw_docs(
        language="hi", 
        num_pages=1, 
        prompt="मशीन लर्निंग के बारे में तकनीकी दस्तावेज़ बनाएं"
    )
    print(f"Generated {len(hi_docs)} Hindi documents")
    
    if hi_docs and hi_docs[0].get("image_path"):
        print(f"  Hindi document saved to: {hi_docs[0]['image_path']}")

    # 4. Apply layout augmentation with real image processing
    print("\n🎨 Applying Layout Augmentation...")
    all_docs = en_docs + hi_docs
    
    if all_docs:
        augmented_dataset = synth.augment_layout(
            documents=all_docs[:2],  # Use first 2 documents
            languages=["en", "hi"],
            fonts=["Arial", "Times New Roman"],
            augmentations=["rotation", "scaling", "brightness"],
        )
        
        num_variations = len(augmented_dataset.get("images", []))
        print(f"✅ Generated {num_variations} layout variations")
        print(f"   Available image paths: {len(augmented_dataset.get('image_paths', []))}")
    else:
        print("⚠️  No documents available for augmentation")

    # 5. Generate VQA dataset with LLM-powered questions
    print("\n❓ Generating VQA Dataset...")
    if all_docs:
        vqa_dataset = synth.generate_vqa(
            source_documents=all_docs[:2],  # Use first 2 documents
            question_types=["factual", "reasoning"],
            difficulty_levels=["easy", "medium"],
            hard_negative_ratio=0.3,
        )
        
        num_questions = len(vqa_dataset.get("questions", []))
        print(f"✅ Generated {num_questions} VQA pairs")
        
        # Show sample Q&A
        if num_questions > 0:
            print(f"\nSample Question: {vqa_dataset['questions'][0]}")
            print(f"Sample Answer: {vqa_dataset['answers'][0]}")
            print(f"Question Type: {vqa_dataset['question_types'][0]}")
            if vqa_dataset['hard_negatives'][0]:
                print(f"Hard Negative: {vqa_dataset['hard_negatives'][0][0]}")
    else:
        print("⚠️  No documents available for VQA generation")

    # 6. Generate handwritten document with paper backgrounds
    print("\n✍️ Generating Handwritten Documents...")
    
    handwriting_examples = [
        {
            "content": "Sample handwritten text for testing",
            "language": "en",
            "writing_style": "print",
            "paper_template": "lined"
        },
        {
            "content": "यह हिंदी में हस्तलिखित पाठ है",
            "language": "hi", 
            "writing_style": "print",
            "paper_template": "blank"
        }
    ]
    
    handwritten_docs = []
    for i, example in enumerate(handwriting_examples):
        handwritten = synth.generate_handwriting(**example)
        handwritten_docs.append(handwritten)
        
        if handwritten.get("image_path"):
            print(f"✅ Handwritten document {i+1} saved to: {handwritten['image_path']}")
        else:
            print(f"⚠️  Failed to generate handwritten document {i+1}: {handwritten.get('error', 'Unknown error')}")

    # 7. Show language-specific information
    print("\n🌐 Language Information Examples:")
    for lang_code in ["en", "hi", "zh", "ar"]:
        lang_info = synth.get_language_info(lang_code)
        if lang_info:
            print(f"\n{lang_code.upper()}:")
            print(f"  Name: {lang_info['name']}")
            print(f"  Script: {lang_info['script']}")
            print(f"  RTL: {lang_info['rtl']}")
            print(f"  Fonts: {', '.join(lang_info['fonts'][:3])}...")

    # 8. Demonstrate augmentation capabilities
    print("\n🔧 Augmentation Capabilities:")
    from synthdoc.augmentations import Augmentor
    
    augmentor = Augmentor()
    available_augs = augmentor.get_available_augmentations()
    print(f"Available augmentations: {', '.join(available_augs)}")
    
    # Test individual augmentations if we have images
    if all_docs and all_docs[0].get("image"):
        print("\n🧪 Testing Individual Augmentations:")
        test_image = all_docs[0]["image"]
        
        for aug_type in ["rotation", "brightness", "noise"]:
            try:
                if aug_type == "rotation":
                    aug_image = augmentor.apply_rotation(test_image, intensity=0.5)
                elif aug_type == "brightness":
                    aug_image = augmentor.apply_brightness(test_image, intensity=0.5)
                elif aug_type == "noise":
                    aug_image = augmentor.apply_noise(test_image, intensity=0.3)
                
                print(f"  ✅ {aug_type}: {aug_image.size} image")
            except Exception as e:
                print(f"  ❌ {aug_type}: {e}")

    # 9. Summary
    print("\n📊 Generation Summary:")
    print(f"  • English documents: {len(en_docs)}")
    print(f"  • Hindi documents: {len(hi_docs)}")
    print(f"  • Layout variations: {num_variations if 'num_variations' in locals() else 0}")
    print(f"  • VQA pairs: {num_questions if 'num_questions' in locals() else 0}")
    print(f"  • Handwritten documents: {len(handwritten_docs)}")
    
    print(f"\n📁 Output Directory: ./examples_output/")
    print("   Check the output directory for generated images!")
    
    # 10. Feature status
    print("\n🏁 Feature Status:")
    features = [
        ("✅", "Multi-language support"),
        ("✅", "LLM integration (with API key)"),
        ("✅", "Document rendering"),
        ("✅", "Image augmentations"),
        ("✅", "VQA generation"),
        ("✅", "Handwriting synthesis"),
        ("✅", "Font management"),
    ]
    
    for status, feature in features:
        print(f"  {status} {feature}")

    print("\n✅ Basic usage example complete!")
    print("\nNext steps:")
    print("  • Set OPENAI_API_KEY for enhanced LLM features")
    print("  • Try the integration_example.py for advanced features")
    print("  • Use the CLI: `synthdoc generate --lang hi --pages 3`")


if __name__ == "__main__":
    main()
