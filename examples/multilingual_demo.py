#!/usr/bin/env python3
"""
Multilingual SynthDoc Demo

This script demonstrates SynthDoc's comprehensive multilingual capabilities,
showcasing document generation across different languages and scripts.
"""

import os
from pathlib import Path
from synthdoc import SynthDoc, LanguageSupport, Language, create_raw_documents, create_handwriting_samples


def main():
    """Demonstrate SynthDoc's multilingual capabilities."""
    print("🌍 SynthDoc Multilingual Demo")
    print("=" * 50)
    
    # Initialize SynthDoc
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    output_dir = "./multilingual_output"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    if api_key:
        print("✅ Using LLM for advanced content generation")
        synth = SynthDoc(
            output_dir=output_dir,
            llm_model="gpt-4o-mini",
            api_key=api_key
        )
    else:
        print("⚠️  Using template content (set GROQ_API_KEY or OPENAI_API_KEY for LLM features)")
        synth = SynthDoc(output_dir=output_dir)

    # Demo 1: Multi-script Support
    print("\n📜 Demo 1: Multi-Script Document Generation")
    print("-" * 40)
    
    script_examples = [
        {
            "language": "en", 
            "name": "English (Latin)", 
            "prompt": "Generate a technical document about artificial intelligence",
            "sample_text": "Hello World - This is English text using Latin script."
        },
        {
            "language": "hi", 
            "name": "Hindi (Devanagari)", 
            "prompt": "कृत्रिम बुद्धिमत्ता के बारे में तकनीकी दस्तावेज़ बनाएं",
            "sample_text": "नमस्ते विश्व - यह देवनागरी लिपि का उदाहरण है।"
        },
        {
            "language": "ar", 
            "name": "Arabic (Arabic Script)", 
            "prompt": "إنشاء وثيقة تقنية حول الذكاء الاصطناعي",
            "sample_text": "مرحبا بالعالم - هذا مثال على النص العربي."
        },
        {
            "language": "zh", 
            "name": "Chinese (Simplified)", 
            "prompt": "生成关于人工智能的技术文档",
            "sample_text": "你好世界 - 这是中文简体字的例子。"
        },
        {
            "language": "ja", 
            "name": "Japanese (Hiragana/Kanji)", 
            "prompt": "人工知能に関する技術文書を生成する",
            "sample_text": "こんにちは世界 - これは日本語の例です。"
        }
    ]
    
    generated_docs = {}
    
    for example in script_examples:
        lang = example["language"]
        name = example["name"]
        
        print(f"\n🔤 Generating {name} document...")
        
        try:
            # Generate document using SynthDoc
            docs = synth.generate_raw_docs(
                language=lang,
                num_pages=1,
                prompt=example["prompt"]
            )
            
            generated_docs[lang] = docs
            
            if docs and docs[0].get("image_path"):
                print(f"  ✅ Generated: {docs[0]['image_path']}")
                print(f"  📝 Content preview: {docs[0].get('content', '')[:50]}...")
            else:
                print(f"  ⚠️  Generated but no image path")
            
        except Exception as e:
            print(f"  ❌ Error generating {name}: {e}")

    # Demo 2: Indic Languages Showcase  
    print("\n🕉️  Demo 2: Indic Languages Showcase")
    print("-" * 40)
    
    indic_languages = [
        {"code": "hi", "name": "Hindi", "script": "Devanagari"},
        {"code": "bn", "name": "Bengali", "script": "Bengali"},
        {"code": "ta", "name": "Tamil", "script": "Tamil"},
        {"code": "te", "name": "Telugu", "script": "Telugu"},
        {"code": "kn", "name": "Kannada", "script": "Kannada"},
        {"code": "ml", "name": "Malayalam", "script": "Malayalam"},
        {"code": "gu", "name": "Gujarati", "script": "Gujarati"},
        {"code": "or", "name": "Odia", "script": "Odia"},
        {"code": "pa", "name": "Punjabi", "script": "Gurmukhi"},
        {"code": "mr", "name": "Marathi", "script": "Devanagari"}
    ]
    
    indic_docs = []
    
    for lang_info in indic_languages[:5]:  # Generate for first 5 to save time
        lang_code = lang_info["code"]
        lang_name = lang_info["name"]
        script = lang_info["script"]
        
        print(f"\n📚 Generating {lang_name} ({script}) document...")
        
        try:
            docs = synth.generate_raw_docs(
                language=lang_code,
                num_pages=1,
                prompt=f"Create a technical document in {lang_name}"
            )
            
            indic_docs.extend(docs)
            
            if docs:
                print(f"  ✅ Generated {lang_name} document")
            
        except Exception as e:
            print(f"  ❌ Error generating {lang_name}: {e}")

    # Demo 3: Handwriting in Multiple Languages
    print("\n✍️  Demo 3: Multilingual Handwriting Generation")
    print("-" * 40)
    
    handwriting_examples = [
        {"lang": "en", "text": "The quick brown fox jumps over the lazy dog.", "style": "cursive"},
        {"lang": "hi", "text": "त्वरित भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।", "style": "print"},
        {"lang": "ar", "text": "الثعلب البني السريع يقفز فوق الكلب الكسول.", "style": "default"},
        {"lang": "zh", "text": "快速的棕色狐狸跳过懒惰的狗。", "style": "print"}
    ]
    
    handwriting_samples = []
    
    for example in handwriting_examples:
        lang = example["lang"]
        text = example["text"]
        style = example["style"]
        
        print(f"\n🖋️  Generating {lang.upper()} handwriting ({style})...")
        
        try:
            # Use new workflow function if API key available
            if api_key:
                result = create_handwriting_samples(
                    text_content=text,
                    language=Language(lang),
                    handwriting_style=style,
                    num_samples=1
                )
                handwriting_samples.append(result)
                print(f"  ✅ Generated handwriting sample")
            else:
                # Use legacy method
                handwritten = synth.generate_handwriting(
                    content=text,
                    language=lang,
                    writing_style=style
                )
                handwriting_samples.append(handwritten)
                print(f"  ✅ Generated handwriting sample")
                
        except Exception as e:
            print(f"  ❌ Error generating {lang} handwriting: {e}")

    # Demo 4: Font Variations Across Languages
    print("\n🔤 Demo 4: Font Variations Across Languages")
    print("-" * 40)
    
    font_demo_langs = ["en", "hi", "ar", "zh"]
    
    for lang in font_demo_langs:
        lang_info = synth.get_language_info(lang)
        if lang_info:
            fonts = lang_info.get("fonts", [])
            print(f"\n{lang.upper()} - Available fonts: {', '.join(fonts[:3])}...")
            
            # Generate document with different fonts
            if len(fonts) > 0:
                try:
                    docs = synth.augment_layout(
                        documents=generated_docs.get(lang, [])[:1],  # Use first doc if available
                        languages=[lang],
                        fonts=fonts[:2],  # Use first 2 fonts
                        augmentations=["rotation"]
                    )
                    
                    num_variations = len(docs.get("images", []))
                    print(f"  ✅ Generated {num_variations} font variations")
                    
                except Exception as e:
                    print(f"  ❌ Error with font variations: {e}")

    # Demo 5: Language Detection and Support Info
    print("\n🔍 Demo 5: Language Support Information")
    print("-" * 40)
    
    # Show all supported languages by category
    all_languages = synth.get_supported_languages()
    print(f"\n📊 Total supported languages: {len(all_languages)}")
    
    categories = ["Base", "Indic", "Other"]
    for category in categories:
        category_langs = LanguageSupport.get_languages_by_category(category)
        print(f"\n{category} Languages ({len(category_langs)}):")
        
        for code, lang_info in list(category_langs.items())[:5]:  # Show first 5
            print(f"  {code}: {lang_info.name} ({lang_info.script.value})")
        
        if len(category_langs) > 5:
            print(f"  ... and {len(category_langs) - 5} more")

    # Demo 6: RTL (Right-to-Left) Language Support
    print("\n↩️  Demo 6: RTL Language Support")
    print("-" * 40)
    
    rtl_languages = ["ar", "he", "fa"]  # Arabic, Hebrew, Persian
    
    for lang in rtl_languages:
        lang_info = synth.get_language_info(lang)
        if lang_info:
            print(f"\n{lang.upper()}: {lang_info['name']}")
            print(f"  RTL Support: {'✅' if lang_info['rtl'] else '❌'}")
            print(f"  Script: {lang_info['script']}")
            
            # Generate RTL document if supported
            if lang_info['rtl']:
                try:
                    rtl_docs = synth.generate_raw_docs(
                        language=lang,
                        num_pages=1,
                        prompt="Generate sample text"
                    )
                    
                    if rtl_docs:
                        print(f"  ✅ Generated RTL document")
                    
                except Exception as e:
                    print(f"  ❌ Error with RTL generation: {e}")

    # Summary
    print("\n📊 Multilingual Demo Summary")
    print("=" * 50)
    
    total_docs = sum(len(docs) for docs in generated_docs.values())
    total_handwriting = len(handwriting_samples)
    
    print(f"📄 Documents generated: {total_docs}")
    print(f"✍️  Handwriting samples: {total_handwriting}")
    print(f"🌍 Languages tested: {len(generated_docs)}")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🎯 Language Coverage:")
    print(f"  • Script systems: Latin, Devanagari, Arabic, Chinese, Japanese")
    print(f"  • Writing directions: LTR (Left-to-Right), RTL (Right-to-Left)")
    print(f"  • Font families: Multiple per language")
    print(f"  • Character sets: Unicode support")
    
    print(f"\n✨ Key Features Demonstrated:")
    features = [
        "✅ Multi-script document generation",
        "✅ Indic language support (10+ languages)",
        "✅ Multilingual handwriting synthesis",
        "✅ Font variation across languages",
        "✅ RTL language handling",
        "✅ Language-specific content generation",
        "✅ Unicode character support",
        "✅ Script-appropriate font selection"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n💡 Tips for Multilingual Usage:")
    tips = [
        "Set appropriate API keys for LLM-powered content generation",
        "Use language-specific prompts for better content quality",
        "Test font availability for target languages",
        "Consider text direction (LTR/RTL) for layout",
        "Validate Unicode encoding for special characters"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")


if __name__ == "__main__":
    main()
