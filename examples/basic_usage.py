"""
Example usage of SynthDoc library.

This script demonstrates various ways to use SynthDoc for document generation.
"""

from synthdoc import SynthDoc, LanguageSupport


def main():
    """Demonstrate SynthDoc usage."""
    print("🚀 SynthDoc Example Usage")
    print("=" * 50)

    # Initialize SynthDoc
    synth = SynthDoc(output_dir="./examples_output")

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

    # 2. Generate raw documents in English
    print("\n📄 Generating English Documents...")
    en_docs = synth.generate_raw_docs(
        language="en",
        num_pages=3,
        prompt="Generate technical documentation about machine learning",
        augmentations=["rotation", "noise"],
    )
    print(f"Generated {len(en_docs)} English documents")

    # 3. Generate documents in Hindi
    print("\n📄 Generating Hindi Documents...")
    hi_docs = synth.generate_raw_docs(
        language="hi", num_pages=2, prompt="मशीन लर्निंग के बारे में तकनीकी दस्तावेज़ बनाएं"
    )
    print(f"Generated {len(hi_docs)} Hindi documents")

    # 4. Apply layout augmentation
    print("\n🎨 Applying Layout Augmentation...")
    all_docs = en_docs + hi_docs
    augmented_dataset = synth.augment_layout(
        documents=all_docs,
        languages=["en", "hi"],
        fonts=["Arial", "Times New Roman", "Noto Sans Devanagari"],
        augmentations=["rotation", "scaling", "brightness"],
    )
    print("Layout augmentation complete")

    # 5. Generate VQA dataset
    print("\n❓ Generating VQA Dataset...")
    vqa_dataset = synth.generate_vqa(
        source_documents=all_docs,
        question_types=["factual", "reasoning", "comparative"],
        difficulty_levels=["easy", "medium", "hard"],
        hard_negative_ratio=0.3,
    )
    print("VQA dataset generation complete")

    # 6. Generate handwritten document
    print("\n✍️ Generating Handwritten Document...")
    handwritten = synth.generate_handwriting(
        content="Sample handwritten text for testing",
        language="en",
        writing_style="print",
        paper_template="lined",
    )
    print("Handwritten document generated")

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

    print("\n✅ Example complete! Check ./examples_output/ for results")


if __name__ == "__main__":
    main()
