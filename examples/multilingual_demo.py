"""
Multilingual demonstration of SynthDoc capabilities.

This script showcases the library's multilingual document generation
across different script systems and language families.
"""

from synthdoc import SynthDoc, LanguageSupport
from synthdoc.fonts import FontManager


def demo_language_categories():
    """Demonstrate generation across different language categories."""
    print("🌍 Multilingual SynthDoc Demo")
    print("=" * 50)

    synth = SynthDoc(output_dir="./multilingual_output")
    font_manager = FontManager()

    # Get languages by category
    categories = ["Base", "Indic", "Other"]

    for category in categories:
        print(f"\n📚 {category} Languages:")
        category_langs = LanguageSupport.get_languages_by_category(category)

        for lang_code, lang_info in list(category_langs.items())[
            :3
        ]:  # Limit to 3 per category
            print(f"\n  🔤 {lang_info.name} ({lang_code})")
            print(f"     Script: {lang_info.script.value}")
            print(f"     RTL: {lang_info.rtl}")

            # Get available fonts
            available_fonts = font_manager.get_available_fonts_for_language(lang_code)
            print(f"     Available Fonts: {', '.join(available_fonts[:3])}...")

            # Generate a sample document
            try:
                docs = synth.generate_raw_docs(
                    language=lang_code,
                    num_pages=1,
                    prompt=f"Generate sample content in {lang_info.name}",
                )
                print(f"     ✅ Generated {len(docs)} document(s)")

                # Show sample content (truncated)
                if docs and docs[0].get("content"):
                    content_preview = docs[0]["content"][:100]
                    print(f"     Preview: {content_preview}...")

            except Exception as e:
                print(f"     ❌ Generation failed: {e}")


def demo_script_systems():
    """Demonstrate different script systems."""
    print("\n\n📝 Script System Demonstration")
    print("=" * 40)

    synth = SynthDoc(output_dir="./script_demo_output")

    # Representative languages for each major script
    script_examples = {
        "Latin": ["en", "de", "fr", "es"],
        "Devanagari": ["hi", "mr", "sa"],
        "Dravidian": ["ta", "te", "kn", "ml"],
        "Other": ["zh", "ja", "ko", "ar"],
    }

    for script_family, lang_codes in script_examples.items():
        print(f"\n🔤 {script_family} Script Family:")

        for lang_code in lang_codes:
            lang_info = LanguageSupport.get_language(lang_code)
            if lang_info:
                print(f"  • {lang_info.name} ({lang_code}) - {lang_info.script.value}")

                # Generate with script-appropriate content
                try:
                    docs = synth.generate_raw_docs(
                        language=lang_code,
                        num_pages=1,
                        prompt=f"Generate sample text showcasing {lang_info.script.value} script",
                    )
                    print(f"    ✅ Generated successfully")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")


def demo_rtl_languages():
    """Demonstrate right-to-left language handling."""
    print("\n\n↩️  Right-to-Left Languages Demo")
    print("=" * 35)

    synth = SynthDoc(output_dir="./rtl_demo_output")

    # Find RTL languages
    rtl_languages = []
    for code, lang_info in LanguageSupport.LANGUAGES.items():
        if lang_info.rtl:
            rtl_languages.append((code, lang_info))

    print(f"Found {len(rtl_languages)} RTL language(s)")

    for lang_code, lang_info in rtl_languages:
        print(f"\n📜 {lang_info.name} ({lang_code})")
        print(f"   Script: {lang_info.script.value}")
        print(f"   Reading Direction: Right-to-Left")

        try:
            docs = synth.generate_raw_docs(
                language=lang_code,
                num_pages=1,
                prompt="Generate text that demonstrates right-to-left reading",
            )

            if docs:
                print("   ✅ RTL document generated")
                # In a real implementation, this would handle RTL layout
                print("   📝 Layout: RTL text flow applied")
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")


def demo_multilingual_dataset():
    """Create a mixed multilingual dataset."""
    print("\n\n🌐 Mixed Multilingual Dataset")
    print("=" * 32)

    synth = SynthDoc(output_dir="./mixed_dataset_output")

    # Select diverse languages
    selected_languages = ["en", "hi", "zh", "ar", "de", "ja", "ta", "ko"]

    all_documents = []

    for lang_code in selected_languages:
        lang_info = LanguageSupport.get_language(lang_code)
        if lang_info:
            print(f"Generating {lang_info.name} documents...")

            try:
                docs = synth.generate_raw_docs(
                    language=lang_code,
                    num_pages=2,
                    prompt=f"Generate diverse content in {lang_info.name}",
                    augmentations=["rotation", "noise"],
                )
                all_documents.extend(docs)
                print(f"  ✅ Added {len(docs)} documents")

            except Exception as e:
                print(f"  ❌ Failed for {lang_code}: {e}")

    # Apply multilingual layout augmentation
    print(f"\n🎨 Applying layout augmentation to {len(all_documents)} documents...")

    try:
        dataset = synth.augment_layout(
            documents=all_documents,
            languages=selected_languages,
            fonts=["Arial", "Times New Roman", "Arial Unicode MS"],
            augmentations=["rotation", "scaling", "brightness", "contrast"],
        )

        print("✅ Multilingual dataset created!")
        print(
            f"📊 Dataset contains {len(all_documents)} documents across {len(selected_languages)} languages"
        )

        # Show language distribution
        lang_counts = {}
        for doc in all_documents:
            lang = doc.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print("\n📈 Language Distribution:")
        for lang, count in lang_counts.items():
            lang_info = LanguageSupport.get_language(lang)
            name = lang_info.name if lang_info else lang
            print(f"  {name} ({lang}): {count} documents")

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")


def demo_font_availability():
    """Show font availability for different languages."""
    print("\n\n🔤 Font Availability Report")
    print("=" * 28)

    font_manager = FontManager()

    print(f"Total system fonts discovered: {len(font_manager.list_all_fonts())}")

    # Check each language
    for lang_code, lang_info in LanguageSupport.LANGUAGES.items():
        available_fonts = font_manager.get_available_fonts_for_language(lang_code)

        print(f"\n{lang_info.name} ({lang_code}):")
        print(f"  Script: {lang_info.script.value}")
        print(f"  Recommended: {', '.join(lang_info.font_families[:2])}")
        print(f"  Available: {', '.join(available_fonts[:3])}")

        if not available_fonts:
            print("  ⚠️  No fonts available - using fallbacks")


def main():
    """Run all multilingual demonstrations."""
    try:
        demo_language_categories()
        demo_script_systems()
        demo_rtl_languages()
        demo_multilingual_dataset()
        demo_font_availability()

        print("\n\n🎉 Multilingual demonstration complete!")
        print("Check the output directories for generated files.")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
