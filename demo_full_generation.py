#!/usr/bin/env python3
"""
SynthDoc Full Generation Demo
============================
This script demonstrates all SynthDoc capabilities and saves outputs 
to a proper directory structure for examination.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def create_output_structure():
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"synthdoc_demo_output_{timestamp}")
    
    # Create subdirectories
    dirs = {
        'raw_documents': base_dir / "1_raw_documents",
        'handwriting': base_dir / "2_handwriting_samples", 
        'layout_augmentation': base_dir / "3_layout_augmentation",
        'vqa_datasets': base_dir / "4_vqa_datasets",
        'language_samples': base_dir / "5_language_samples",
        'summary': base_dir / "6_summary_reports"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created output directory: {base_dir}")
    print(f"📋 Structure: {len(dirs)} subdirectories")
    
    return base_dir, dirs

def demo_raw_document_generation(output_dir, api_key):
    """Generate raw documents with full LLM integration."""
    print("\n🔥 Generating Raw Documents with LLM...")
    
    try:
        from synthdoc import RawDocumentGenerator, RawDocumentGenerationConfig, Language, LayoutType
        
        # Initialize generator
        generator = RawDocumentGenerator(
            groq_api_key=api_key,
            save_dir=str(output_dir)
        )
        
        # Generate different types of documents
        configs = [
            {
                "name": "technical_report",
                "config": RawDocumentGenerationConfig(
                    language=Language.EN,
                    num_pages=2,
                    prompt="Generate a comprehensive technical report about renewable energy technologies, including solar, wind, and hydroelectric power systems",
                    layout_type=LayoutType.SINGLE_COLUMN,
                    include_graphs=True,
                    include_tables=True,
                    include_ai_images=False
                )
            },
            {
                "name": "business_analysis", 
                "config": RawDocumentGenerationConfig(
                    language=Language.EN,
                    num_pages=1,
                    prompt="Create a business analysis document discussing market trends in artificial intelligence and machine learning",
                    layout_type=LayoutType.TWO_COLUMN,
                    include_graphs=False,
                    include_tables=True,
                    include_ai_images=False
                )
            }
        ]
        
        results = []
        for doc_config in configs:
            print(f"  📄 Generating: {doc_config['name']}")
            result = generator.process(doc_config['config'])
            
            # Save additional metadata
            summary_file = output_dir / f"{doc_config['name']}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Document: {doc_config['name']}\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Pages: {doc_config['config'].num_pages}\n")
                f.write(f"Layout: {doc_config['config'].layout_type.value}\n")
                f.write(f"Metadata: {result.metadata}\n")
            
            results.append(result)
            print(f"    ✅ Saved with {result.num_samples} samples")
        
        return results
        
    except Exception as e:
        print(f"    ❌ Raw document generation failed: {e}")
        return []

def demo_handwriting_generation(output_dir):
    """Generate handwriting samples in multiple styles."""
    print("\n✍️ Generating Handwriting Samples...")
    
    try:
        from synthdoc import HandwritingGenerator, HandwritingGenerationConfig, Language
        
        generator = HandwritingGenerator(save_dir=str(output_dir))
        
        # Different handwriting styles and languages
        configs = [
            {
                "name": "english_print",
                "config": HandwritingGenerationConfig(
                    text_content="Dear Friend,\n\nThis is a sample of print handwriting in English. The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.\n\nBest regards,\nSynthDoc Demo",
                    language=Language.EN,
                    handwriting_style="print",
                    num_samples=2
                )
            },
            {
                "name": "english_cursive",
                "config": HandwritingGenerationConfig(
                    text_content="Meeting Notes - Project Planning\n\nDiscussed timeline for next quarter:\n• Q1: Research and development\n• Q2: Implementation phase\n• Q3: Testing and validation\n• Q4: Deployment and launch",
                    language=Language.EN,
                    handwriting_style="cursive", 
                    num_samples=2
                )
            },
            {
                "name": "multilingual_sample",
                "config": HandwritingGenerationConfig(
                    text_content="Shopping List:\n• Milk and bread\n• Fresh vegetables\n• Coffee beans\n• Notebook for writing\n• Stamps for letters",
                    language=Language.EN,
                    handwriting_style="default",
                    num_samples=3
                )
            }
        ]
        
        results = []
        for hw_config in configs:
            print(f"  ✍️ Generating: {hw_config['name']}")
            result = generator.process(hw_config['config'])
            
            # Save metadata
            summary_file = output_dir / f"{hw_config['name']}_info.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Handwriting Sample: {hw_config['name']}\n")
                f.write(f"Style: {hw_config['config'].handwriting_style}\n")
                f.write(f"Language: {hw_config['config'].language.value}\n")
                f.write(f"Samples: {hw_config['config'].num_samples}\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Files: {len(result.output_files)}\n")
            
            results.append(result)
            print(f"    ✅ Created {len(result.output_files)} handwriting samples")
        
        return results
        
    except Exception as e:
        print(f"    ❌ Handwriting generation failed: {e}")
        return []

def demo_layout_augmentation(output_dir):
    """Demonstrate layout augmentation capabilities."""
    print("\n🎨 Generating Layout Variations...")
    
    try:
        from synthdoc import LayoutAugmenter, LayoutAugmentationConfig, Language, LayoutType, AugmentationType
        from PIL import Image, ImageDraw, ImageFont
        
        # Create sample documents first
        sample_docs = []
        for i in range(2):
            # Create a sample document image
            img = Image.new('RGB', (800, 600), 'white')
            draw = ImageDraw.Draw(img)
            
            # Add some sample content
            try:
                font = ImageFont.truetype("Arial", 14)
            except:
                font = ImageFont.load_default()
            
            sample_text = f"""
Sample Document {i+1}

This is a demonstration of layout augmentation capabilities in SynthDoc.
The system can apply various transformations to existing documents:

• Layout modifications (single-column, two-column, newsletter)
• Font variations and language-specific typography  
• Visual augmentations (rotation, scaling, noise, blur)
• Multi-language content rendering

Technology Benefits:
- Improved model training diversity
- Enhanced document understanding
- Better OCR performance
- Robust layout detection

This sample contains realistic document content that will be processed
through the layout augmentation pipeline to create training variations.
            """.strip()
            
            # Draw text with basic formatting
            y_offset = 50
            for line in sample_text.split('\n'):
                draw.text((50, y_offset), line, fill='black', font=font)
                y_offset += 20
            
            doc_path = output_dir / f"sample_document_{i+1}.png"
            img.save(doc_path)
            sample_docs.append(str(doc_path))
        
        # Generate layout variations
        augmenter = LayoutAugmenter(save_dir=str(output_dir))
        
        config = LayoutAugmentationConfig(
            documents=sample_docs,
            languages=[Language.EN, Language.ES],
            layout_types=[LayoutType.SINGLE_COLUMN, LayoutType.TWO_COLUMN],
            augmentations=[AugmentationType.ROTATION, AugmentationType.NOISE]
        )
        
        result = augmenter.process(config)
        
        # Save summary
        summary_file = output_dir / "layout_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Layout Augmentation Results\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Input documents: {len(sample_docs)}\n")
            f.write(f"Languages: {[lang.value for lang in config.languages]}\n")
            f.write(f"Layout types: {[lt.value for lt in config.layout_types]}\n")
            f.write(f"Augmentations: {[aug.value for aug in config.augmentations]}\n")
            f.write(f"Output files: {len(result.output_files)}\n")
        
        print(f"    ✅ Generated {len(result.output_files)} layout variations")
        return [result]
        
    except Exception as e:
        print(f"    ❌ Layout augmentation failed: {e}")
        return []

def demo_vqa_generation(output_dir, api_key):
    """Generate VQA datasets with LLM integration."""
    print("\n❓ Generating VQA Datasets...")
    
    try:
        from synthdoc import VQAGenerator, VQAGenerationConfig
        
        # Sample document content
        documents = [
            "Renewable Energy Report: Solar power technology has advanced significantly in recent years. Modern solar panels achieve 20-22% efficiency rates, making them cost-effective for residential and commercial use. The global solar market is expected to grow by 25% annually through 2030.",
            
            "Financial Analysis Q3 2024: Revenue increased by 18% compared to Q2, reaching $2.4 million. Marketing campaigns drove a 35% increase in new customer acquisition. Operating expenses remained stable at $1.8 million, resulting in improved profit margins.",
            
            "Technology Overview: Artificial Intelligence applications in healthcare include diagnostic imaging, drug discovery, and patient monitoring. Machine learning models can analyze medical images with 95% accuracy, often outperforming human specialists in specific domains."
        ]
        
        generator = VQAGenerator(
            api_key=api_key,
            llm_model="groq/llama3-8b-8192"
        )
        
        config = VQAGenerationConfig(
            documents=documents,
            num_questions_per_doc=4,
            include_hard_negatives=True,
            question_types=["factual", "reasoning", "comparative"]
        )
        
        result = generator.process(config)
        
        # Save VQA summary
        summary_file = output_dir / "vqa_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"VQA Dataset Results\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Source documents: {len(documents)}\n")
            f.write(f"Questions per document: {config.num_questions_per_doc}\n")
            f.write(f"Question types: {config.question_types}\n")
            f.write(f"Include hard negatives: {config.include_hard_negatives}\n")
            f.write(f"Total questions: {result.metadata.get('total_questions', 'N/A')}\n")
            f.write(f"Processing cost: {result.metadata.get('cost_info', 'N/A')}\n")
        
        print(f"    ✅ Generated {result.metadata.get('total_questions', 'N/A')} questions")
        return [result]
        
    except Exception as e:
        print(f"    ❌ VQA generation failed: {e}")
        return []

def demo_language_support(output_dir):
    """Demonstrate multi-language capabilities."""
    print("\n🌍 Demonstrating Multi-Language Support...")
    
    try:
        from synthdoc import Language, LanguageSupport, HandwritingGenerator, HandwritingGenerationConfig
        from synthdoc.languages import get_language_name
        
        # Test different languages
        test_languages = [Language.EN, Language.HI, Language.ES, Language.ZH, Language.BN]
        
        # Sample content in different languages
        content_samples = {
            Language.EN: "Hello World! This is English text for multilingual testing.",
            Language.HI: "नमस्ते! यह बहुभाषी परीक्षण के लिए हिंदी पाठ है।",
            Language.ES: "¡Hola Mundo! Este es texto en español para pruebas multilingües.",
            Language.ZH: "你好世界！这是用于多语言测试的中文文本。",
            Language.BN: "হ্যালো ওয়ার্ল্ড! এটি বহুভাষিক পরীক্ষার জন্য বাংলা পাঠ।"
        }
        
        generator = HandwritingGenerator(save_dir=str(output_dir))
        
        language_results = []
        
        for lang in test_languages:
            lang_name = get_language_name(lang)
            print(f"  🌐 Testing: {lang.value} ({lang_name})")
            
            config = HandwritingGenerationConfig(
                text_content=content_samples.get(lang, content_samples[Language.EN]),
                language=lang,
                handwriting_style="default",
                num_samples=1
            )
            
            try:
                result = generator.process(config)
                language_results.append({
                    'language': lang,
                    'name': lang_name,
                    'result': result
                })
                print(f"    ✅ Generated sample for {lang_name}")
            except Exception as e:
                print(f"    ⚠️ {lang_name} generation skipped: {e}")
        
        # Save language support summary
        summary_file = output_dir / "language_support_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Multi-Language Support Demo\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Tested languages: {len(test_languages)}\n")
            f.write(f"Successful generations: {len(language_results)}\n\n")
            
            lang_support = LanguageSupport()
            supported = lang_support.get_supported_languages()
            f.write(f"Total supported languages: {len(supported)}\n")
            f.write(f"Supported codes: {', '.join(supported)}\n\n")
            
            for lang_result in language_results:
                f.write(f"- {lang_result['language'].value}: {lang_result['name']}\n")
        
        print(f"    ✅ Successfully tested {len(language_results)} languages")
        return language_results
        
    except Exception as e:
        print(f"    ❌ Language support demo failed: {e}")
        return []

def create_final_summary(base_dir, all_results):
    """Create a comprehensive summary of all generated content."""
    print("\n📊 Creating Final Summary...")
    
    summary_file = base_dir / "COMPLETE_DEMO_SUMMARY.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SynthDoc Complete Generation Demo - Summary Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Output Directory: {base_dir}\n\n")
        
        # Count generated files
        total_files = 0
        for item in base_dir.rglob("*"):
            if item.is_file() and item.suffix in ['.png', '.jpg', '.json']:
                total_files += 1
        
        f.write(f"📁 Directory Structure:\n")
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*")))
                f.write(f"  {subdir.name}: {file_count} files\n")
        
        f.write(f"\n📈 Generation Statistics:\n")
        f.write(f"  Total output files: {total_files}\n")
        
        if 'raw_documents' in all_results:
            f.write(f"  Raw documents: {len(all_results['raw_documents'])} generated\n")
        
        if 'handwriting' in all_results:
            total_hw_files = sum(len(r.output_files) for r in all_results['handwriting'])
            f.write(f"  Handwriting samples: {total_hw_files} generated\n")
        
        if 'layout' in all_results:
            total_layout_files = sum(len(r.output_files) for r in all_results['layout'])
            f.write(f"  Layout variations: {total_layout_files} generated\n")
        
        if 'vqa' in all_results:
            for r in all_results['vqa']:
                total_questions = r.metadata.get('total_questions', 0)
                f.write(f"  VQA questions: {total_questions} generated\n")
        
        if 'languages' in all_results:
            f.write(f"  Language samples: {len(all_results['languages'])} languages tested\n")
        
        f.write(f"\n🎯 Demo Completed Successfully!\n")
        f.write(f"All generated content is available in: {base_dir}\n")
        f.write(f"Explore the subdirectories to examine the outputs.\n")
    
    print(f"    ✅ Summary saved: {summary_file}")

def main():
    """Run the complete SynthDoc generation demo."""
    print("🚀 SynthDoc Complete Generation Demo")
    print("=" * 50)
    print("This demo showcases all SynthDoc capabilities with real output generation.")
    
    # API key for LLM features
    api_key = "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"
    
    # Create output structure
    base_dir, dirs = create_output_structure()
    
    # Run all demonstrations
    all_results = {}
    
    # Raw document generation
    all_results['raw_documents'] = demo_raw_document_generation(dirs['raw_documents'], api_key)
    
    # Handwriting generation
    all_results['handwriting'] = demo_handwriting_generation(dirs['handwriting'])
    
    # Layout augmentation
    all_results['layout'] = demo_layout_augmentation(dirs['layout_augmentation'])
    
    # VQA generation
    all_results['vqa'] = demo_vqa_generation(dirs['vqa_datasets'], api_key)
    
    # Language support
    all_results['languages'] = demo_language_support(dirs['language_samples'])
    
    # Create final summary
    create_final_summary(base_dir, all_results)
    
    print(f"\n🎉 Demo Complete!")
    print(f"📁 All outputs saved to: {base_dir}")
    print(f"📋 Check COMPLETE_DEMO_SUMMARY.txt for overview")
    print(f"🔍 Explore subdirectories to see generated content")
    
    return str(base_dir)

if __name__ == "__main__":
    output_dir = main()
    print(f"\n💡 To explore outputs: cd {output_dir}") 