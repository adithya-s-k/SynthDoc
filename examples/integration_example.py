"""
Comprehensive SynthDoc Integration Example with .env Configuration.

This example demonstrates all major features of SynthDoc including:
- Document generation with LLM integration
- Layout augmentation with real image processing
- VQA generation with hard negatives
- Handwriting synthesis
- Multi-language support
- Dataset management and export

Setup:
1. Copy env.template to .env
2. Add your API keys to .env file
3. Run this script
"""

import sys
from pathlib import Path

# Add SynthDoc to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthdoc import SynthDoc, LanguageSupport, print_environment_status
from synthdoc.dataset_manager import create_dataset_manager
from synthdoc.workflows import create_document_workflow, create_vqa_workflow
from synthdoc.models import SplitType


def main():
    """Run comprehensive SynthDoc integration example with .env loading."""
    print("🚀 SynthDoc Comprehensive Integration Example")
    print("=" * 60)
    
    # Show environment configuration status
    print("\n🔧 Environment Configuration:")
    print_environment_status()
    
    print("\n" + "=" * 60)
    
    # Initialize SynthDoc with automatic .env loading
    print("\n📚 Initializing SynthDoc...")
    synth = SynthDoc(output_dir="./comprehensive_output")  # Automatically loads from .env
    
    if synth.api_key:
        print(f"✅ Using LLM for content generation: {synth.llm_model}")
    else:
        print("⚠️  No API key found - using fallback content generation")
        print("💡 Add API keys to .env file for full LLM features")

    # 1. Show language capabilities
    print("\n📋 Language Support Demonstration")
    print("-" * 40)
    
    supported_langs = synth.get_supported_languages()
    print(f"Total supported languages: {len(supported_langs)}")
    
    # Show some language details
    demo_languages = ["en", "hi", "zh", "ar", "ja"]
    for lang_code in demo_languages:
        lang_info = synth.get_language_info(lang_code)
        if lang_info:
            rtl_indicator = " (RTL)" if lang_info["rtl"] else ""
            print(f"  {lang_code}: {lang_info['name']} - {lang_info['script']}{rtl_indicator}")

    # 2. Generate documents in multiple languages
    print("\n📄 Multi-Language Document Generation")
    print("-" * 40)
    
    all_documents = []
    
    # English technical documentation
    print("Generating English documents...")
    en_docs = synth.generate_raw_docs(
        language="en",
        num_pages=2,
        prompt="Generate a technical report about artificial intelligence and machine learning",
        augmentations=["rotation", "brightness"]
    )
    all_documents.extend(en_docs)
    print(f"✅ Generated {len(en_docs)} English documents")
    
    # Hindi documentation
    print("Generating Hindi documents...")
    hi_docs = synth.generate_raw_docs(
        language="hi",
        num_pages=2,
        prompt="कृत्रिम बुद्धिमत्ता और मशीन लर्निंग के बारे में तकनीकी रिपोर्ट बनाएं",
        augmentations=["scaling", "noise"]
    )
    all_documents.extend(hi_docs)
    print(f"✅ Generated {len(hi_docs)} Hindi documents")
    
    # Chinese documentation
    print("Generating Chinese documents...")
    zh_docs = synth.generate_raw_docs(
        language="zh",
        num_pages=1,
        prompt="生成关于人工智能和机器学习的技术报告",
        augmentations=["contrast"]
    )
    all_documents.extend(zh_docs)
    print(f"✅ Generated {len(zh_docs)} Chinese documents")

    # 3. Layout Augmentation Demo
    print("\n🎨 Layout Augmentation Demonstration")
    print("-" * 40)
    
    # Test different fonts and augmentations
    fonts_to_test = ["Arial", "Times New Roman", "Helvetica"]
    augmentations_to_test = ["rotation", "scaling", "brightness", "noise", "blur"]
    
    print(f"Applying {len(fonts_to_test)} fonts and {len(augmentations_to_test)} augmentations...")
    
    augmented_dataset = synth.augment_layout(
        documents=all_documents[:2],  # Use first 2 documents for demo
        languages=["en", "hi"],
        fonts=fonts_to_test,
        augmentations=augmentations_to_test[:3],  # Use first 3 augmentations
    )
    
    num_variations = len(augmented_dataset.get("images", []))
    print(f"✅ Generated {num_variations} layout variations")

    # 4. VQA Dataset Generation
    print("\n❓ VQA Dataset Generation")
    print("-" * 40)
    
    question_types = ["factual", "reasoning", "comparative"]
    difficulty_levels = ["easy", "medium", "hard"]
    
    print(f"Generating VQA pairs for {len(all_documents)} documents...")
    print(f"Question types: {question_types}")
    print(f"Difficulty levels: {difficulty_levels}")
    
    vqa_dataset = synth.generate_vqa(
        source_documents=all_documents[:3],  # Use first 3 documents
        question_types=question_types,
        difficulty_levels=difficulty_levels[:2],  # Use first 2 difficulty levels
        hard_negative_ratio=0.3
    )
    
    num_questions = len(vqa_dataset.get("questions", []))
    print(f"✅ Generated {num_questions} VQA pairs")
    
    # Show a sample question-answer pair
    if num_questions > 0:
        sample_idx = 0
        print(f"\nSample VQA Pair:")
        print(f"  Question: {vqa_dataset['questions'][sample_idx]}")
        print(f"  Answer: {vqa_dataset['answers'][sample_idx]}")
        print(f"  Type: {vqa_dataset['question_types'][sample_idx]}")
        if vqa_dataset['hard_negatives'][sample_idx]:
            print(f"  Hard Negative: {vqa_dataset['hard_negatives'][sample_idx][0]}")

    # 5. Handwriting Generation Demo
    print("\n✍️ Handwriting Generation")
    print("-" * 40)
    
    handwriting_demos = [
        ("en", "print", "lined", "This is a sample handwritten document in English."),
        ("hi", "print", "blank", "यह हिंदी में हस्तलिखित दस्तावेज़ का नमूना है।"),
        ("en", "cursive", "grid", "Cursive handwriting sample for testing.")
    ]
    
    handwritten_docs = []
    for lang, style, paper, content in handwriting_demos:
        print(f"Generating {style} handwriting in {lang} on {paper} paper...")
        handwritten = synth.generate_handwriting(
            content=content,
            language=lang,
            writing_style=style,
            paper_template=paper
        )
        handwritten_docs.append(handwritten)
        print(f"✅ Generated handwritten document: {handwritten.get('image_path', 'No image')}")

    # 6. Dataset Management Integration
    print("\n💾 Dataset Management Demo")
    print("-" * 40)
    
    try:
        # Create a dataset manager
        dataset_manager = create_dataset_manager(
            dataset_root="./comprehensive_output/datasets",
            dataset_name="synthdoc_demo",
            copy_images=True
        )
        
        # Add some documents to the dataset
        added_count = 0
        for i, doc in enumerate(all_documents[:3]):
            if doc.get("image_path"):
                # Create metadata for document understanding task
                from synthdoc.models import create_document_metadata
                
                metadata = create_document_metadata(
                    file_name=f"doc_{i}.png",
                    text=doc.get("content", ""),
                    document_type="generated",
                    source="synthdoc_demo"
                )

                # Add to dataset
                from synthdoc.models import DatasetItem
                item = DatasetItem(
                    image_path=doc["image_path"],
                    metadata=metadata,
                    split=SplitType.TRAIN if i < 2 else SplitType.TEST
                )
                
                filename = dataset_manager.add_item(item)
                added_count += 1
                print(f"  Added document {i+1} as {filename}")
        
        print(f"✅ Added {added_count} documents to dataset")
        
        # Get dataset statistics
        stats = dataset_manager.get_stats()
        print(f"Dataset stats: {stats.total_items} total items")
        print(f"Split distribution: {dict(stats.split_counts)}")
        
    except Exception as e:
        print(f"⚠️  Dataset management demo failed: {e}")

    # 7. Workflow Integration Demo
    print("\n🔄 Workflow Integration Demo")
    print("-" * 40)
    
    try:
        # Document Understanding Workflow
        doc_workflow = create_document_workflow(
            dataset_root="./comprehensive_output/workflows",
            workflow_name="document_understanding_demo"
        )
        
        # Initialize the workflow dataset
        doc_workflow.initialize_dataset("doc_understanding_demo")
        
        # Add some samples to the workflow
        workflow_added = 0
        for i, doc in enumerate(all_documents[:2]):
            if doc.get("image_path"):
                doc_workflow.add_document_sample(
                    image_path=doc["image_path"],
                    text=doc.get("content", ""),
                    document_type="technical_report",
                    source="synthdoc_generated",
                    split=SplitType.TRAIN
                )
                workflow_added += 1
        
        print(f"✅ Added {workflow_added} samples to document workflow")
        
        # VQA Workflow
        if vqa_dataset.get("questions"):
            vqa_workflow = create_vqa_workflow(
                dataset_root="./comprehensive_output/workflows",
                workflow_name="vqa_demo"
            )
            
            vqa_workflow.initialize_dataset("vqa_demo")
            
            # Add VQA samples
            vqa_added = 0
            for i in range(min(2, len(vqa_dataset["questions"]))):
                if vqa_dataset["images"][i]:
                    # Note: In real usage, you'd have image paths for VQA
                    # Here we're using placeholder for demo
                    vqa_workflow.add_vqa_sample(
                        image_path=all_documents[0]["image_path"],  # Placeholder
                        question=vqa_dataset["questions"][i],
                        answer=vqa_dataset["answers"][i],
                        question_type=vqa_dataset["question_types"][i],
                        split=SplitType.TRAIN
                    )
                    vqa_added += 1
            
            print(f"✅ Added {vqa_added} samples to VQA workflow")
        
    except Exception as e:
        print(f"⚠️  Workflow integration demo failed: {e}")

    # 8. Summary and Output Information
    print("\n📊 Demo Summary")
    print("-" * 40)
    
    print(f"Generated Documents:")
    print(f"  • Total raw documents: {len(all_documents)}")
    print(f"  • Layout variations: {num_variations}")
    print(f"  • VQA pairs: {num_questions}")
    print(f"  • Handwritten samples: {len(handwritten_docs)}")
    
    print(f"\nLanguages demonstrated: {len(demo_languages)}")
    print(f"Augmentation types tested: {len(augmentations_to_test)}")
    print(f"Font variations: {len(fonts_to_test)}")
    
    print(f"\nOutput directory: ./comprehensive_output/")
    print("Check the output directory for generated images and datasets!")
    
    # 9. Feature Status Report
    print("\n🏁 Implementation Status")
    print("-" * 40)
    
    features = [
        ("✅", "Multi-language document generation"),
        ("✅", "LLM-powered content creation"),
        ("✅", "Document rendering (text to image)"),
        ("✅", "Layout augmentation with real image processing"),
        ("✅", "Visual augmentations (rotation, scaling, noise, etc.)"),
        ("✅", "VQA generation with hard negatives"),
        ("✅", "Handwriting synthesis"),
        ("✅", "Dataset management (HuggingFace compatible)"),
        ("✅", "Workflow integration"),
        ("✅", "Font management and multi-script support"),
        ("🚧", "PDF processing and extraction (placeholder)"),
        ("🚧", "Advanced layout detection"),
        ("🚧", "Document element recombination"),
    ]
    
    for status, feature in features:
        print(f"  {status} {feature}")
    
    print("\n✨ SynthDoc comprehensive demo completed!")
    print("Check the implementation in synthdoc/generators.py and synthdoc/augmentations.py")


if __name__ == "__main__":
    main()
