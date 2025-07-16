#!/usr/bin/env python3
"""
Raw Document Generation Example

This example demonstrates how to generate synthetic documents from scratch
using SynthDoc's Raw Document Generation workflow.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthdoc.workflows.raw_document_generator import RawDocumentGenerator
from synthdoc.models import RawDocumentGenerationConfig
from synthdoc.languages import Language


def basic_document_generation():
    """Basic example: Generate a simple English document."""
    print("🔄 Basic Document Generation")
    print("-" * 40)
    
    # Initialize generator
    generator = RawDocumentGenerator(save_dir="basic_documents")
    
    # Create configuration
    config = RawDocumentGenerationConfig(
        language=Language.EN,
        num_pages=2,
        prompt="Write a technical report about artificial intelligence in healthcare",
        include_graphs=True,
        include_tables=True,
        include_ai_images=False
    )
    
    # Generate documents
    result = generator.process(config)
    
    # Display results
    print(f"✅ Generated {result.num_samples} pages")
    print(f"🆔 Document ID: {result.metadata['document_id']}")
    print(f"💰 Cost: ${result.metadata.get('total_cost', 0):.6f}")
    print(f"⏱️  Time: {result.metadata.get('processing_time', 0):.2f}s")
    
    return result


def multilingual_document_generation():
    """Generate documents in multiple languages."""
    print("\n🌍 Multilingual Document Generation")
    print("-" * 40)
    
    # Languages to generate
    languages = [
        (Language.EN, "Write about renewable energy technologies"),
        (Language.HI, "नवीकरणीय ऊर्जा प्रौद्योगिकियों के बारे में लिखें"),
        (Language.ES, "Escriba sobre tecnologías de energía renovable"),
        (Language.FR, "Écrivez sur les technologies d'énergie renouvelable")
    ]
    
    generator = RawDocumentGenerator(save_dir="multilingual_documents")
    
    for language, prompt in languages:
        print(f"\n📝 Generating {language.value} document...")
        
        config = RawDocumentGenerationConfig(
            language=language,
            num_pages=2,
            prompt=prompt,
            include_graphs=False,
            include_tables=True,
            include_ai_images=False
        )
        
        result = generator.process(config)
        print(f"   ✅ Generated: {result.metadata['document_id']}")
        print(f"   💰 Cost: ${result.metadata.get('total_cost', 0):.6f}")


def advanced_layout_generation():
    """Generate documents with different layout types."""
    print("\n📐 Advanced Layout Generation")
    print("-" * 40)
    
    layout_types = [
        "SINGLE_COLUMN",
        "TWO_COLUMN", 
        "THREE_COLUMN"
    ]
    
    generator = RawDocumentGenerator(save_dir="layout_documents")
    
    for layout_type in layout_types:
        print(f"\n📄 Generating {layout_type} layout...")
        
        config = RawDocumentGenerationConfig(
            language=Language.EN,
            num_pages=1,
            prompt=f"Write a comprehensive article about sustainable architecture with {layout_type.lower().replace('_', ' ')} layout",
            include_graphs=True,
            include_tables=True,
            include_ai_images=False,
            layout_type=layout_type
        )
        
        result = generator.process(config)
        print(f"   ✅ Generated: {result.metadata['document_id']}")


def batch_document_generation():
    """Generate multiple documents efficiently."""
    print("\n📚 Batch Document Generation")
    print("-" * 40)
    
    prompts = [
        "Write about climate change and global warming effects",
        "Discuss machine learning applications in medicine",
        "Explain blockchain technology and cryptocurrencies",
        "Analyze the future of space exploration",
        "Review sustainable agriculture practices"
    ]
    
    generator = RawDocumentGenerator(save_dir="batch_documents")
    
    total_cost = 0
    total_pages = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Generating document {i}/{len(prompts)}...")
        print(f"   Topic: {prompt[:50]}...")
        
        config = RawDocumentGenerationConfig(
            language=Language.EN,
            num_pages=1,
            prompt=prompt,
            include_graphs=True,
            include_tables=False,
            include_ai_images=False
        )
        
        result = generator.process(config)
        
        cost = result.metadata.get('total_cost', 0)
        total_cost += cost
        total_pages += result.num_samples
        
        print(f"   ✅ Generated: {result.metadata['document_id']}")
        print(f"   💰 Cost: ${cost:.6f}")
    
    print(f"\n📊 Batch Summary:")
    print(f"   Total pages: {total_pages}")
    print(f"   Total cost: ${total_cost:.6f}")
    print(f"   Average cost per page: ${total_cost/total_pages:.6f}")


def load_and_inspect_dataset():
    """Load and inspect a generated dataset."""
    print("\n🔍 Dataset Inspection")
    print("-" * 40)
    
    try:
        # Load dataset using built-in loader
        dataset = RawDocumentGenerator.load_dataset_from_directory("basic_documents")
        
        if len(dataset) > 0:
            print(f"📊 Dataset loaded: {len(dataset)} samples")
            print(f"📝 Columns: {dataset.column_names}")
            
            # Inspect first sample
            sample = dataset[0]
            print(f"\n📄 First sample:")
            print(f"   File: {sample['file_name']}")
            print(f"   Language: {sample['language']}")
            print(f"   Page: {sample['page_number']}")
            print(f"   Layout: {sample['layout_type']}")
            print(f"   Has graphs: {sample['has_graphs']}")
            print(f"   Has tables: {sample['has_tables']}")
            print(f"   Content preview: {sample['content_preview'][:100]}...")
            
            if 'image' in sample:
                img = sample['image']
                print(f"   Image size: {img.size}")
        else:
            print("❌ No samples found in dataset")
            
    except FileNotFoundError:
        print("❌ Dataset directory not found. Run basic_document_generation() first.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")


def main():
    """Run all examples."""
    print("🚀 SynthDoc Raw Document Generation Examples")
    print("=" * 60)
    
    # Check environment
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found. API calls may fail.")
        print("   Copy env.template to .env and configure your API keys.")
    
    try:
        # Run examples
        basic_document_generation()
        multilingual_document_generation()
        advanced_layout_generation()
        batch_document_generation()
        load_and_inspect_dataset()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📁 Generated datasets:")
        print("   - basic_documents/")
        print("   - multilingual_documents/")
        print("   - layout_documents/")
        print("   - batch_documents/")
        
        print("\n💡 Next steps:")
        print("   - Examine the generated images and metadata")
        print("   - Try different prompts and configurations")
        print("   - Combine with other SynthDoc workflows")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 Check your .env configuration and API keys")


if __name__ == "__main__":
    main() 