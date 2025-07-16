#!/usr/bin/env python3
"""
Advanced Augmentation Demo

This script demonstrates SynthDoc's advanced augmentation capabilities,
showcasing sophisticated document transformation techniques for creating
robust training datasets.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Any
from synthdoc import SynthDoc, AugmentationType, LayoutType, augment_layouts
from synthdoc.augmentations import Augmentor


def main():
    """Demonstrate advanced augmentation techniques."""
    print("🎨 SynthDoc Advanced Augmentation Demo")
    print("=" * 50)
    
    # Initialize SynthDoc
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    output_dir = "./advanced_augmentation_output"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    if api_key:
        print("✅ Using LLM for content generation")
        synth = SynthDoc(
            output_dir=output_dir,
            llm_model="gpt-4o-mini",
            api_key=api_key
        )
    else:
        print("⚠️  Using template content")
        synth = SynthDoc(output_dir=output_dir)

    # Initialize standalone augmentor for detailed control
    augmentor = Augmentor()
    
    # Demo 1: Generate Base Documents for Augmentation
    print("\n📄 Demo 1: Generating Base Documents")
    print("-" * 40)
    
    # Generate diverse base documents
    base_documents = []
    document_types = [
        {"lang": "en", "prompt": "Generate a technical research paper abstract about machine learning"},
        {"lang": "en", "prompt": "Create a business report with tables and data analysis"},
        {"lang": "hi", "prompt": "भारतीय प्रौद्योगिकी के बारे में एक लेख लिखें"},
        {"lang": "zh", "prompt": "创建一份关于人工智能的技术文档"},
    ]
    
    for i, doc_type in enumerate(document_types):
        print(f"\n📝 Generating document {i+1}: {doc_type['lang'].upper()}")
        
        try:
            docs = synth.generate_raw_docs(
                language=doc_type["lang"],
                num_pages=1,
                prompt=doc_type["prompt"]
            )
            
            base_documents.extend(docs)
            print(f"  ✅ Generated base document")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📊 Total base documents: {len(base_documents)}")

    # Demo 2: Individual Augmentation Techniques
    print("\n🔧 Demo 2: Individual Augmentation Techniques")
    print("-" * 40)
    
    if base_documents:
        test_document = base_documents[0]
        
        # Get available augmentations
        available_augs = augmentor.get_available_augmentations()
        print(f"\n🛠️  Available augmentations: {', '.join(available_augs)}")
        
        # Test each augmentation individually
        individual_results = {}
        
        for aug_type in available_augs[:6]:  # Test first 6 augmentations
            print(f"\n🎯 Testing {aug_type} augmentation...")
            
            try:
                augmented_docs = synth.augment_layout(
                    documents=[test_document],
                    languages=["en"],
                    augmentations=[aug_type],
                    fonts=["Arial"]
                )
                
                num_results = len(augmented_docs.get("images", []))
                individual_results[aug_type] = num_results
                print(f"  ✅ Generated {num_results} variations")
                
            except Exception as e:
                print(f"  ❌ Error with {aug_type}: {e}")
                individual_results[aug_type] = 0
        
        print(f"\n📈 Individual Augmentation Results:")
        for aug_type, count in individual_results.items():
            print(f"  {aug_type}: {count} variations")

    # Demo 3: Compound Augmentation Pipelines
    print("\n🔗 Demo 3: Compound Augmentation Pipelines")
    print("-" * 40)
    
    # Define sophisticated augmentation pipelines
    augmentation_pipelines = [
        {
            "name": "Document Scanning Simulation",
            "augmentations": [AugmentationType.ROTATION, AugmentationType.NOISE, AugmentationType.BLUR],
            "description": "Simulates scanned document artifacts"
        },
        {
            "name": "Mobile Capture Simulation", 
            "augmentations": [AugmentationType.ROTATION, AugmentationType.SCALING, AugmentationType.COLOR_SHIFT],
            "description": "Simulates mobile phone document capture"
        },
        {
            "name": "Degradation Pipeline",
            "augmentations": [AugmentationType.NOISE, AugmentationType.BLUR, AugmentationType.COLOR_SHIFT],
            "description": "Simulates document aging and degradation"
        },
        {
            "name": "Geometric Transformation",
            "augmentations": [AugmentationType.ROTATION, AugmentationType.SCALING, AugmentationType.CROPPING],
            "description": "Various geometric transformations"
        }
    ]
    
    pipeline_results = {}
    
    for pipeline in augmentation_pipelines:
        name = pipeline["name"]
        augs = pipeline["augmentations"]
        desc = pipeline["description"]
        
        print(f"\n🔄 Running {name} pipeline...")
        print(f"   Description: {desc}")
        print(f"   Augmentations: {[aug.value for aug in augs]}")
        
        try:
            if base_documents:
                augmented_docs = synth.augment_layout(
                    documents=base_documents[:2],  # Use first 2 base docs
                    languages=["en", "hi"],
                    augmentations=[aug.value for aug in augs],
                    fonts=["Arial", "Times New Roman"]
                )
                
                num_results = len(augmented_docs.get("images", []))
                pipeline_results[name] = num_results
                print(f"   ✅ Generated {num_results} augmented documents")
            
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            pipeline_results[name] = 0

    # Demo 4: Multi-Font Augmentation
    print("\n🔤 Demo 4: Multi-Font Augmentation")
    print("-" * 40)
    
    # Font combinations for different effects
    font_combinations = [
        {
            "name": "Professional Documents",
            "fonts": ["Arial", "Times New Roman", "Calibri"],
            "description": "Standard business/academic fonts"
        },
        {
            "name": "Design Variety",
            "fonts": ["Arial", "Georgia", "Verdana"],
            "description": "Mix of sans-serif and serif fonts"
        },
        {
            "name": "Accessibility Focus",
            "fonts": ["Arial", "Tahoma", "Trebuchet MS"],
            "description": "High-readability fonts"
        }
    ]
    
    font_results = {}
    
    for font_combo in font_combinations:
        name = font_combo["name"]
        fonts = font_combo["fonts"]
        desc = font_combo["description"]
        
        print(f"\n🔠 Testing {name}...")
        print(f"   Fonts: {', '.join(fonts)}")
        print(f"   Purpose: {desc}")
        
        try:
            if base_documents:
                font_augmented = synth.augment_layout(
                    documents=base_documents[:1],  # Use first base doc
                    languages=["en"],
                    fonts=fonts,
                    augmentations=["rotation"]  # Minimal augmentation to focus on fonts
                )
                
                num_results = len(font_augmented.get("images", []))
                font_results[name] = num_results
                print(f"   ✅ Generated {num_results} font variations")
            
        except Exception as e:
            print(f"   ❌ Font augmentation failed: {e}")
            font_results[name] = 0

    # Demo 5: Layout Template Variations
    print("\n📐 Demo 5: Layout Template Variations")
    print("-" * 40)
    
    # Different layout styles
    layout_styles = [
        {
            "name": "Academic Paper",
            "description": "Single column, formal layout",
            "augmentations": ["rotation"],
            "fonts": ["Times New Roman", "Arial"]
        },
        {
            "name": "Newsletter Format", 
            "description": "Multi-column layout with varied spacing",
            "augmentations": ["scaling", "rotation"],
            "fonts": ["Arial", "Calibri"]
        },
        {
            "name": "Technical Manual",
            "description": "Structured layout with emphasis on readability",
            "augmentations": ["noise", "rotation"],
            "fonts": ["Arial", "Tahoma"]
        }
    ]
    
    layout_results = {}
    
    for layout in layout_styles:
        name = layout["name"]
        desc = layout["description"]
        augs = layout["augmentations"]
        fonts = layout["fonts"]
        
        print(f"\n📋 Creating {name} layout...")
        print(f"   Style: {desc}")
        print(f"   Augmentations: {', '.join(augs)}")
        
        try:
            if base_documents:
                layout_docs = synth.augment_layout(
                    documents=base_documents[:1],
                    languages=["en"],
                    fonts=fonts,
                    augmentations=augs
                )
                
                num_results = len(layout_docs.get("images", []))
                layout_results[name] = num_results
                print(f"   ✅ Generated {num_results} layout variations")
            
        except Exception as e:
            print(f"   ❌ Layout generation failed: {e}")
            layout_results[name] = 0

    # Demo 6: Quality and Degradation Simulation
    print("\n🎚️  Demo 6: Quality and Degradation Simulation")
    print("-" * 40)
    
    quality_simulations = [
        {
            "name": "High Quality Scan",
            "augmentations": ["rotation"],  # Minimal artifacts
            "intensity": "low",
            "description": "Clean, high-resolution document capture"
        },
        {
            "name": "Standard Office Scan",
            "augmentations": ["rotation", "noise"],
            "intensity": "medium", 
            "description": "Typical office scanner quality"
        },
        {
            "name": "Poor Quality Capture",
            "augmentations": ["rotation", "noise", "blur", "color_shift"],
            "intensity": "high",
            "description": "Low-quality mobile capture or degraded document"
        }
    ]
    
    quality_results = {}
    
    for simulation in quality_simulations:
        name = simulation["name"]
        augs = simulation["augmentations"]
        intensity = simulation["intensity"]
        desc = simulation["description"]
        
        print(f"\n📊 Simulating {name}...")
        print(f"   Quality: {intensity}")
        print(f"   Description: {desc}")
        print(f"   Augmentations: {', '.join(augs)}")
        
        try:
            if base_documents:
                quality_docs = synth.augment_layout(
                    documents=base_documents[:1],
                    languages=["en"],
                    augmentations=augs,
                    fonts=["Arial"]
                )
                
                num_results = len(quality_docs.get("images", []))
                quality_results[name] = num_results
                print(f"   ✅ Generated {num_results} quality variations")
            
        except Exception as e:
            print(f"   ❌ Quality simulation failed: {e}")
            quality_results[name] = 0

    # Demo 7: Batch Processing for Large Datasets
    print("\n📦 Demo 7: Batch Processing Demo")
    print("-" * 40)
    
    print(f"🔄 Processing {len(base_documents)} documents in batch...")
    
    if base_documents:
        try:
            # Large batch with multiple augmentations
            batch_result = synth.augment_layout(
                documents=base_documents,
                languages=["en", "hi", "zh"],
                fonts=["Arial", "Times New Roman"],
                augmentations=["rotation", "scaling", "noise"]
            )
            
            total_generated = len(batch_result.get("images", []))
            print(f"✅ Batch processing completed: {total_generated} total variations")
            
            # Calculate augmentation factor
            augmentation_factor = total_generated / len(base_documents) if base_documents else 0
            print(f"📈 Augmentation factor: {augmentation_factor:.1f}x")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

    # Demo 8: Custom Augmentation Intensities
    print("\n⚡ Demo 8: Custom Augmentation Intensities")
    print("-" * 40)
    
    intensity_levels = ["low", "medium", "high"]
    intensity_results = {}
    
    for intensity in intensity_levels:
        print(f"\n🎛️  Testing {intensity} intensity augmentation...")
        
        try:
            if base_documents:
                # Simulate different intensities by varying augmentation combinations
                if intensity == "low":
                    test_augs = ["rotation"]
                elif intensity == "medium":
                    test_augs = ["rotation", "scaling"]
                else:  # high
                    test_augs = ["rotation", "scaling", "noise", "blur"]
                
                intensity_docs = synth.augment_layout(
                    documents=base_documents[:1],
                    languages=["en"],
                    augmentations=test_augs,
                    fonts=["Arial"]
                )
                
                num_results = len(intensity_docs.get("images", []))
                intensity_results[intensity] = num_results
                print(f"   ✅ {intensity.capitalize()} intensity: {num_results} variations")
            
        except Exception as e:
            print(f"   ❌ {intensity} intensity failed: {e}")
            intensity_results[intensity] = 0

    # Summary Report
    print("\n📊 Advanced Augmentation Summary Report")
    print("=" * 50)
    
    print(f"📄 Base documents generated: {len(base_documents)}")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🔧 Individual Augmentation Results:")
    for aug_type, count in individual_results.items():
        print(f"  {aug_type}: {count} variations")
    
    print(f"\n🔗 Pipeline Results:")
    for pipeline_name, count in pipeline_results.items():
        print(f"  {pipeline_name}: {count} documents")
    
    print(f"\n🔤 Font Combination Results:")
    for font_combo, count in font_results.items():
        print(f"  {font_combo}: {count} variations")
    
    print(f"\n📐 Layout Template Results:")
    for layout_name, count in layout_results.items():
        print(f"  {layout_name}: {count} layouts")
    
    print(f"\n🎚️  Quality Simulation Results:")
    for quality_name, count in quality_results.items():
        print(f"  {quality_name}: {count} simulations")
    
    print(f"\n⚡ Intensity Level Results:")
    for intensity, count in intensity_results.items():
        print(f"  {intensity.capitalize()}: {count} variations")
    
    print(f"\n✨ Advanced Features Demonstrated:")
    features = [
        "✅ Individual augmentation control",
        "✅ Compound augmentation pipelines", 
        "✅ Multi-font transformation",
        "✅ Layout template variations",
        "✅ Quality degradation simulation",
        "✅ Batch processing capabilities",
        "✅ Custom intensity levels",
        "✅ Cross-language augmentation"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n💡 Best Practices for Advanced Augmentation:")
    practices = [
        "Combine multiple augmentation types for realistic variations",
        "Use different font combinations to improve model robustness",
        "Simulate various capture conditions (scan, mobile, etc.)",
        "Apply appropriate intensity levels based on use case",
        "Test augmentations on diverse document types",
        "Balance augmentation diversity with computational resources",
        "Consider domain-specific augmentation requirements"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"  {i}. {practice}")
    
    print(f"\n🎯 Use Cases for Advanced Augmentation:")
    use_cases = [
        "Document OCR model training with various quality conditions",
        "Layout analysis system robustness testing", 
        "Multi-language document understanding",
        "Document classification with style variations",
        "Handwriting recognition system training",
        "Document fraud detection (authentic vs. synthetic)",
        "Cross-domain document adaptation"
    ]
    
    for i, use_case in enumerate(use_cases, 1):
        print(f"  {i}. {use_case}")


if __name__ == "__main__":
    main() 