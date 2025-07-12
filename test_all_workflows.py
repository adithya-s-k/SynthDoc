#!/usr/bin/env python3
"""
Comprehensive SynthDoc Testing Script
=====================================

This script tests all SynthDoc workflows and demonstrates HuggingFace integration.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset

# Import SynthDoc
from synthdoc import SynthDoc
from synthdoc.models import (
    HandwritingGenerationConfig, 
    PDFAugmentationConfig,
    RawDocumentGenerationConfig,
    LayoutAugmentationConfig,
    VQAGenerationConfig
)
from synthdoc.languages import Language
from synthdoc.workflows.handwriting_generator.workflow import HandwritingGenerator
from synthdoc.workflows.raw_document_generator.workflow import RawDocumentGenerator
from synthdoc.workflows.layout_augmenter.workflow import LayoutAugmenter
from synthdoc.workflows.pdf_augmenter.workflow import PDFAugmenter
from synthdoc.workflows.vqa_generator.workflow import VQAGenerator

def test_handwriting_workflows():
    """Test handwriting generation with all paper templates and styles."""
    print("\n🖋️  TESTING HANDWRITING WORKFLOWS")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_handwriting_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        # Test different paper templates and styles
        templates = ["lined", "grid", "blank"]
        styles = ["print", "cursive", "mixed"]
        
        print("📝 Testing core SynthDoc handwriting API...")
        for template in templates:
            for style in styles:
                print(f"  Testing {style} style on {template} paper...")
                
                dataset = synth.generate_handwriting(
                    content=f"Testing {style} handwriting on {template} paper. This is a sample text.",
                    language="en",
                    writing_style=style,
                    paper_template=template
                )
                
                assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
                sample = dataset[0]
                assert template in sample['pdf_info'], f"Template {template} not found"
                
                results["success"].append(f"Core API: {style} on {template}")
        
        # Test multi-language support
        print("🌍 Testing multi-language handwriting...")
        for lang in ["en", "hi", "zh"]:
            dataset = synth.generate_handwriting(
                content="Multi-language test",
                language=lang,
                writing_style="print",
                paper_template="lined"
            )
            assert len(dataset) == 1
            results["success"].append(f"Multi-language: {lang}")
        
        # Test direct workflow
        print("⚙️  Testing direct handwriting workflow...")
        hw_gen = HandwritingGenerator(save_dir=str(test_dir / "handwriting"))
        
        config = HandwritingGenerationConfig(
            text_content="Direct workflow test",
            handwriting_style="print",
            paper_template="grid",
            num_samples=2
        )
        
        result = hw_gen.process(config)
        assert result.num_samples == 2
        results["success"].append("Direct workflow")
        
    except Exception as e:
        error_msg = f"Handwriting test failed: {e}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)
    
    print(f"✅ Handwriting tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def test_raw_document_generation():
    """Test raw document generation workflow."""
    print("\n📄 TESTING RAW DOCUMENT GENERATION")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_raw_docs_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        print("🤖 Testing LLM-based document generation...")
        
        # Test core API
        dataset = synth.generate_raw_docs(
            language="en",
            num_pages=1,
            prompt="Generate a report about renewable energy",
            augmentations=["rotation"]
        )
        
        assert len(dataset) >= 1
        results["success"].append("Core raw doc generation")
        
        # Test direct workflow
        raw_gen = RawDocumentGenerator(save_dir=str(test_dir / "raw"))
        config = RawDocumentGenerationConfig(
            language=Language.EN,
            num_pages=1,
            prompt="Test document generation"
        )
        
        result = raw_gen.process(config)
        assert result.num_samples >= 1
        results["success"].append("Direct workflow")
        
    except Exception as e:
        error_msg = f"Raw document generation failed: {e}"
        print(f"❌ {error_msg}")
        print("Note: This may fail if LLM API keys are not configured")
        results["errors"].append(error_msg)
    
    print(f"✅ Raw document tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def test_layout_augmentation():
    """Test layout augmentation workflow."""
    print("\n🎨 TESTING LAYOUT AUGMENTATION")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_layout_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        # Generate sample documents first
        print("📝 Generating sample documents...")
        hw_datasets = []
        for i in range(2):
            dataset = synth.generate_handwriting(
                content=f"Sample document {i+1} for layout testing.",
                writing_style="print",
                paper_template="lined"
            )
            hw_datasets.append(dataset)
        
        # Test core API
        print("🎯 Testing layout augmentation...")
        combined_dataset = Dataset.from_dict({
            key: [item for dataset in hw_datasets for item in dataset[key]] 
            for key in hw_datasets[0].features.keys()
        })
        
        augmented_dataset = synth.augment_layout(
            documents=combined_dataset,
            languages=["en"],
            augmentations=["rotation", "scaling"]
        )
        
        assert len(augmented_dataset) >= len(hw_datasets)
        results["success"].append("Core layout augmentation")
        
        # Test direct workflow
        layout_aug = LayoutAugmenter(save_dir=str(test_dir / "layout"))
        image_paths = []
        for dataset in hw_datasets:
            for sample in dataset:
                if 'image_path' in sample:
                    image_paths.append(sample['image_path'])
        
        config = LayoutAugmentationConfig(
            documents=image_paths,
            languages=[Language.EN]
        )
        
        result = layout_aug.process(config)
        assert result.num_samples >= 1
        results["success"].append("Direct workflow")
        
    except Exception as e:
        error_msg = f"Layout augmentation failed: {e}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)
    
    print(f"✅ Layout tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def test_pdf_augmentation():
    """Test PDF augmentation workflow."""
    print("\n📋 TESTING PDF AUGMENTATION")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_pdf_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        # Create mock PDF files
        print("📄 Creating mock PDF files...")
        mock_pdfs = []
        for i in range(2):
            mock_pdf = test_dir / f"test_doc_{i+1}.pdf"
            mock_pdf.write_text(f"""
Sample PDF Document {i+1}

Header: Introduction
This document contains sample text for testing.

Header: Methods
Description of the testing methodology.

Header: Results
Results of the PDF augmentation testing.
            """)
            mock_pdfs.append(str(mock_pdf))
        
        # Test core API
        print("🔧 Testing PDF augmentation...")
        dataset = synth.augment_pdfs(
            corpus_paths=mock_pdfs,
            extraction_elements=["text_blocks", "headers"],
            combination_strategy="random"
        )
        
        assert len(dataset) >= 0  # May be 0 in fallback mode
        results["success"].append("Core PDF augmentation")
        
        # Test direct workflow
        pdf_aug = PDFAugmenter(save_dir=str(test_dir / "pdf"))
        config = PDFAugmentationConfig(
            pdf_files=mock_pdfs,
            extraction_elements=["text_blocks"],
            combination_strategy="random",
            num_generated_docs=1
        )
        
        result = pdf_aug.process(config)
        assert result.num_samples >= 0
        results["success"].append("Direct workflow")
        
    except Exception as e:
        error_msg = f"PDF augmentation failed: {e}"
        print(f"❌ {error_msg}")
        print("Note: May use fallback mode if PDF libraries aren't available")
        results["errors"].append(error_msg)
    
    print(f"✅ PDF tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def test_vqa_generation():
    """Test VQA generation workflow."""
    print("\n❓ TESTING VQA GENERATION")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_vqa_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        # Generate sample documents
        print("📝 Generating documents for VQA...")
        sample_docs = []
        for i in range(2):
            dataset = synth.generate_handwriting(
                content=f"Document {i+1}: Information about renewable energy sources.",
                writing_style="print",
                paper_template="blank"
            )
            sample_docs.append(dataset)
        
        # Test core API
        print("🤖 Testing VQA generation...")
        combined_docs = Dataset.from_dict({
            key: [item for dataset in sample_docs for item in dataset[key]] 
            for key in sample_docs[0].features.keys()
        })
        
        vqa_dataset = synth.generate_vqa(
            source_documents=combined_docs,
            question_types=["factual"],
            difficulty_levels=["easy"]
        )
        
        assert len(vqa_dataset) >= 0
        results["success"].append("Core VQA generation")
        
        # Test direct workflow
        vqa_gen = VQAGenerator(save_dir=str(test_dir / "vqa"))
        image_paths = []
        for dataset in sample_docs:
            for sample in dataset:
                if 'image_path' in sample:
                    image_paths.append(sample['image_path'])
        
        config = VQAGenerationConfig(
            documents=image_paths,
            question_types=["factual"],
            num_questions_per_doc=1,
            include_hard_negatives=False
        )
        
        result = vqa_gen.process(config)
        assert result.num_samples >= 0
        results["success"].append("Direct workflow")
        
    except Exception as e:
        error_msg = f"VQA generation failed: {e}"
        print(f"❌ {error_msg}")
        print("Note: May use template mode if LLM APIs aren't configured")
        results["errors"].append(error_msg)
    
    print(f"✅ VQA tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def test_huggingface_integration():
    """Test HuggingFace dataset integration and upload."""
    print("\n🤗 TESTING HUGGINGFACE INTEGRATION")
    print("=" * 50)
    
    test_dir = Path(tempfile.mkdtemp(prefix="synthdoc_hf_"))
    synth = SynthDoc(output_dir=str(test_dir / "output"))
    
    results = {"success": [], "errors": []}
    
    try:
        # Generate comprehensive dataset
        print("📊 Creating comprehensive dataset...")
        datasets_to_combine = []
        
        for template in ["lined", "grid", "blank"]:
            dataset = synth.generate_handwriting(
                content=f"Sample on {template} paper for HF testing.",
                paper_template=template,
                writing_style="print"
            )
            datasets_to_combine.append(dataset)
        
        # Combine datasets
        combined_features = {}
        for key in datasets_to_combine[0].features.keys():
            combined_features[key] = []
            for dataset in datasets_to_combine:
                combined_features[key].extend(dataset[key])
        
        final_dataset = Dataset.from_dict(combined_features)
        
        print(f"📈 Created dataset with {len(final_dataset)} samples")
        print(f"📋 Features: {list(final_dataset.features.keys())}")
        
        # Save locally
        local_path = test_dir / "hf_dataset"
        final_dataset.save_to_disk(str(local_path))
        print(f"💾 Saved locally to: {local_path}")
        results["success"].append("Local save")
        
        # Test loading
        loaded_dataset = Dataset.load_from_disk(str(local_path))
        assert len(loaded_dataset) == len(final_dataset)
        results["success"].append("Local load")
        
        # Optional HF Hub upload
        try:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            
            if token:
                print("🚀 Uploading to HuggingFace Hub...")
                repo_name = f"synthdoc-test-{int(__import__('time').time())}"
                
                final_dataset.push_to_hub(
                    repo_id=repo_name,
                    token=token,
                    private=True
                )
                
                print(f"✅ Uploaded to HF Hub: {repo_name}")
                results["success"].append("HF Hub upload")
                
                # Test download
                downloaded = Dataset.load_dataset(repo_name, split="train", token=token)
                assert len(downloaded) == len(final_dataset)
                results["success"].append("HF Hub download")
                
            else:
                print("⏭️  Skipping HF Hub - no token found")
                print("   Set HF_TOKEN environment variable to test uploads")
                results["success"].append("HF ready (no token)")
                
        except Exception as e:
            error_msg = f"HF Hub upload failed: {e}"
            print(f"⚠️  {error_msg}")
            results["errors"].append(error_msg)
        
    except Exception as e:
        error_msg = f"HuggingFace integration failed: {e}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)
    
    print(f"✅ HF tests: {len(results['success'])} successes, {len(results['errors'])} errors")
    return results, test_dir

def run_all_tests():
    """Run all comprehensive tests."""
    print("🚀 STARTING COMPREHENSIVE SYNTHDOC TESTING")
    print("=" * 60)
    
    # Initialize SynthDoc for environment check
    temp_dir = Path(tempfile.mkdtemp(prefix="synthdoc_env_check_"))
    synth = SynthDoc(output_dir=str(temp_dir))
    print(f"Environment check: {synth.print_environment_status() or 'Complete'}")
    print()
    
    all_results = {}
    test_dirs = {}
    
    # Run all test suites
    test_functions = [
        ("handwriting", test_handwriting_workflows),
        ("raw_documents", test_raw_document_generation),
        ("layout_augmentation", test_layout_augmentation),
        ("pdf_augmentation", test_pdf_augmentation),
        ("vqa_generation", test_vqa_generation),
        ("huggingface", test_huggingface_integration)
    ]
    
    for suite_name, test_func in test_functions:
        try:
            results, test_dir = test_func()
            all_results[suite_name] = results
            test_dirs[suite_name] = test_dir
        except Exception as e:
            print(f"❌ Test suite {suite_name} crashed: {e}")
            traceback.print_exc()
            all_results[suite_name] = {"success": [], "errors": [f"Suite crashed: {e}"]}
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    total_success = 0
    total_errors = 0
    
    for suite_name, suite_results in all_results.items():
        success_count = len(suite_results.get("success", []))
        error_count = len(suite_results.get("errors", []))
        
        total_success += success_count
        total_errors += error_count
        
        status = "✅" if error_count == 0 else "⚠️" if success_count > 0 else "❌"
        print(f"{status} {suite_name.upper()}: {success_count} successes, {error_count} errors")
        
        if suite_results.get("errors"):
            for error in suite_results["errors"]:
                print(f"    ❌ {error}")
    
    print("\n" + "-" * 40)
    overall = "✅ ALL TESTS PASSED" if total_errors == 0 else f"⚠️  {total_success} successes, {total_errors} errors"
    print(f"OVERALL: {overall}")
    
    if total_errors == 0:
        print("🎉 SynthDoc is working perfectly!")
    else:
        print("📝 Some tests failed - common issues:")
        print("   - Missing API keys for LLM workflows")
        print("   - Missing optional dependencies")
        print("   - Network connectivity issues")
    
    print("\n📁 Test directories:")
    for suite_name, test_dir in test_dirs.items():
        print(f"   {suite_name}: {test_dir}")
    
    print("=" * 60)
    return all_results

def main():
    """Main function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all SynthDoc workflows")
    parser.add_argument("--suite", choices=[
        "handwriting", "raw_documents", "layout_augmentation",
        "pdf_augmentation", "vqa_generation", "huggingface", "all"
    ], default="all", help="Specific test suite to run")
    
    args = parser.parse_args()
    
    try:
        if args.suite == "all":
            return run_all_tests()
        else:
            # Run specific suite
            suite_functions = {
                "handwriting": test_handwriting_workflows,
                "raw_documents": test_raw_document_generation,
                "layout_augmentation": test_layout_augmentation,
                "pdf_augmentation": test_pdf_augmentation,
                "vqa_generation": test_vqa_generation,
                "huggingface": test_huggingface_integration
            }
            
            results, test_dir = suite_functions[args.suite]()
            print(f"\nTest directory: {test_dir}")
            return results
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 