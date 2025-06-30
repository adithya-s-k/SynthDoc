#!/usr/bin/env python3
"""
Script to showcase the results of the multilingual SynthDoc pipeline demo.

This script displays generated content, statistics, and provides options for
HuggingFace dataset upload.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add SynthDoc to path
sys.path.insert(0, str(Path(__file__).parent))

def find_latest_demo_output():
    """Find the most recent multilingual demo output directory."""
    current_dir = Path(".")
    demo_dirs = list(current_dir.glob("multilingual_demo_*"))
    
    if not demo_dirs:
        print("❌ No multilingual demo output directories found")
        return None
    
    # Sort by name (which includes timestamp)
    latest_dir = sorted(demo_dirs)[-1]
    return latest_dir

def load_summary(demo_dir: Path) -> Dict[str, Any]:
    """Load the pipeline summary from the demo directory."""
    summary_file = demo_dir / "pipeline_summary.json"
    
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return {}
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def show_directory_structure(demo_dir: Path):
    """Display the directory structure of generated content."""
    print(f"\n📁 Directory Structure: {demo_dir}")
    print("-" * 50)
    
    for item in sorted(demo_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(demo_dir)
            size = item.stat().st_size
            print(f"  📄 {rel_path} ({size:,} bytes)")
        elif item.is_dir() and item != demo_dir:
            rel_path = item.relative_to(demo_dir)
            file_count = len(list(item.glob("*")))
            print(f"  📂 {rel_path}/ ({file_count} items)")

def show_sample_content(demo_dir: Path, summary: Dict[str, Any]):
    """Show sample content from each workflow."""
    print(f"\n🎨 Sample Generated Content")
    print("=" * 50)
    
    # Show raw documents
    raw_docs_dir = demo_dir / "raw_documents"
    if raw_docs_dir.exists():
        print(f"\n📝 Raw Documents:")
        for lang_dir in sorted(raw_docs_dir.iterdir()):
            if lang_dir.is_dir():
                images = list(lang_dir.glob("*.png"))
                print(f"  🌍 {lang_dir.name.upper()}: {len(images)} documents")
    
    # Show handwriting samples
    handwriting_dir = demo_dir / "handwriting"
    if handwriting_dir.exists():
        print(f"\n🖋️  Handwriting Samples:")
        total_handwriting = 0
        for lang_dir in sorted(handwriting_dir.iterdir()):
            if lang_dir.is_dir():
                styles = list(lang_dir.iterdir())
                style_count = 0
                for style_dir in styles:
                    if style_dir.is_dir():
                        images = list(style_dir.glob("*.png"))
                        style_count += len(images)
                print(f"  🌍 {lang_dir.name.upper()}: {style_count} samples")
                total_handwriting += style_count
        print(f"  📊 Total handwriting samples: {total_handwriting}")
    
    # Show VQA data
    vqa_dir = demo_dir / "vqa"
    if vqa_dir.exists():
        print(f"\n❓ VQA Datasets:")
        for lang_dir in sorted(vqa_dir.iterdir()):
            if lang_dir.is_dir():
                vqa_file = lang_dir / "vqa_data.json"
                if vqa_file.exists():
                    with open(vqa_file, 'r', encoding='utf-8') as f:
                        vqa_data = json.load(f)
                    print(f"  🌍 {lang_dir.name.upper()}: {len(vqa_data)} Q&A pairs")
                    
                    # Show a sample question
                    if vqa_data:
                        sample = vqa_data[0]
                        print(f"    📋 Sample Q: {sample.get('question', 'N/A')[:100]}...")
                        print(f"    📋 Sample A: {sample.get('answer', 'N/A')[:100]}...")

def show_hf_datasets(demo_dir: Path, summary: Dict[str, Any]):
    """Show information about created HuggingFace datasets."""
    print(f"\n🤗 HuggingFace Datasets")
    print("=" * 50)
    
    datasets_created = summary.get("datasets_created", [])
    
    for dataset_info in datasets_created:
        name = dataset_info["name"]
        samples = dataset_info["samples"]
        features = dataset_info["features"]
        
        print(f"\n📦 Dataset: {name}")
        print(f"  📊 Samples: {samples:,}")
        print(f"  🔧 Features: {len(features)}")
        print(f"  📁 Local path: {dataset_info['local_path']}")
        
        # Show feature list
        feature_groups = []
        current_group = []
        for feature in features[:10]:  # Show first 10 features
            current_group.append(feature)
            if len(current_group) == 3:
                feature_groups.append(current_group)
                current_group = []
        if current_group:
            feature_groups.append(current_group)
        
        for group in feature_groups:
            print(f"    • {', '.join(group)}")
        
        if len(features) > 10:
            print(f"    ... and {len(features) - 10} more features")

def upload_to_huggingface(demo_dir: Path, summary: Dict[str, Any]):
    """Upload datasets to HuggingFace Hub."""
    print(f"\n🚀 HuggingFace Upload")
    print("=" * 50)
    
    # Check for HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("⚠️  No HuggingFace token found in environment variables")
        print("To upload datasets, set HUGGINGFACE_TOKEN or HF_TOKEN in your .env file")
        return
    
    try:
        from datasets import Dataset
        
        datasets_created = summary.get("datasets_created", [])
        timestamp = summary.get("timestamp", "unknown")
        
        for dataset_info in datasets_created:
            name = dataset_info["name"]
            local_path = dataset_info["local_path"]
            
            print(f"\n📤 Uploading {name}...")
            
            try:
                # Load dataset from disk
                dataset = Dataset.load_from_disk(local_path)
                
                # Create repository name
                repo_name = f"synthdoc-{name}-multilingual-{timestamp}"
                
                # Upload to HuggingFace Hub
                dataset.push_to_hub(
                    repo_name,
                    token=hf_token,
                    private=False  # Set to True for private datasets
                )
                
                print(f"  ✅ Uploaded to: https://huggingface.co/datasets/{repo_name}")
                
            except Exception as e:
                print(f"  ❌ Upload failed: {e}")
        
    except ImportError:
        print("❌ datasets library not installed. Install with: pip install datasets")
    except Exception as e:
        print(f"❌ Upload error: {e}")

def main():
    """Main function to showcase multilingual demo results."""
    print("🌍 SynthDoc Multilingual Demo Results")
    print("=" * 60)
    
    # Find latest demo output
    demo_dir = find_latest_demo_output()
    if not demo_dir:
        return
    
    print(f"📁 Demo directory: {demo_dir}")
    
    # Load summary
    summary = load_summary(demo_dir)
    if not summary:
        return
    
    # Show high-level statistics
    print(f"\n📊 Pipeline Statistics")
    print("-" * 30)
    print(f"  🕐 Timestamp: {summary.get('timestamp', 'Unknown')}")
    print(f"  🌍 Languages tested: {len(summary.get('languages_tested', []))}")
    print(f"  ⚙️  Workflows completed: {len(summary.get('workflows_completed', []))}")
    print(f"  📈 Total samples: {summary.get('total_samples_generated', 0):,}")
    print(f"  📦 Datasets created: {len(summary.get('datasets_created', []))}")
    
    languages = summary.get('languages_tested', [])
    workflows = summary.get('workflows_completed', [])
    
    print(f"\n🌐 Languages: {', '.join(lang.upper() for lang in languages)}")
    print(f"🔧 Workflows: {', '.join(workflows)}")
    
    # Show directory structure
    show_directory_structure(demo_dir)
    
    # Show sample content
    show_sample_content(demo_dir, summary)
    
    # Show HuggingFace dataset info
    show_hf_datasets(demo_dir, summary)
    
    # Offer HuggingFace upload
    print(f"\n🤗 HuggingFace Upload Options")
    print("-" * 40)
    
    response = input("Would you like to upload datasets to HuggingFace? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        upload_to_huggingface(demo_dir, summary)
    else:
        print("⏭️  Skipping HuggingFace upload")
    
    print(f"\n🎉 Demo Results Summary Complete!")
    print(f"📁 All files available in: {demo_dir}")

if __name__ == "__main__":
    main() 