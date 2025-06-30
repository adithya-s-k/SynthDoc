#!/usr/bin/env python3
"""
Upload SynthDoc multilingual datasets to HuggingFace Hub.

This script uploads the generated datasets to make them publicly available.
"""

import os
import sys
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

def upload_datasets():
    """Upload all generated datasets to HuggingFace Hub."""
    
    # Check for HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("🚨 HuggingFace token required!")
        print("\nTo upload datasets:")
        print("1. Get a token from: https://huggingface.co/settings/tokens")
        print("2. Add to your .env file: HUGGINGFACE_TOKEN=your_token_here")
        print("3. Or set environment variable: export HUGGINGFACE_TOKEN=your_token")
        return False
    
    # Find the demo directory
    demo_dirs = list(Path(".").glob("multilingual_demo_*"))
    if not demo_dirs:
        print("❌ No demo output directories found")
        return False
    
    latest_demo = sorted(demo_dirs)[-1]
    datasets_dir = latest_demo / "hf_datasets"
    
    if not datasets_dir.exists():
        print(f"❌ No datasets directory found: {datasets_dir}")
        return False
    
    print(f"🚀 Uploading SynthDoc Multilingual Datasets")
    print(f"📁 From: {datasets_dir}")
    print("=" * 60)
    
    # Dataset descriptions
    descriptions = {
        "raw_documents": "SynthDoc generated multilingual documents using latest LLMs (GPT-4o-mini, Claude-3.5-Sonnet). Contains technical documents in English, Spanish, and Chinese with comprehensive layout annotations.",
        "handwriting": "SynthDoc multilingual handwriting samples in 6 languages (EN, ES, ZH, HI, AR, FR) with various styles (print, cursive, mixed) and paper templates (lined, grid, blank).",
        "vqa": "SynthDoc generated Visual Question Answering dataset with intelligent questions, answers, and hard negatives for training document understanding models."
    }
    
    success_count = 0
    
    for dataset_name in ["raw_documents", "handwriting", "vqa"]:
        dataset_path = datasets_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  Skipping {dataset_name} - not found")
            continue
        
        try:
            print(f"\n📤 Uploading {dataset_name}...")
            
            # Load dataset
            dataset = Dataset.load_from_disk(str(dataset_path))
            
            # Create repository name
            repo_name = f"synthdoc-{dataset_name}-multilingual"
            
            # Upload to HuggingFace Hub
            dataset.push_to_hub(
                repo_name,
                token=hf_token,
                private=False,  # Public dataset
                commit_message=f"Add SynthDoc multilingual {dataset_name} dataset"
            )
            
            print(f"  ✅ Success! Dataset: https://huggingface.co/datasets/{repo_name}")
            print(f"  📊 Samples: {len(dataset):,}")
            print(f"  🔧 Features: {list(dataset.features.keys())}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n🎉 Upload Complete!")
    print(f"✅ Successfully uploaded {success_count}/3 datasets")
    
    if success_count > 0:
        print(f"\n🔗 View your datasets at: https://huggingface.co/datasets")
        print(f"🏷️  Search for: synthdoc-multilingual")
    
    return success_count > 0

if __name__ == "__main__":
    success = upload_datasets()
    sys.exit(0 if success else 1) 