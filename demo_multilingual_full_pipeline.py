#!/usr/bin/env python3
"""
Comprehensive Multilingual SynthDoc Pipeline Demo

This script demonstrates SynthDoc's complete functionality:
1. Raw document generation in multiple languages using latest LLMs
2. Handwriting generation with different styles
3. Layout augmentation with various fonts and techniques
4. VQA dataset creation with hard negatives
5. PDF augmentation with element recombination
6. HuggingFace dataset creation and upload

Usage:
    python demo_multilingual_full_pipeline.py

Requirements:
    - API keys configured in .env file
    - HuggingFace token for dataset upload
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json

# Add SynthDoc to path
sys.path.insert(0, str(Path(__file__).parent))

from synthdoc import (
    SynthDoc, 
    create_raw_documents,
    create_handwriting_samples,
    augment_layouts,
    create_vqa_dataset,
    Language,
    AugmentationType,
    print_environment_status
)
from synthdoc.models import (
    RawDocumentGenerationConfig,
    VQAGenerationConfig,
    HandwritingGenerationConfig,
    LayoutAugmentationConfig,
    PDFAugmentationConfig,
    OutputFormat
)
from synthdoc.workflows import (
    RawDocumentGenerator,
    VQAGenerator,
    HandwritingGenerator,
    LayoutAugmenter,
    PDFAugmenter
)

class MultilingualPipelineDemo:
    """Comprehensive demo of all SynthDoc workflows across multiple languages."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the demo with output directory."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or f"./multilingual_demo_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Languages to test
        self.languages = [
            Language.EN,  # English
            Language.ES,  # Spanish
            Language.ZH,  # Chinese
            Language.HI,  # Hindi
            Language.AR,  # Arabic
            Language.FR,  # French
        ]
        
        # Track generated datasets for HuggingFace upload
        self.datasets = {}
        self.results_summary = {
            "timestamp": self.timestamp,
            "languages_tested": [lang.value for lang in self.languages],
            "workflows_completed": [],
            "total_samples_generated": 0,
            "datasets_created": [],
            "hf_uploads": []
        }
        
    def print_header(self, title: str):
        """Print formatted section header."""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
        
    def print_step(self, step: str):
        """Print formatted step information."""
        print(f"\n📋 {step}")
        print("-" * 50)
    
    def check_environment(self):
        """Check and display environment configuration."""
        self.print_header("Environment Configuration Check")
        print_environment_status()
        
        # Check HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            print(f"\n🤗 HuggingFace: ✅ CONFIGURED (token length: {len(hf_token)})")
        else:
            print(f"\n🤗 HuggingFace: ⚠️  No token found - uploads will be skipped")
            
        return bool(hf_token)
    
    def generate_raw_documents(self):
        """Generate raw documents in multiple languages using LLM."""
        self.print_header("Raw Document Generation (Multi-Language)")
        
        # Document prompts for different languages
        prompts = {
            Language.EN: "Generate a technical report about renewable energy systems and their implementation in smart cities",
            Language.ES: "Genera un informe técnico sobre inteligencia artificial y su aplicación en la medicina moderna",
            Language.ZH: "生成一份关于量子计算技术发展及其在未来科技中应用的技术报告",
            Language.HI: "नवीकरणीय ऊर्जा प्रणालियों और स्मार्ट शहरों में उनके कार्यान्वयन के बारे में एक तकनीकी रिपोर्ट तैयार करें",
            Language.AR: "إنشاء تقرير تقني حول الذكاء الاصطناعي وتطبيقاته في الطب الحديث",
            Language.FR: "Générer un rapport technique sur les systèmes d'énergie renouvelable et leur mise en œuvre dans les villes intelligentes"
        }
        
        all_documents = []
        language_docs = {}
        
        for i, language in enumerate(self.languages):
            self.print_step(f"Generating documents in {language.value.upper()}")
            
            try:
                config = RawDocumentGenerationConfig(
                    language=language,
                    num_pages=2,
                    prompt=prompts.get(language, "Generate a technical document"),
                    output_format=OutputFormat.HUGGINGFACE
                )
                
                generator = RawDocumentGenerator()
                result = generator.process(config)
                
                print(f"  ✅ Generated {result.num_samples} documents in {language.value}")
                print(f"  📊 Dataset type: {type(result.dataset).__name__}")
                print(f"  💰 Cost: ${result.metadata.get('cost_summary', {}).get('total_cost', 0):.4f}")
                
                all_documents.extend(result.dataset)
                language_docs[language.value] = result.dataset
                
                # Save language-specific dataset
                lang_output_dir = self.output_dir / "raw_documents" / language.value
                lang_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save individual images and metadata
                for j, sample in enumerate(result.dataset):
                    if 'image' in sample:
                        image_path = lang_output_dir / f"doc_{j+1}.png"
                        sample['image'].save(image_path)
                        print(f"    💾 Saved: {image_path}")
                
            except Exception as e:
                print(f"  ❌ Error generating {language.value} documents: {e}")
                continue
                
        self.datasets['raw_documents'] = {
            'all_languages': all_documents,
            'by_language': language_docs
        }
        
        self.results_summary["workflows_completed"].append("raw_documents")
        self.results_summary["total_samples_generated"] += len(all_documents)
        
        print(f"\n🎉 Raw Document Generation Complete!")
        print(f"   📊 Total documents: {len(all_documents)}")
        print(f"   🌍 Languages: {len(self.languages)}")
        
        return all_documents
    
    def generate_handwriting_samples(self):
        """Generate handwriting samples in multiple styles."""
        self.print_header("Handwriting Generation (Multi-Style)")
        
        handwriting_texts = {
            Language.EN: "The quick brown fox jumps over the lazy dog. Technology advances rapidly in modern times.",
            Language.ES: "La tecnología avanza rápidamente en los tiempos modernos. El desarrollo es constante.",
            Language.ZH: "科技在现代时代快速发展。人工智能改变世界。",
            Language.HI: "प्रौद्योगिकी आधुनिक समय में तेजी से आगे बढ़ रही है।",
            Language.AR: "التكنولوجيا تتقدم بسرعة في العصر الحديث.",
            Language.FR: "La technologie progresse rapidement à l'époque moderne."
        }
        
        writing_styles = ["print", "cursive", "mixed"]
        paper_templates = ["lined", "grid", "blank"]
        
        all_handwriting = []
        
        for language in self.languages:
            for style in writing_styles:
                for paper in paper_templates:
                    self.print_step(f"Generating {style} handwriting on {paper} paper ({language.value})")
                    
                    try:
                        config = HandwritingGenerationConfig(
                            text_content=handwriting_texts[language],
                            language=language,
                            writing_style=style,
                            paper_template=paper,
                            num_samples=1
                        )
                        
                        generator = HandwritingGenerator()
                        result = generator.process(config)
                        
                        print(f"    ✅ Generated {result.num_samples} handwriting samples")
                        
                        all_handwriting.extend(result.dataset)
                        
                        # Save handwriting samples
                        hw_output_dir = self.output_dir / "handwriting" / language.value / f"{style}_{paper}"
                        hw_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for i, sample in enumerate(result.dataset):
                            if 'image' in sample:
                                image_path = hw_output_dir / f"handwriting_{i+1}.png"
                                sample['image'].save(image_path)
                                print(f"      💾 Saved: {image_path}")
                                
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        continue
        
        self.datasets['handwriting'] = all_handwriting
        self.results_summary["workflows_completed"].append("handwriting")
        self.results_summary["total_samples_generated"] += len(all_handwriting)
        
        print(f"\n🖋️  Handwriting Generation Complete!")
        print(f"   📊 Total samples: {len(all_handwriting)}")
        
        return all_handwriting
    
    def generate_layout_augmentation(self, documents: List[Dict]):
        """Apply layout augmentation to generated documents."""
        self.print_header("Layout Augmentation (Multi-Font & Style)")
        
        if not documents:
            print("⚠️  No documents available for layout augmentation")
            return []
        
        # Extract content from documents for layout augmentation
        document_contents = []
        for doc in documents[:10]:  # Limit for demo
            if 'content' in doc:
                document_contents.append(doc['content'])
            elif 'markdown' in doc:
                document_contents.append(doc['markdown'])
        
        if not document_contents:
            print("⚠️  No content found in documents")
            return []
        
        # Different font combinations for variety
        font_sets = [
            ["Arial", "Times New Roman"],
            ["Calibri", "Georgia"],
            ["Helvetica", "Book Antiqua"]
        ]
        
        all_layouts = []
        
        for i, fonts in enumerate(font_sets):
            self.print_step(f"Applying layout augmentation with fonts: {', '.join(fonts)}")
            
            try:
                config = LayoutAugmentationConfig(
                    documents=document_contents[:3],  # Use subset for each font set
                    languages=self.languages[:3],  # Use subset of languages
                    fonts=fonts,
                    augmentations=[
                        AugmentationType.ROTATION,
                        AugmentationType.SCALING,
                        AugmentationType.COLOR_SHIFT
                    ]
                )
                
                generator = LayoutAugmenter()
                result = generator.process(config)
                
                print(f"  ✅ Generated {result.num_samples} layout variations")
                
                all_layouts.extend(result.dataset)
                
                # Save layout samples
                layout_output_dir = self.output_dir / "layout_augmentation" / f"fontset_{i+1}"
                layout_output_dir.mkdir(parents=True, exist_ok=True)
                
                for j, sample in enumerate(result.dataset):
                    if 'image' in sample:
                        image_path = layout_output_dir / f"layout_{j+1}.png"
                        sample['image'].save(image_path)
                        print(f"    💾 Saved: {image_path}")
                
            except Exception as e:
                print(f"  ❌ Error in layout augmentation: {e}")
                continue
        
        self.datasets['layout_augmentation'] = all_layouts
        self.results_summary["workflows_completed"].append("layout_augmentation")
        self.results_summary["total_samples_generated"] += len(all_layouts)
        
        print(f"\n🎨 Layout Augmentation Complete!")
        print(f"   📊 Total variations: {len(all_layouts)}")
        
        return all_layouts
    
    def generate_vqa_dataset(self, documents: List[Dict]):
        """Generate VQA datasets with hard negatives."""
        self.print_header("VQA Dataset Generation (Multi-Language)")
        
        if not documents:
            print("⚠️  No documents available for VQA generation")
            return []
        
        # Organize documents by language for targeted VQA
        docs_by_lang = {}
        for doc in documents:
            lang = doc.get('language', 'en')
            if lang not in docs_by_lang:
                docs_by_lang[lang] = []
            docs_by_lang[lang].append(doc.get('content', doc.get('markdown', '')))
        
        all_vqa = []
        
        for lang, lang_docs in docs_by_lang.items():
            if not lang_docs:
                continue
                
            self.print_step(f"Generating VQA for {lang.upper()} documents")
            
            try:
                config = VQAGenerationConfig(
                    documents=lang_docs[:2],  # Use subset for demo
                    num_questions_per_doc=3,
                    question_types=["factual", "reasoning", "comparative"],
                    difficulty_levels=["easy", "medium", "hard"],
                    hard_negative_ratio=0.3
                )
                
                generator = VQAGenerator()
                result = generator.process(config)
                
                print(f"  ✅ Generated {result.num_samples} VQA pairs")
                print(f"  💰 Cost: ${result.metadata.get('cost_summary', {}).get('total_cost', 0):.4f}")
                
                all_vqa.extend(result.dataset)
                
                # Save VQA dataset
                vqa_output_dir = self.output_dir / "vqa" / lang
                vqa_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save VQA data as JSON
                vqa_data = []
                for sample in result.dataset:
                    # Convert to serializable format
                    serializable_sample = {}
                    for key, value in sample.items():
                        if key == 'image':
                            # Save image and store path
                            image_path = vqa_output_dir / f"vqa_image_{len(vqa_data)}.png"
                            value.save(image_path)
                            serializable_sample['image_path'] = str(image_path)
                        else:
                            serializable_sample[key] = value
                    vqa_data.append(serializable_sample)
                
                # Save VQA metadata
                with open(vqa_output_dir / "vqa_data.json", "w", encoding="utf-8") as f:
                    json.dump(vqa_data, f, indent=2, ensure_ascii=False)
                
                print(f"    💾 Saved VQA data: {vqa_output_dir}")
                
            except Exception as e:
                print(f"  ❌ Error in VQA generation: {e}")
                continue
        
        self.datasets['vqa'] = all_vqa
        self.results_summary["workflows_completed"].append("vqa")
        self.results_summary["total_samples_generated"] += len(all_vqa)
        
        print(f"\n❓ VQA Generation Complete!")
        print(f"   📊 Total VQA pairs: {len(all_vqa)}")
        
        return all_vqa
    
    def create_huggingface_datasets(self):
        """Create and optionally upload HuggingFace datasets."""
        self.print_header("HuggingFace Dataset Creation")
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        for dataset_name, dataset_data in self.datasets.items():
            if not dataset_data:
                continue
                
            self.print_step(f"Creating HuggingFace dataset: {dataset_name}")
            
            try:
                from datasets import Dataset
                
                # Handle different dataset structures
                if isinstance(dataset_data, dict) and 'all_languages' in dataset_data:
                    samples = dataset_data['all_languages']
                else:
                    samples = dataset_data
                
                if not samples:
                    print(f"  ⚠️  No samples found for {dataset_name}")
                    continue
                
                # Convert PIL Images to paths for serialization
                processed_samples = []
                for i, sample in enumerate(samples):
                    processed_sample = {}
                    for key, value in sample.items():
                        if hasattr(value, 'save'):  # PIL Image
                            # Save image
                            image_dir = self.output_dir / "hf_images" / dataset_name
                            image_dir.mkdir(parents=True, exist_ok=True)
                            image_path = image_dir / f"image_{i}_{key}.png"
                            value.save(image_path)
                            processed_sample[f"{key}_path"] = str(image_path)
                        else:
                            processed_sample[key] = value
                    processed_samples.append(processed_sample)
                
                # Create dataset
                hf_dataset = Dataset.from_list(processed_samples)
                
                print(f"  ✅ Created dataset with {len(hf_dataset)} samples")
                print(f"  📊 Features: {list(hf_dataset.features.keys())}")
                
                # Save locally
                local_path = self.output_dir / "hf_datasets" / dataset_name
                local_path.mkdir(parents=True, exist_ok=True)
                hf_dataset.save_to_disk(str(local_path))
                
                print(f"  💾 Saved locally: {local_path}")
                
                self.results_summary["datasets_created"].append({
                    "name": dataset_name,
                    "samples": len(hf_dataset),
                    "features": list(hf_dataset.features.keys()),
                    "local_path": str(local_path)
                })
                
                # Optional: Upload to HuggingFace Hub
                if hf_token:
                    try:
                        repo_name = f"synthdoc-{dataset_name}-multilingual-{self.timestamp}"
                        print(f"  🚀 Uploading to HuggingFace Hub: {repo_name}")
                        
                        hf_dataset.push_to_hub(
                            repo_name,
                            token=hf_token,
                            private=False  # Set to True for private datasets
                        )
                        
                        print(f"  ✅ Uploaded to: https://huggingface.co/datasets/{repo_name}")
                        
                        self.results_summary["hf_uploads"].append({
                            "dataset": dataset_name,
                            "repo_name": repo_name,
                            "url": f"https://huggingface.co/datasets/{repo_name}",
                            "samples": len(hf_dataset)
                        })
                        
                    except Exception as e:
                        print(f"  ⚠️  Upload failed: {e}")
                else:
                    print(f"  ℹ️  Skipping upload (no HuggingFace token)")
                
            except Exception as e:
                print(f"  ❌ Error creating dataset: {e}")
                continue
    
    def save_results_summary(self):
        """Save comprehensive results summary."""
        self.print_header("Results Summary")
        
        # Save detailed summary
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Pipeline Results Summary")
        print(f"   🕐 Timestamp: {self.results_summary['timestamp']}")
        print(f"   🌍 Languages tested: {len(self.results_summary['languages_tested'])}")
        print(f"   ⚙️  Workflows completed: {len(self.results_summary['workflows_completed'])}")
        print(f"   📈 Total samples: {self.results_summary['total_samples_generated']}")
        print(f"   📦 Datasets created: {len(self.results_summary['datasets_created'])}")
        print(f"   🚀 HuggingFace uploads: {len(self.results_summary['hf_uploads'])}")
        print(f"   💾 Summary saved: {summary_path}")
        
        # Print workflow details
        for workflow in self.results_summary['workflows_completed']:
            dataset_info = next((d for d in self.results_summary['datasets_created'] if d['name'] == workflow), None)
            if dataset_info:
                print(f"   • {workflow}: {dataset_info['samples']} samples")
        
        # Print HuggingFace uploads
        if self.results_summary['hf_uploads']:
            print(f"\n🤗 HuggingFace Datasets:")
            for upload in self.results_summary['hf_uploads']:
                print(f"   • {upload['dataset']}: {upload['url']}")
    
    def run_complete_pipeline(self):
        """Run the complete multilingual pipeline."""
        print("🚀 Starting Comprehensive Multilingual SynthDoc Pipeline")
        print(f"📁 Output directory: {self.output_dir}")
        
        try:
            # 1. Environment check
            has_hf_token = self.check_environment()
            
            # 2. Generate raw documents
            documents = self.generate_raw_documents()
            
            # 3. Generate handwriting samples
            handwriting = self.generate_handwriting_samples()
            
            # 4. Apply layout augmentation
            layouts = self.generate_layout_augmentation(documents)
            
            # 5. Generate VQA datasets
            vqa = self.generate_vqa_dataset(documents)
            
            # 6. Create HuggingFace datasets
            self.create_huggingface_datasets()
            
            # 7. Save results summary
            self.save_results_summary()
            
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"   📁 All outputs saved to: {self.output_dir}")
            print(f"   📊 Generated {self.results_summary['total_samples_generated']} total samples")
            print(f"   🌍 Tested {len(self.languages)} languages")
            print(f"   ⚙️  Completed {len(self.results_summary['workflows_completed'])} workflows")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point."""
    print("🌍 SynthDoc Multilingual Full Pipeline Demo")
    print("=" * 60)
    
    # Create and run demo
    demo = MultilingualPipelineDemo()
    success = demo.run_complete_pipeline()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print(f"📁 Check output directory: {demo.output_dir}")
    else:
        print("\n❌ Demo failed. Check error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 