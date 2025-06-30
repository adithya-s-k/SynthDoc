#!/usr/bin/env python3
"""
VQA Pipeline Demo

This script demonstrates SynthDoc's complete Visual Question Answering (VQA) 
dataset generation pipeline, including question generation, hard negatives, 
and comprehensive dataset creation for training robust VQA models.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from synthdoc import SynthDoc, create_vqa_dataset, VQAGenerator
from synthdoc.models import VQAGenerationConfig


def main():
    """Demonstrate complete VQA dataset generation pipeline."""
    print("❓ SynthDoc VQA Pipeline Demo")
    print("=" * 50)
    
    # Initialize SynthDoc with API key for LLM features
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    output_dir = "./vqa_pipeline_output"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    if not api_key:
        print("⚠️  VQA generation requires API key. Set GROQ_API_KEY or OPENAI_API_KEY")
        print("    Continuing with limited functionality...")
    else:
        print("✅ Using LLM for intelligent VQA generation")
    
    synth = SynthDoc(
        output_dir=output_dir,
        llm_model="gpt-4o-mini",
        api_key=api_key
    )

    # Step 1: Generate Source Documents for VQA
    print("\n📄 Step 1: Generating Source Documents")
    print("-" * 40)
    
    # Create diverse document types for VQA
    document_templates = [
        {
            "lang": "en",
            "prompt": "Generate a technical research paper abstract about machine learning with specific statistics and figures",
            "doc_type": "research_paper"
        },
        {
            "lang": "en", 
            "prompt": "Create a business quarterly report with financial data, charts, and performance metrics",
            "doc_type": "business_report"
        },
        {
            "lang": "en",
            "prompt": "Generate a medical document describing treatment procedures and patient information",
            "doc_type": "medical_document"
        },
        {
            "lang": "en",
            "prompt": "Create a legal contract document with clauses, dates, and party information", 
            "doc_type": "legal_document"
        },
        {
            "lang": "hi",
            "prompt": "भारतीय सरकारी नीति के बारे में एक आधिकारिक दस्तावेज़ बनाएं",
            "doc_type": "government_document"
        }
    ]
    
    source_documents = []
    document_metadata = []
    
    for i, template in enumerate(document_templates):
        print(f"\n📝 Generating {template['doc_type']}...")
        
        try:
            docs = synth.generate_raw_docs(
                language=template["lang"],
                num_pages=1,
                prompt=template["prompt"]
            )
            
            if docs:
                source_documents.extend(docs)
                document_metadata.append({
                    "id": f"doc_{i}",
                    "type": template["doc_type"],
                    "language": template["lang"],
                    "source": "synthetic_generation"
                })
                print(f"  ✅ Generated {template['doc_type']}")
                
                # Show content preview
                content_preview = docs[0].get("content", "")[:100]
                print(f"  📝 Preview: {content_preview}...")
            
        except Exception as e:
            print(f"  ❌ Error generating {template['doc_type']}: {e}")
    
    print(f"\n📊 Total source documents: {len(source_documents)}")

    # Step 2: Basic VQA Generation
    print("\n❓ Step 2: Basic VQA Generation")
    print("-" * 40)
    
    if source_documents and api_key:
        print("🤖 Generating basic question-answer pairs...")
        
        try:
            basic_vqa = synth.generate_vqa(
                source_documents=source_documents[:3],  # Use first 3 documents
                question_types=["factual", "reasoning"],
                difficulty_levels=["easy", "medium"],
                hard_negative_ratio=0.2
            )
            
            num_questions = len(basic_vqa.get("questions", []))
            print(f"✅ Generated {num_questions} basic VQA pairs")
            
            # Show sample Q&A
            if num_questions > 0:
                print(f"\n📋 Sample Basic VQA:")
                print(f"   Q: {basic_vqa['questions'][0]}")
                print(f"   A: {basic_vqa['answers'][0]}")
                if basic_vqa.get('hard_negatives') and basic_vqa['hard_negatives'][0]:
                    print(f"   Hard Negative: {basic_vqa['hard_negatives'][0][0]}")
            
        except Exception as e:
            print(f"❌ Basic VQA generation failed: {e}")
            basic_vqa = {"questions": [], "answers": [], "hard_negatives": []}
    else:
        print("⚠️  Skipping basic VQA (no API key or documents)")
        basic_vqa = {"questions": [], "answers": [], "hard_negatives": []}

    # Step 3: Advanced VQA with Different Question Types
    print("\n🎯 Step 3: Advanced VQA Generation")
    print("-" * 40)
    
    # Define comprehensive question types
    question_type_configs = [
        {
            "name": "Factual Questions",
            "types": ["factual"],
            "description": "Direct questions about document content",
            "difficulty": ["easy", "medium"]
        },
        {
            "name": "Reasoning Questions", 
            "types": ["reasoning"],
            "description": "Questions requiring logical inference",
            "difficulty": ["medium", "hard"]
        },
        {
            "name": "Comparative Questions",
            "types": ["comparative"],
            "description": "Questions comparing information across documents",
            "difficulty": ["medium", "hard"]
        },
        {
            "name": "Numerical Questions",
            "types": ["numerical"],
            "description": "Questions about numbers, dates, and quantities",
            "difficulty": ["easy", "medium"]
        },
        {
            "name": "Structural Questions",
            "types": ["structural"],
            "description": "Questions about document layout and organization",
            "difficulty": ["easy", "medium"]
        }
    ]
    
    advanced_vqa_results = {}
    
    for config in question_type_configs:
        name = config["name"]
        types = config["types"]
        description = config["description"]
        difficulty = config["difficulty"]
        
        print(f"\n🔍 Generating {name}...")
        print(f"   Types: {', '.join(types)}")
        print(f"   Description: {description}")
        
        if source_documents and api_key:
            try:
                vqa_result = synth.generate_vqa(
                    source_documents=source_documents[:2],  # Use first 2 documents
                    question_types=types,
                    difficulty_levels=difficulty,
                    hard_negative_ratio=0.3
                )
                
                num_questions = len(vqa_result.get("questions", []))
                advanced_vqa_results[name] = vqa_result
                print(f"   ✅ Generated {num_questions} questions")
                
                # Show sample question
                if num_questions > 0:
                    sample_q = vqa_result["questions"][0]
                    sample_a = vqa_result["answers"][0]
                    print(f"   📝 Sample: Q: {sample_q[:50]}...")
                    print(f"           A: {sample_a[:50]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                advanced_vqa_results[name] = {"questions": [], "answers": [], "hard_negatives": []}
        else:
            print("   ⚠️  Skipped (no API key or documents)")
            advanced_vqa_results[name] = {"questions": [], "answers": [], "hard_negatives": []}

    # Step 4: Hard Negative Generation
    print("\n🎭 Step 4: Hard Negative Generation")
    print("-" * 40)
    
    print("🧠 Generating challenging hard negatives...")
    
    hard_negative_strategies = [
        {
            "name": "Semantic Similarity",
            "ratio": 0.4,
            "description": "Answers that are semantically similar but factually incorrect"
        },
        {
            "name": "Context Confusion",
            "ratio": 0.3,
            "description": "Answers from related but different document sections"
        },
        {
            "name": "Numerical Variations", 
            "ratio": 0.3,
            "description": "Similar numbers or dates with slight modifications"
        }
    ]
    
    hard_negative_results = {}
    
    for strategy in hard_negative_strategies:
        name = strategy["name"]
        ratio = strategy["ratio"]
        description = strategy["description"]
        
        print(f"\n🎯 Strategy: {name}")
        print(f"   Ratio: {ratio}")
        print(f"   Description: {description}")
        
        if source_documents and api_key:
            try:
                # Generate with high hard negative ratio
                hn_result = synth.generate_vqa(
                    source_documents=source_documents[:1],
                    question_types=["factual", "reasoning"],
                    difficulty_levels=["medium"],
                    hard_negative_ratio=ratio
                )
                
                num_negatives = sum(len(hn) for hn in hn_result.get("hard_negatives", []))
                hard_negative_results[name] = hn_result
                print(f"   ✅ Generated {num_negatives} hard negatives")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                hard_negative_results[name] = {"questions": [], "answers": [], "hard_negatives": []}
        else:
            print("   ⚠️  Skipped (no API key or documents)")

    # Step 5: Cross-Document VQA
    print("\n🔗 Step 5: Cross-Document VQA")
    print("-" * 40)
    
    print("🌐 Generating questions that span multiple documents...")
    
    if len(source_documents) >= 2 and api_key:
        try:
            cross_doc_vqa = synth.generate_vqa(
                source_documents=source_documents,  # Use all documents
                question_types=["comparative", "reasoning"],
                difficulty_levels=["medium", "hard"],
                hard_negative_ratio=0.25
            )
            
            num_cross_questions = len(cross_doc_vqa.get("questions", []))
            print(f"✅ Generated {num_cross_questions} cross-document questions")
            
            # Show sample cross-document question
            if num_cross_questions > 0:
                print(f"\n📋 Sample Cross-Document VQA:")
                print(f"   Q: {cross_doc_vqa['questions'][0]}")
                print(f"   A: {cross_doc_vqa['answers'][0]}")
            
        except Exception as e:
            print(f"❌ Cross-document VQA failed: {e}")
            cross_doc_vqa = {"questions": [], "answers": [], "hard_negatives": []}
    else:
        print("⚠️  Insufficient documents or no API key for cross-document VQA")
        cross_doc_vqa = {"questions": [], "answers": [], "hard_negatives": []}

    # Step 6: Multilingual VQA
    print("\n🌍 Step 6: Multilingual VQA")
    print("-" * 40)
    
    print("🗣️  Generating VQA in multiple languages...")
    
    multilingual_vqa = {}
    
    # Find documents in different languages
    lang_documents = {}
    for i, doc in enumerate(source_documents):
        # Determine language from metadata
        lang = document_metadata[i]["language"] if i < len(document_metadata) else "en"
        if lang not in lang_documents:
            lang_documents[lang] = []
        lang_documents[lang].append(doc)
    
    for lang, docs in lang_documents.items():
        print(f"\n🔤 Generating VQA in {lang.upper()}...")
        
        if docs and api_key:
            try:
                lang_vqa = synth.generate_vqa(
                    source_documents=docs[:1],  # Use first document in language
                    question_types=["factual", "reasoning"],
                    difficulty_levels=["easy", "medium"],
                    hard_negative_ratio=0.2
                )
                
                num_lang_questions = len(lang_vqa.get("questions", []))
                multilingual_vqa[lang] = lang_vqa
                print(f"   ✅ Generated {num_lang_questions} questions in {lang}")
                
            except Exception as e:
                print(f"   ❌ Error for {lang}: {e}")
                multilingual_vqa[lang] = {"questions": [], "answers": [], "hard_negatives": []}
        else:
            print(f"   ⚠️  Skipped {lang} (no documents or API key)")

    # Step 7: Dataset Quality Analysis
    print("\n📊 Step 7: Dataset Quality Analysis")
    print("-" * 40)
    
    print("🔍 Analyzing generated VQA dataset quality...")
    
    # Collect all VQA data for analysis
    all_questions = []
    all_answers = []
    all_hard_negatives = []
    
    # Add basic VQA
    all_questions.extend(basic_vqa.get("questions", []))
    all_answers.extend(basic_vqa.get("answers", []))
    all_hard_negatives.extend(basic_vqa.get("hard_negatives", []))
    
    # Add advanced VQA
    for vqa_result in advanced_vqa_results.values():
        all_questions.extend(vqa_result.get("questions", []))
        all_answers.extend(vqa_result.get("answers", []))
        all_hard_negatives.extend(vqa_result.get("hard_negatives", []))
    
    # Add cross-document VQA
    all_questions.extend(cross_doc_vqa.get("questions", []))
    all_answers.extend(cross_doc_vqa.get("answers", []))
    all_hard_negatives.extend(cross_doc_vqa.get("hard_negatives", []))
    
    # Add multilingual VQA
    for lang_vqa in multilingual_vqa.values():
        all_questions.extend(lang_vqa.get("questions", []))
        all_answers.extend(lang_vqa.get("answers", []))
        all_hard_negatives.extend(lang_vqa.get("hard_negatives", []))
    
    # Quality metrics
    total_qa_pairs = len(all_questions)
    total_hard_negatives = sum(len(hn) for hn in all_hard_negatives)
    
    if total_qa_pairs > 0:
        avg_question_length = sum(len(q.split()) for q in all_questions) / total_qa_pairs
        avg_answer_length = sum(len(a.split()) for a in all_answers) / total_qa_pairs
        
        print(f"\n📈 Quality Metrics:")
        print(f"  Total Q&A pairs: {total_qa_pairs}")
        print(f"  Total hard negatives: {total_hard_negatives}")
        print(f"  Average question length: {avg_question_length:.1f} words")
        print(f"  Average answer length: {avg_answer_length:.1f} words")
        print(f"  Hard negative ratio: {total_hard_negatives/total_qa_pairs:.2f}" if total_qa_pairs > 0 else "  Hard negative ratio: 0")
        
        # Question type distribution
        question_types = {}
        for result in advanced_vqa_results.values():
            if "question_types" in result:
                for qt in result["question_types"]:
                    question_types[qt] = question_types.get(qt, 0) + 1
        
        if question_types:
            print(f"\n📊 Question Type Distribution:")
            for qtype, count in question_types.items():
                print(f"  {qtype}: {count} questions")
    
    # Step 8: Export Dataset
    print("\n💾 Step 8: Exporting VQA Dataset")
    print("-" * 40)
    
    print("📦 Creating comprehensive VQA dataset export...")
    
    # Create comprehensive dataset structure
    complete_dataset = {
        "metadata": {
            "dataset_name": "SynthDoc_VQA_Pipeline",
            "version": "1.0",
            "creation_date": "2024",
            "total_documents": len(source_documents),
            "total_qa_pairs": total_qa_pairs,
            "languages": list(lang_documents.keys()),
            "question_types": list(question_types.keys()) if 'question_types' in locals() else [],
            "has_hard_negatives": total_hard_negatives > 0
        },
        "documents": [
            {
                "id": f"doc_{i}",
                "metadata": document_metadata[i] if i < len(document_metadata) else {},
                "content_preview": doc.get("content", "")[:200] + "..." if doc.get("content") else ""
            }
            for i, doc in enumerate(source_documents)
        ],
        "vqa_data": {
            "basic_vqa": basic_vqa,
            "advanced_vqa": advanced_vqa_results,
            "cross_document_vqa": cross_doc_vqa,
            "multilingual_vqa": multilingual_vqa,
            "hard_negatives": hard_negative_results
        },
        "quality_metrics": {
            "total_qa_pairs": total_qa_pairs,
            "total_hard_negatives": total_hard_negatives,
            "avg_question_length": avg_question_length if 'avg_question_length' in locals() else 0,
            "avg_answer_length": avg_answer_length if 'avg_answer_length' in locals() else 0
        }
    }
    
    # Save dataset
    dataset_path = Path(output_dir) / "complete_vqa_dataset.json"
    try:
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(complete_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ Dataset exported to: {dataset_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Save individual components
    components = [
        ("basic_vqa.json", basic_vqa),
        ("advanced_vqa.json", advanced_vqa_results),
        ("cross_document_vqa.json", cross_doc_vqa),
        ("multilingual_vqa.json", multilingual_vqa)
    ]
    
    for filename, data in components:
        if data and any(data.get("questions", [])):  # Only save if has questions
            component_path = Path(output_dir) / filename
            try:
                with open(component_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"❌ Failed to save {filename}: {e}")

    # Final Summary
    print("\n🎉 VQA Pipeline Summary")
    print("=" * 50)
    
    print(f"📄 Source documents generated: {len(source_documents)}")
    print(f"❓ Total VQA pairs created: {total_qa_pairs}")
    print(f"🎭 Total hard negatives: {total_hard_negatives}")
    print(f"🌍 Languages covered: {', '.join(lang_documents.keys())}")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🎯 VQA Types Generated:")
    vqa_types = [
        f"✅ Basic factual and reasoning questions",
        f"✅ Advanced question types (comparative, numerical, structural)",
        f"✅ Hard negative generation with multiple strategies",
        f"✅ Cross-document comparative questions",
        f"✅ Multilingual VQA support",
        f"✅ Quality metrics and analysis"
    ]
    
    for vqa_type in vqa_types:
        print(f"  {vqa_type}")
    
    print(f"\n💡 Best Practices Demonstrated:")
    practices = [
        "Generate diverse source documents for comprehensive coverage",
        "Use multiple question types for robust training data",
        "Include hard negatives to improve model discrimination",
        "Support cross-document reasoning capabilities",
        "Implement multilingual VQA for global applications",
        "Perform quality analysis for dataset validation",
        "Export structured datasets for easy integration"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"  {i}. {practice}")
    
    print(f"\n🔧 Technical Implementation:")
    tech_details = [
        f"LLM-powered question and answer generation",
        f"Configurable hard negative generation ratios",
        f"Support for multiple difficulty levels",
        f"Cross-language VQA generation",
        f"Comprehensive quality metrics",
        f"Structured JSON dataset export",
        f"Modular pipeline architecture"
    ]
    
    for detail in tech_details:
        print(f"  • {detail}")


if __name__ == "__main__":
    main() 