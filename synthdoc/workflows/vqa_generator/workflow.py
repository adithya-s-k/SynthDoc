import random
import json
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import litellm
from datasets import Dataset
from ..base import BaseWorkflow
from ...models import VQAGenerationConfig, WorkflowResult
from ...utils import CostTracker


class VQAGenerator(BaseWorkflow):
    """Generate visual question-answering datasets with LLM integration."""

    def __init__(self, api_key: Optional[str] = None, llm_model: str = "gpt-4o-mini"):
        super().__init__()
        self.api_key = api_key
        self.llm_model = llm_model
        self.cost_tracker = CostTracker()
        
        # Setup LLM if API key provided
        if api_key:
            import os
            if "gpt" in llm_model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "claude" in llm_model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key

    def process(self, config: VQAGenerationConfig) -> WorkflowResult:
        """Generate VQA dataset based on configuration."""
        print(f"🤖 Starting VQA generation for {len(config.documents)} documents...")
        
        all_questions = []
        all_answers = []
        all_hard_negatives = []
        all_metadata = []
        
        for doc_idx, doc_path in enumerate(config.documents):
            print(f"Processing document {doc_idx + 1}/{len(config.documents)}: {doc_path}")
            
            # Load document content (this would need to be implemented based on document type)
            doc_content = self._extract_document_content(doc_path)
            
            # Generate questions for this document
            questions, answers, hard_negatives, metadata = self._generate_vqa_for_document(
                doc_content, doc_path, config
            )
            
            all_questions.extend(questions)
            all_answers.extend(answers)
            all_hard_negatives.extend(hard_negatives)
            all_metadata.extend(metadata)
        
                # Create dataset samples for image-based VQA
        samples = []
        for i in range(len(all_questions)):
            # Handle both document paths and image objects
            doc_path = all_metadata[i]["document_path"]
            
            # Try to load image if document is an image file
            image = None
            if isinstance(doc_path, str) and doc_path.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    from PIL import Image
                    image = Image.open(doc_path)
                except Exception as e:
                    print(f"⚠️ Could not load image {doc_path}: {e}")
            
            sample = {
                "id": f"vqa_{i}",
                "image": image,  # Include image for VQA
                "text": self._extract_document_content(doc_path),  # Include text content
                "document_path": doc_path,
                "question": all_questions[i],
                "answer": all_answers[i],
                "hard_negatives": all_hard_negatives[i] if i < len(all_hard_negatives) else [],
                "question_type": all_metadata[i]["question_type"],
                "difficulty": all_metadata[i]["difficulty"],
                "similarity_scores": all_metadata[i].get("similarity_scores", []),
                "metadata": {
                    "source": str(doc_path),
                    "language": "en",  # Default language
                    "question_type": all_metadata[i]["question_type"]
                }
            }
            samples.append(sample)

        # Create comprehensive dataset using README schema
        if not samples:
            dataset = Dataset.from_dict({})
        else:
            # Extract data for comprehensive dataset creation
            images = [s['image'] for s in samples]
            image_paths = [s.get('image_path', '') for s in samples]
            pdf_names = [s.get('pdf_name', f"vqa_doc_{i}") for i, s in enumerate(samples)]
            page_numbers = [s.get('page_number', 0) for s in samples]
            
            # VQA-specific content with questions and answers
            markdown_content = []
            html_content = []
            for s in samples:
                q = s.get('question', '')
                a = s.get('answer', '')
                base_md = s.get('markdown', f"Document: {s.get('text', '')}")
                base_html = s.get('html', f"<p>{s.get('text', '')}</p>")
                
                enhanced_md = f"{base_md}\n\n**Question:** {q}\n**Answer:** {a}"
                enhanced_html = f"{base_html}<br><strong>Q:</strong> {q}<br><strong>A:</strong> {a}"
                
                markdown_content.append(enhanced_md)
                html_content.append(enhanced_html)
            
            dataset = self._create_comprehensive_hf_dataset(
                images=images,
                image_paths=image_paths,
                pdf_names=pdf_names,
                page_numbers=page_numbers,
                markdown_content=markdown_content,
                html_content=html_content,
                additional_metadata={
                    "workflow": "vqa_generation",
                    "config": config.dict(),
                    "total_cost": self.cost_tracker.get_summary()
                }
            )

        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "vqa_generation",
                "total_questions": len(all_questions),
                "cost_summary": self.cost_tracker.get_summary()
            },
            num_samples=len(samples),
            output_files=[]
        )

    def _extract_document_content(self, doc_path: str) -> str:
        """Extract text content from document."""
        # This is a simplified version - would need to handle different document types
        try:
            if isinstance(doc_path, dict) and "content" in doc_path:
                return doc_path["content"]
            elif str(doc_path).endswith(('.txt', '.md')):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For now, return sample content for testing
                return f"Sample document content for {doc_path}"
        except Exception as e:
            print(f"⚠️ Error extracting content from {doc_path}: {e}")
            return f"Sample document content for {doc_path}"

    def _generate_vqa_for_document(self, content: str, doc_path: str, config: VQAGenerationConfig) -> tuple:
        """Generate VQA pairs for a single document."""
        questions = []
        answers = []
        hard_negatives = []
        metadata = []
        
        question_types = config.question_types or ["factual", "reasoning", "comparative"]
        
        for q_idx in range(config.num_questions_per_doc):
            question_type = question_types[q_idx % len(question_types)]
            
            if self.api_key and self.llm_model:
                # Use LLM to generate high-quality questions
                question, answer, neg_answers = self._generate_llm_vqa(
                    content, question_type, doc_path
                )
            else:
                # Use fallback template-based generation
                question, answer, neg_answers = self._generate_template_vqa(
                    content, question_type, q_idx
                )
            
            questions.append(question)
            answers.append(answer)
            
            # Add hard negatives if enabled
            if config.include_hard_negatives:
                hard_negatives.append(neg_answers)
            else:
                hard_negatives.append([])
            
            metadata.append({
                "document_path": str(doc_path),
                "question_type": question_type,
                "difficulty": self._assess_difficulty(question),
                "similarity_scores": self._calculate_similarity_scores(answer, neg_answers)
            })
        
        return questions, answers, hard_negatives, metadata

    def _generate_llm_vqa(self, content: str, question_type: str, doc_path: str) -> tuple:
        """Generate VQA using LLM."""
        try:
            prompt = self._create_vqa_prompt(content, question_type)
            
            response = litellm.completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating educational questions and answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            cost = self.cost_tracker.track_usage(response, self.llm_model)
            print(f"💰 VQA Generation: ${cost:.6f}")
            
            result = response.choices[0].message.content.strip()
            question, answer, negatives = self._parse_llm_response(result)
            
            return question, answer, negatives
            
        except Exception as e:
            print(f"⚠️ LLM VQA generation failed: {e}")
            return self._generate_template_vqa(content, question_type, 0)

    def _create_vqa_prompt(self, content: str, question_type: str) -> str:
        """Create a prompt for LLM VQA generation."""
        content_preview = content[:1000] + "..." if len(content) > 1000 else content
        
        prompt = f"""
Based on this document content:
"{content_preview}"

Generate a {question_type} question and answer pair, plus 3 hard negative answers.

Requirements:
- Question should be clear and specific
- Answer should be accurate and based on the content
- Hard negatives should be plausible but incorrect
- Format as JSON: {{"question": "...", "answer": "...", "hard_negatives": ["...", "...", "..."]}}

Question type guidelines:
- factual: Ask about specific facts or details
- reasoning: Require inference or analysis  
- comparative: Compare different elements or concepts
"""
        return prompt

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract question, answer, and negatives."""
        try:
            # Try to parse JSON response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return (
                    data.get("question", "Generated question"),
                    data.get("answer", "Generated answer"),
                    data.get("hard_negatives", ["Negative 1", "Negative 2", "Negative 3"])
                )
        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
        
        # Fallback parsing
        lines = response.split('\n')
        question = next((line for line in lines if '?' in line), "Generated question?")
        answer = next((line for line in lines if 'answer' in line.lower() or 'Answer' in line), "Generated answer")
        
        return question, answer, ["Hard negative 1", "Hard negative 2", "Hard negative 3"]

    def _generate_template_vqa(self, content: str, question_type: str, idx: int) -> tuple:
        """Generate VQA using templates when LLM is not available."""
        templates = {
            "factual": {
                "questions": [
                    "What is the main topic discussed in this document?",
                    "What are the key points mentioned?",
                    "What specific details are provided?",
                ],
                "answers": [
                    "The main topic is document processing and analysis.",
                    "Key points include methodology and results.",
                    "Specific technical details and measurements.",
                ]
            },
            "reasoning": {
                "questions": [
                    "Why is this approach effective?",
                    "How do the different components work together?",
                    "What can be inferred from the results?",
                ],
                "answers": [
                    "This approach is effective because it combines multiple techniques.",
                    "Components work together through systematic integration.",
                    "Results suggest improved performance and reliability.",
                ]
            },
            "comparative": {
                "questions": [
                    "How does this compare to alternative approaches?",
                    "What are the advantages and disadvantages?",
                    "Which method performs better?",
                ],
                "answers": [
                    "This approach offers better accuracy than alternatives.",
                    "Advantages include speed; disadvantages include complexity.",
                    "The proposed method shows superior performance.",
                ]
            }
        }
        
        template = templates.get(question_type, templates["factual"])
        q_idx = idx % len(template["questions"])
        
        question = template["questions"][q_idx]
        answer = template["answers"][q_idx]
        
        # Generate hard negatives
        hard_negatives = [
            f"Incorrect answer option A for question {idx + 1}",
            f"Incorrect answer option B for question {idx + 1}",
            f"Incorrect answer option C for question {idx + 1}"
        ]
        
        return question, answer, hard_negatives

    def _assess_difficulty(self, question: str) -> str:
        """Assess question difficulty based on complexity indicators."""
        difficulty_indicators = {
            "easy": ["what", "where", "when", "who"],
            "medium": ["how", "why", "explain", "describe"],
            "hard": ["analyze", "compare", "evaluate", "synthesize", "infer"]
        }
        
        question_lower = question.lower()
        
        for difficulty, indicators in difficulty_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return difficulty
        
        return "medium"  # Default

    def _calculate_similarity_scores(self, correct_answer: str, hard_negatives: List[str]) -> List[float]:
        """Calculate similarity scores between correct answer and hard negatives."""
        # Simplified similarity calculation based on word overlap
        def word_overlap_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
        
        scores = []
        for negative in hard_negatives:
            score = word_overlap_similarity(correct_answer, negative)
            scores.append(round(score, 3))
        
        return scores 