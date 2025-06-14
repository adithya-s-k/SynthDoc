"""
Document generators for different types of synthetic documents.

This module contains generators for raw documents, layout-based documents,
VQA datasets, and handwritten documents.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Base class for all document generators."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate documents."""
        pass


class DocumentGenerator(BaseGenerator):
    """Generator for raw document content using LLMs."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize DocumentGenerator.

        Args:
            model: Model name for LiteLLM (e.g., "gpt-3.5-turbo", "claude-3-sonnet", "ollama/llama2")
            api_key: API key for the model provider
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model

        if not LITELLM_AVAILABLE:
            self.logger.warning(
                "LiteLLM not available. Install with: pip install litellm"
            )
            self._llm_enabled = False
        else:
            self._llm_enabled = True
            if api_key:
                # Set API key if provided
                import os

                os.environ["OPENAI_API_KEY"] = api_key

    def generate_raw_documents(
        self,
        language: str,
        num_pages: int,
        prompt: Optional[str] = None,
        augmentations: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate raw documents from scratch.

        Args:
            language: Target language code
            num_pages: Number of pages to generate
            prompt: Custom prompt for content generation
            augmentations: List of augmentation techniques

        Returns:
            List of generated documents
        """
        self.logger.info(f"Generating {num_pages} documents in {language}")

        documents = []
        for i in range(num_pages):
            # TODO: Implement LLM-based content generation
            doc = {
                "id": f"doc_{i:04d}",
                "language": language,
                "content": self._generate_content(language, prompt),
                "metadata": {
                    "page_number": i,
                    "generation_prompt": prompt,
                    "augmentations": augmentations or [],
                },
            }
            documents.append(doc)

        return documents

    def generate_handwriting(
        self,
        content: Optional[str],
        language: str,
        handwriting_template: Optional[str],
        writing_style: str,
        paper_template: str,
    ) -> Dict[str, Any]:
        """
        Generate handwritten documents.

        Args:
            content: Text content to render
            language: Target language
            handwriting_template: Handwriting style template
            writing_style: cursive, print, or mixed
            paper_template: Background paper style

        Returns:
            Handwritten document dataset
        """
        self.logger.info(f"Generating handwritten document in {language}")

        # TODO: Implement handwriting generation
        return {
            "image": None,  # Generated handwritten image
            "content": content or self._generate_content(language),
            "style": writing_style,
            "template": handwriting_template,
            "paper": paper_template,
            "language": language,
        }

    def _generate_content(self, language: str, prompt: Optional[str] = None) -> str:
        """Generate text content for documents using LiteLLM."""
        if not self._llm_enabled:
            # Fallback to sample content if LLM not available
            if prompt:
                return f"Generated content based on: {prompt} (Language: {language})"
            else:
                return f"Sample document content in {language}"

        try:
            # Create system prompt for document generation
            system_prompt = f"""You are a document content generator. Generate realistic document content in {language}. 
            The content should be appropriate for training document understanding models and include various text structures like:
            - Headers and subheaders
            - Paragraphs of varying lengths
            - Lists (numbered and bulleted)
            - Technical terms when appropriate
            - Natural language that would appear in real documents
            
            Keep the content focused and coherent."""

            # Create user prompt
            if prompt:
                user_prompt = (
                    f"Generate document content based on this request: {prompt}"
                )
            else:
                user_prompt = f"Generate diverse, realistic document content in {language} that would be suitable for training document understanding models."

            # Call LiteLLM
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}. Using fallback content.")
            # Fallback to sample content
            if prompt:
                return f"Generated content based on: {prompt} (Language: {language})"
            else:
                return f"Sample document content in {language}"

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.generate_raw_documents(*args, **kwargs)


class LayoutGenerator(BaseGenerator):
    """Generator for layout-based document transformations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def augment_layouts(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        document_paths: Optional[List[Union[str, Path]]] = None,
        languages: List[str] = None,
        fonts: Optional[List[str]] = None,
        augmentations: List[str] = None,
        layout_templates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply layout transformations to documents.

        Returns:
            HuggingFace dataset with layout annotations
        """
        self.logger.info("Augmenting document layouts")

        # TODO: Implement layout augmentation logic
        dataset = {
            "images": [],
            "annotations": [],
            "metadata": {
                "languages": languages,
                "fonts": fonts,
                "augmentations": augmentations,
                "templates": layout_templates,
            },
        }

        return dataset

    def augment_pdfs(
        self,
        corpus_paths: List[Union[str, Path]],
        extraction_elements: List[str],
        combination_strategy: str,
        output_layout_types: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Create new documents by recombining PDF elements.

        Returns:
            HuggingFace dataset with recombined documents
        """
        self.logger.info("Augmenting PDFs with element recombination")

        # TODO: Implement PDF element extraction and recombination
        dataset = {
            "images": [],
            "annotations": [],
            "metadata": {
                "source_count": len(corpus_paths),
                "extraction_elements": extraction_elements,
                "combination_strategy": combination_strategy,
                "output_layouts": output_layout_types,
            },
        }

        return dataset

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.augment_layouts(*args, **kwargs)


class VQAGenerator(BaseGenerator):
    """Generator for Visual Question Answering datasets."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize VQAGenerator.

        Args:
            model: Model name for LiteLLM
            api_key: API key for the model provider
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model

        if not LITELLM_AVAILABLE:
            self.logger.warning(
                "LiteLLM not available. Install with: pip install litellm"
            )
            self._llm_enabled = False
        else:
            self._llm_enabled = True
            if api_key:
                import os

                os.environ["OPENAI_API_KEY"] = api_key

    def generate_vqa_dataset(
        self,
        documents: List[Dict[str, Any]],
        question_types: List[str],
        difficulty_levels: List[str],
        hard_negative_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate VQA datasets with hard negatives.

        Returns:
            Extended dataset with VQA annotations
        """
        self.logger.info(f"Generating VQA dataset for {len(documents)} documents")

        # TODO: Implement VQA generation logic
        vqa_data = {
            "questions": [],
            "answers": [],
            "hard_negatives": [],
            "metadata": {
                "document_count": len(documents),
                "question_types": question_types,
                "difficulty_levels": difficulty_levels,
                "hard_negative_ratio": hard_negative_ratio,
            },
        }

        for doc in documents:
            # Generate questions for each document
            questions = self._generate_questions(doc, question_types, difficulty_levels)
            vqa_data["questions"].extend(questions)

        return vqa_data

    def _generate_questions(
        self,
        document: Dict[str, Any],
        question_types: List[str],
        difficulty_levels: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate questions for a single document using LiteLLM."""
        questions = []

        if not self._llm_enabled:
            # Fallback to sample questions if LLM not available
            for q_type in question_types:
                for difficulty in difficulty_levels:
                    question = {
                        "question": f"Sample {q_type} question ({difficulty})",
                        "answer": "Sample answer",
                        "type": q_type,
                        "difficulty": difficulty,
                        "document_id": document.get("id"),
                    }
                    questions.append(question)
            return questions

        document_content = document.get("content", "")

        for q_type in question_types:
            for difficulty in difficulty_levels:
                try:
                    # Create system prompt for VQA generation
                    system_prompt = f"""You are a VQA dataset generator. Generate a {q_type} question with {difficulty} difficulty level about the given document content. 
                    
                    Question types:
                    - factual: Direct questions about facts in the document
                    - reasoning: Questions requiring logical inference
                    - comparative: Questions comparing different parts of the document
                    - spatial: Questions about layout and positioning
                    - counting: Questions about quantities in the document
                    
                    Difficulty levels:
                    - easy: Simple, direct questions
                    - medium: Requires some understanding of context
                    - hard: Requires complex reasoning or inference
                    
                    Return ONLY a JSON object with "question" and "answer" keys."""

                    user_prompt = f"""Document content: {document_content[:1000]}...
                    
                    Generate a {q_type} question with {difficulty} difficulty about this document."""

                    # Call LiteLLM
                    response = litellm.completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=200,
                        temperature=0.7,
                    )

                    content = response.choices[0].message.content.strip()

                    # Try to parse JSON response
                    import json

                    try:
                        qa_data = json.loads(content)
                        question_data = {
                            "question": qa_data.get(
                                "question", f"Sample {q_type} question"
                            ),
                            "answer": qa_data.get("answer", "Sample answer"),
                            "type": q_type,
                            "difficulty": difficulty,
                            "document_id": document.get("id"),
                        }
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        question_data = {
                            "question": content
                            if len(content) < 200
                            else f"Generated {q_type} question",
                            "answer": "Generated answer",
                            "type": q_type,
                            "difficulty": difficulty,
                            "document_id": document.get("id"),
                        }

                    questions.append(question_data)

                except Exception as e:
                    self.logger.warning(
                        f"VQA generation failed for {q_type}/{difficulty}: {e}"
                    )
                    # Fallback question
                    question_data = {
                        "question": f"Sample {q_type} question ({difficulty})",
                        "answer": "Sample answer",
                        "type": q_type,
                        "difficulty": difficulty,
                        "document_id": document.get("id"),
                    }
                    questions.append(question_data)

        return questions

    def generate(self, *args, **kwargs) -> Any:
        """Generic generate method."""
        return self.generate_vqa_dataset(*args, **kwargs)
