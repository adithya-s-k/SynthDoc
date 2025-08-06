import re
import json
from typing import Dict, List, Tuple, Any

# Try to import text_utils, fallback to basic implementations if not available
try:
    from .text_utils import extract_keywords, categorize_content
except ImportError:
    # Basic fallback implementations
    def extract_keywords(text: str) -> List[str]:
        """Basic keyword extraction fallback"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return list(set(words))[:10]
    
    def categorize_content(keywords: List[str]) -> str:
        """Basic content categorization fallback"""
        tech_words = {'algorithm', 'machine', 'learning', 'data', 'neural', 'api', 'database'}
        business_words = {'revenue', 'profit', 'market', 'sales', 'customer', 'growth'}
        research_words = {'study', 'research', 'analysis', 'hypothesis', 'methodology'}
        
        if any(word in tech_words for word in keywords):
            return 'tech'
        elif any(word in business_words for word in keywords):
            return 'business'
        elif any(word in research_words for word in keywords):
            return 'research'
        else:
            return 'general'

class ContentAnalyzer:
    """Analyze text content to extract relevant data for visual generation"""
    
    def extract_visual_data(self, text: str) -> Dict[str, Any]:
        """Main method to extract all relevant data from text"""
        return {
            'numbers': self._extract_numbers(text),
            'entities': self._extract_entities(text),
            'comparisons': self._extract_comparisons(text),
            'temporal_data': self._extract_temporal_data(text),
            'technical_concepts': self._extract_technical_concepts(text),
            'data_relationships': self._extract_data_relationships(text),
            'categories': self._extract_categories(text),
            'metrics': self._extract_metrics(text)
        }
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers with context"""
        number_pattern = r'(\w+(?:\s+\w+)*)\s+(?:is|was|increased|decreased|reached|shows|equals?)\s+(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|USD|dollars?)?'
        matches = re.findall(number_pattern, text, re.IGNORECASE)
        
        numbers = []
        for match in matches:
            context, value, unit = match
            numbers.append({
                'context': context.strip(),
                'value': float(value),
                'unit': unit or '',
                'original_text': f"{context} {value} {unit}".strip()
            })
        
        # Also extract standalone numbers
        standalone_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|USD|dollars?)?\b', text)
        for num, unit in standalone_numbers[:5]:  # Limit to first 5
            if float(num) > 1:  # Ignore very small numbers
                numbers.append({
                    'context': 'value',
                    'value': float(num),
                    'unit': unit or '',
                    'original_text': f"{num} {unit}".strip()
                })
        
        return numbers[:8]  # Return max 8 numbers
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract main entities and proper nouns"""
        # Companies, products, places, people
        entity_patterns = [
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Group))\b',  # Companies
            r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2}\b',  # General proper nouns
            r'\b(?:iPhone|Android|Tesla|Microsoft|Google|Amazon|Apple|Facebook|Twitter)\b',  # Tech brands
        ]
        
        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))[:6]  # Top 6 unique entities
    
    def _extract_comparisons(self, text: str) -> List[Dict[str, str]]:
        """Extract comparison relationships"""
        comparison_pattern = r'(\w+(?:\s+\w+)*)\s+(increased|decreased|higher|lower|compared to|versus|vs|outperformed|exceeded)\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(comparison_pattern, text, re.IGNORECASE)
        
        comparisons = []
        for item1, relation, item2 in matches:
            comparisons.append({
                'item1': item1.strip(),
                'relation': relation.lower(),
                'item2': item2.strip()
            })
        
        return comparisons[:4]
    
    def _extract_temporal_data(self, text: str) -> List[str]:
        """Extract dates, years, quarters"""
        temporal_patterns = [
            r'\b(?:19|20)\d{2}\b',  # Years
            r'\b(?:Q[1-4]|Quarter\s+[1-4])\s*(?:19|20)\d{2}\b',  # Quarters
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b',  # Month Year
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:19|20)\d{2}\b'  # Short months
        ]
        
        temporal_data = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_data.extend(matches)
        
        return list(set(temporal_data))[:6]
    
    def _extract_technical_concepts(self, text: str) -> List[str]:
        """Extract technical terms and concepts"""
        tech_patterns = [
            r'\b(?:machine learning|artificial intelligence|neural network|deep learning|algorithm|blockchain|cloud computing|cybersecurity|data science|big data)\b',
            r'\b(?:API|SDK|framework|database|server|infrastructure|architecture|deployment|scalability)\b',
            r'\b(?:revenue|profit|ROI|KPI|metrics|analytics|performance|growth|market share|conversion)\b',
            r'\b(?:research|methodology|hypothesis|analysis|findings|conclusion|study|experiment)\b'
        ]
        
        concepts = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))[:5]
    
    def _extract_data_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract data relationships for graphs"""
        # Look for patterns like "X correlates with Y", "X depends on Y"
        relationship_pattern = r'(\w+(?:\s+\w+)*)\s+(?:correlates? with|depends? on|affects?|influences?|impacts?)\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(relationship_pattern, text, re.IGNORECASE)
        
        relationships = []
        for cause, effect in matches:
            relationships.append({
                'cause': cause.strip(),
                'effect': effect.strip(),
                'type': 'causal'
            })
        
        return relationships[:3]
    
    def _extract_categories(self, text: str) -> List[str]:
        """Extract categories for table organization"""
        # Look for enumerated items, bullet points, or category indicators
        category_patterns = [
            r'(?:types? of|categories of|kinds of)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:include|consists? of|comprises?)',
            r'(?:first|second|third|1\.|2\.|3\.)\s+(\w+(?:\s+\w+)*)',
        ]
        
        categories = []
        for pattern in category_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            categories.extend(matches)
        
        return list(set(categories))[:5]
    
    def _extract_metrics(self, text: str) -> List[Dict[str, Any]]:
        """Extract performance metrics and KPIs"""
        metric_pattern = r'(\w+(?:\s+\w+)*)\s+(?:rate|ratio|percentage|score|index|metric)\s+(?:is|was|of)\s+(\d+(?:\.\d+)?)\s*(%|percent)?'
        matches = re.findall(metric_pattern, text, re.IGNORECASE)
        
        metrics = []
        for metric_name, value, unit in matches:
            metrics.append({
                'name': metric_name.strip(),
                'value': float(value),
                'unit': unit or '',
                'type': 'percentage' if unit else 'absolute'
            })
        
        return metrics[:4]