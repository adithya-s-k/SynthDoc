import re  
from collections import Counter  
from typing import List 
def extract_keywords(text: str) -> List[str]:
    """Extract key terms from text content"""
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    word_freq = Counter(words)
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'more', 'than', 'about', 'after', 'first', 'never', 'these', 'could', 'where', 'much', 'through', 'before', 'right', 'should', 'those', 'while', 'another', 'being', 'during', 'years', 'around', 'within', 'under', 'between', 'without', 'across', 'since', 'still', 'example'}
    keywords = [word for word, freq in word_freq.most_common(10) if word not in stop_words]
    return keywords[:5]

def categorize_content(keywords: List[str]) -> str:
    """Categorize content based on keywords"""
    tech_words = {'technology', 'artificial', 'intelligence', 'computing', 'digital', 'software', 'system', 'data', 'algorithm', 'machine', 'learning'}
    business_words = {'business', 'market', 'financial', 'economic', 'company', 'organization', 'strategy', 'management', 'industry', 'commercial'}
    research_words = {'research', 'study', 'analysis', 'scientific', 'method', 'theory', 'experiment', 'results', 'findings', 'investigation'}
    
    keyword_set = set(keywords)
    if keyword_set & tech_words:
        return 'tech'
    elif keyword_set & business_words:
        return 'business'
    elif keyword_set & research_words:
        return 'research'
    else:
        return 'general'

# Helper functions for advanced rendering
def wrap_text_to_width(text: str, max_width: int, font, draw) -> List[str]:
    """Wrap text to fit within specified width"""
    if not text:
        return []
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Test if adding this word would exceed width
        test_line = ' '.join(current_line + [word])
        
        if font:
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]
            except:
                line_width = len(test_line) * 7  # Fallback
        else:
            line_width = len(test_line) * 7  # Fallback
        
        if line_width <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines