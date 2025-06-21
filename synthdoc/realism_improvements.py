"""
Realism improvements for synthetic document generation.
This module contains functions to make generated documents look more realistic.
"""

import random
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any
import re

class RealismEnhancer:
    """Class to enhance the realism of synthetic documents"""
    
    def __init__(self):
        self.realistic_margins = {
            'academic': {'top': 72, 'bottom': 72, 'left': 72, 'right': 72},  # 1 inch margins
            'business': {'top': 60, 'bottom': 60, 'left': 90, 'right': 60},  # Business letter style
            'technical': {'top': 54, 'bottom': 54, 'left': 72, 'right': 72},  # Technical report
            'newsletter': {'top': 36, 'bottom': 36, 'left': 36, 'right': 36},  # Newsletter style
        }
        
        self.realistic_line_heights = {
            'tight': 1.2,
            'normal': 1.4, 
            'loose': 1.6,
            'double': 2.0
        }
        
        self.document_headers = {
            'academic': [
                "Journal of Advanced Research",
                "International Conference Proceedings", 
                "Research Quarterly",
                "Academic Review",
                "Scientific Journal"
            ],
            'business': [
                "Business Analysis Report",
                "Market Research Summary",
                "Strategic Planning Document",
                "Financial Analysis",
                "Executive Summary"
            ],
            'technical': [
                "Technical Specification",
                "System Documentation",
                "Engineering Report",
                "Implementation Guide",
                "Technical Manual"
            ]
        }
        
        self.page_numbers_styles = [
            lambda page, total: f"Page {page} of {total}",
            lambda page, total: f"{page} / {total}",
            lambda page, total: f"- {page} -",
            lambda page, total: f"{page}",
            lambda page, total: f"[{page}]"
        ]
    
    def add_realistic_headers_footers(self, draw: ImageDraw, layout_info: Dict, 
                                    page_num: int, total_pages: int, document_type: str = 'academic',
                                    title: str = None, author: str = None):
        """Add realistic headers and footers to documents"""
        
        page_width = layout_info['page_width']
        page_height = layout_info['page_height']
        margin = layout_info.get('margin', 60)
        
        # Header styling
        header_y = 20
        footer_y = page_height - 40
        
        # Choose document-appropriate header
        if title:
            header_text = title[:50] + "..." if len(title) > 50 else title
        else:
            header_text = random.choice(self.document_headers.get(document_type, self.document_headers['academic']))
        
        # Add header (only on pages after first for academic style)
        if document_type == 'academic' and page_num > 1:
            draw.text((margin, header_y), header_text, fill='gray', font=None)
        elif document_type != 'academic':
            draw.text((margin, header_y), header_text, fill='gray', font=None)
        
        # Add footer with page numbers
        page_style = random.choice(self.page_numbers_styles)
        page_text = page_style(page_num, total_pages)
        
        # Center page number
        bbox = draw.textbbox((0, 0), page_text)
        text_width = bbox[2] - bbox[0]
        page_x = (page_width - text_width) // 2
        
        draw.text((page_x, footer_y), page_text, fill='gray', font=None)
        
        # Add author/date in footer for some document types
        if document_type == 'academic' and author and random.random() < 0.3:
            draw.text((margin, footer_y), author, fill='lightgray', font=None)
    
    def add_realistic_spacing(self, text: str, document_type: str = 'academic') -> str:
        """Add realistic paragraph spacing and formatting"""
        
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Add indentation for some paragraph types
            if document_type == 'academic' and random.random() < 0.7:
                para = "    " + para  # 4-space indent
            
            # Add realistic line breaks within long paragraphs
            if len(para) > 400:
                sentences = re.split(r'[.!?]+', para)
                if len(sentences) > 3:
                    # Split long paragraphs
                    mid_point = len(sentences) // 2
                    first_half = '. '.join(sentences[:mid_point]) + '.'
                    second_half = '. '.join(sentences[mid_point:])
                    formatted_paragraphs.extend([first_half, second_half])
                else:
                    formatted_paragraphs.append(para)
            else:
                formatted_paragraphs.append(para)
        
        # Join with appropriate spacing
        spacing = '\n\n' if document_type == 'academic' else '\n\n'
        return spacing.join(formatted_paragraphs)
    
    def add_realistic_citations(self, text: str, document_type: str = 'academic') -> str:
        """Add realistic citations and references to academic text"""
        
        if document_type != 'academic':
            return text
        
        # Common citation patterns
        citations = [
            "[1]", "[2]", "[3]", "[Smith et al., 2023]", "[Johnson, 2024]", 
            "[Brown & Davis, 2023]", "[Wilson et al., 2024]", "[Lee, 2023]"
        ]
        
        sentences = text.split('. ')
        cited_sentences = []
        
        for sentence in sentences:
            # Add citations to ~20% of sentences
            if random.random() < 0.2 and len(sentence) > 50:
                citation = random.choice(citations)
                sentence = sentence + f" {citation}"
            cited_sentences.append(sentence)
        
        return '. '.join(cited_sentences)
    
    def enhance_mathematical_equations(self, text: str) -> str:
        """Make mathematical equations look more realistic"""
        
        # Replace simple math with more formatted versions
        replacements = {
            r'x = ': r'x = ',
            r'y = ': r'y = ',
            r'f\(x\) = ': r'f(x) = ',
            r'∫': r'∫',  # Keep unicode symbols
            r'∑': r'∑',
            r'∂': r'∂',
            r'∇': r'∇',
        }
        
        enhanced_text = text
        for pattern, replacement in replacements.items():
            enhanced_text = re.sub(pattern, replacement, enhanced_text)
        
        return enhanced_text
    
    def add_realistic_section_headings(self, text: str, document_type: str = 'academic') -> str:
        """Add realistic section headings and formatting"""
        
        # Detect potential headings (lines that are short and followed by content)
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if (len(line) < 100 and  # Short line
                line.strip() and  # Not empty
                not line.strip().endswith('.') and  # Doesn't end with period
                i < len(lines) - 1 and  # Not last line
                len(lines[i + 1]) > 50):  # Followed by substantial content
                
                # Format as heading
                if document_type == 'academic':
                    formatted_line = f"\n{line.strip()}\n"
                else:
                    formatted_line = f"\n{line.strip().upper()}\n"
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def get_realistic_margins(self, document_type: str = 'academic', layout_type: str = 'single') -> Dict:
        """Get realistic margins based on document type"""
        
        base_margins = self.realistic_margins.get(document_type, self.realistic_margins['academic'])
        
        # Adjust for layout type
        if layout_type in ['two_column', 'three_column']:
            # Tighter margins for multi-column layouts
            return {k: int(v * 0.8) for k, v in base_margins.items()}
        elif layout_type == 'newsletter':
            return self.realistic_margins['newsletter']
        
        return base_margins
    
    def add_realistic_watermarks(self, draw: ImageDraw, layout_info: Dict, 
                                watermark_type: str = 'draft') -> None:
        """Add subtle watermarks for realism"""
        
        if random.random() > 0.1:  # Only 10% chance of watermark
            return
        
        page_width = layout_info['page_width']
        page_height = layout_info['page_height']
        
        watermarks = {
            'draft': 'DRAFT',
            'confidential': 'CONFIDENTIAL',
            'internal': 'INTERNAL USE ONLY',
            'preliminary': 'PRELIMINARY',
            'review': 'FOR REVIEW'
        }
        
        watermark_text = watermarks.get(watermark_type, 'DRAFT')
        
        # Center watermark with rotation
        center_x = page_width // 2
        center_y = page_height // 2
        
        # Very light gray watermark
        draw.text((center_x - 100, center_y), watermark_text, 
                 fill=(240, 240, 240), font=None)

def apply_realism_enhancements(text: str, layout_info: Dict, 
                              document_type: str = 'academic',
                              page_num: int = 1, total_pages: int = 1) -> Tuple[str, Dict]:
    """Apply all realism enhancements to text and layout"""
    
    enhancer = RealismEnhancer()
    
    # Enhance text content
    enhanced_text = enhancer.add_realistic_spacing(text, document_type)
    enhanced_text = enhancer.add_realistic_citations(enhanced_text, document_type)
    enhanced_text = enhancer.enhance_mathematical_equations(enhanced_text)
    enhanced_text = enhancer.add_realistic_section_headings(enhanced_text, document_type)
    
    # Enhance layout
    realistic_margins = enhancer.get_realistic_margins(document_type, 
                                                     layout_info.get('layout_type', 'single'))
    
    enhanced_layout = layout_info.copy()
    enhanced_layout.update({
        'margins': realistic_margins,
        'line_height_multiplier': enhancer.realistic_line_heights.get('normal', 1.4),
        'document_type': document_type,
        'realism_applied': True
    })
    
    return enhanced_text, enhanced_layout
