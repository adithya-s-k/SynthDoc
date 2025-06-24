"""
Advanced layout generation for multi-column documents.

This module provides sophisticated layout algorithms for creating
realistic document layouts including multi-column, academic papers,
newsletters, and sidebar layouts.
"""

from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import random
import math
from config import LayoutType

class LayoutManager:
    """Manages different document layout types and text flow."""
    
    def __init__(self):
        self.margin = 60
        self.column_gap = 30
        self.line_height = 24
        self.paragraph_spacing = 18
        
    def calculate_layout_dimensions(self, layout_type: LayoutType, page_width: int, page_height: int) -> Dict[str, Any]:
        """Calculate dimensions for different layout types."""
        
        content_width = page_width - (2 * self.margin)
        content_height = page_height - (2 * self.margin)
        
        layouts = {
            LayoutType.SINGLE_COLUMN: {
                'columns': 1,
                'column_width': content_width,
                'column_positions': [self.margin],
                'header_height': 80,
                'footer_height': 40
            },
            
            LayoutType.TWO_COLUMN: {
                'columns': 2,
                'column_width': (content_width - self.column_gap) // 2,
                'column_positions': [
                    self.margin,
                    self.margin + (content_width - self.column_gap) // 2 + self.column_gap
                ],
                'header_height': 60,
                'footer_height': 40
            },
            
            LayoutType.THREE_COLUMN: {
                'columns': 3,
                'column_width': (content_width - 2 * self.column_gap) // 3,
                'column_positions': [
                    self.margin,
                    self.margin + (content_width - 2 * self.column_gap) // 3 + self.column_gap,
                    self.margin + 2 * ((content_width - 2 * self.column_gap) // 3 + self.column_gap)
                ],
                'header_height': 60,
                'footer_height': 40
            },
            
            LayoutType.ACADEMIC_PAPER: {
                'columns': 2,
                'column_width': (content_width - self.column_gap) // 2,
                'column_positions': [
                    self.margin,
                    self.margin + (content_width - self.column_gap) // 2 + self.column_gap
                ],
                'header_height': 120,  # Space for title, authors, abstract
                'footer_height': 40,
                'title_area': True,
                'abstract_area': True
            },
              LayoutType.NEWSLETTER: {
                'columns': 2, 
                'column_width': (content_width - self.column_gap) // 2,
                'column_positions': [
                    self.margin,
                    self.margin + (content_width - self.column_gap) // 2 + self.column_gap
                ],
                'header_height': 100,
                'footer_height': 40,
                'mixed_layout': True
            },
            
            LayoutType.SIDEBAR: {
                'columns': 'sidebar',
                'main_width': int(content_width * 0.7),
                'sidebar_width': int(content_width * 0.25),
                'main_position': self.margin,
                'sidebar_position': self.margin + int(content_width * 0.7) + self.column_gap,
                'header_height': 60,
                'footer_height': 40
            }
        }
        
        layout = layouts[layout_type].copy()
        layout.update({
            'content_width': content_width,
            'content_height': content_height,
            'page_width': page_width,
            'page_height': page_height,
            'margin': self.margin,
            'column_gap': self.column_gap
        })
        
        return layout
    
    def wrap_text_to_columns(self, text: str, layout_info: Dict, font: ImageFont, 
                           draw: ImageDraw) -> List[Dict]:
        """Wrap text into columns based on layout."""
        
        words = text.split()
        columns_data = []
        
        # Handle different layout types
        if layout_info['columns'] == 1:
            # Single column
            column_text = self._wrap_text_to_width(words, layout_info['column_width'], font, draw)
            columns_data.append({
                'text_lines': column_text,
                'x_position': layout_info['column_positions'][0],
                'width': layout_info['column_width']
            })
            
        elif isinstance(layout_info['columns'], int) and layout_info['columns'] > 1:
            # Multi-column (equal width)
            words_per_column = len(words) // layout_info['columns']
            
            for col_idx in range(layout_info['columns']):
                start_idx = col_idx * words_per_column
                if col_idx == layout_info['columns'] - 1:
                    # Last column gets remaining words
                    col_words = words[start_idx:]
                else:
                    col_words = words[start_idx:start_idx + words_per_column]
                
                column_text = self._wrap_text_to_width(col_words, layout_info['column_width'], font, draw)
                columns_data.append({
                    'text_lines': column_text,
                    'x_position': layout_info['column_positions'][col_idx],
                    'width': layout_info['column_width']
                })
                
        elif layout_info['columns'] == 'sidebar':
            # Sidebar layout
            main_words = words[:int(len(words) * 0.75)]  # 75% to main content
            sidebar_words = words[int(len(words) * 0.75):]  # 25% to sidebar
            
            main_text = self._wrap_text_to_width(main_words, layout_info['main_width'], font, draw)
            sidebar_text = self._wrap_text_to_width(sidebar_words, layout_info['sidebar_width'], font, draw)
            
            columns_data.append({
                'text_lines': main_text,
                'x_position': layout_info['main_position'],
                'width': layout_info['main_width'],
                'type': 'main'
            })
            columns_data.append({
                'text_lines': sidebar_text,
                'x_position': layout_info['sidebar_position'],
                'width': layout_info['sidebar_width'],
                'type': 'sidebar'
            })
            
        return columns_data
    
    def _wrap_text_to_width(self, words: List[str], max_width: int, font: ImageFont, 
                           draw: ImageDraw) -> List[str]:
        """Wrap words to fit within specified width."""
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            
            try:
                if font:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_width = bbox[2] - bbox[0]
                else:
                    line_width = len(test_line) * 8
            except:
                line_width = len(test_line) * 8
                
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines
    
    def add_page_elements(self, draw: ImageDraw, layout_info: Dict, page_num: int, 
                         title_font: ImageFont, font: ImageFont, title: str = None):
        """Add page headers, footers, and other elements."""
        
        page_width = layout_info['page_width']
        page_height = layout_info['page_height']
        margin = layout_info['margin']
        
        # Add header based on layout type
        if layout_info.get('title_area') and page_num == 1:
            # Academic paper title area
            self._add_academic_header(draw, layout_info, title_font, font, title)
        else:
            # Regular header
            self._add_regular_header(draw, layout_info, font, page_num)
        
        # Add footer
        self._add_footer(draw, layout_info, font, page_num)
        
        # Add column separators for multi-column layouts
        if isinstance(layout_info.get('columns'), int) and layout_info['columns'] > 1:
            self._add_column_separators(draw, layout_info)
    
    def _add_academic_header(self, draw: ImageDraw, layout_info: Dict, title_font: ImageFont, 
                           font: ImageFont, title: str):
        """Add academic paper style header with title and authors."""
        margin = layout_info['margin']
        center_x = layout_info['page_width'] // 2
        
        # Title
        if title and title_font:
            title_text = title or "Research Paper Title"
            bbox = draw.textbbox((0, 0), title_text, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text((center_x - title_width//2, margin), title_text, 
                     fill='black', font=title_font)
        
        # Authors (placeholder)
        if font:
            author_text = "Author Name, Institution"
            bbox = draw.textbbox((0, 0), author_text, font=font)
            author_width = bbox[2] - bbox[0]
            draw.text((center_x - author_width//2, margin + 40), author_text, 
                     fill='black', font=font)
    
    def _add_regular_header(self, draw: ImageDraw, layout_info: Dict, font: ImageFont, page_num: int):
        """Add regular document header."""
        margin = layout_info['margin']
        page_width = layout_info['page_width']
        
        # Simple header line
        draw.line([(margin, margin - 20), (page_width - margin, margin - 20)], 
                 fill='gray', width=1)
    
    def _add_footer(self, draw: ImageDraw, layout_info: Dict, font: ImageFont, page_num: int):
        """Add page footer with page number."""
        margin = layout_info['margin']
        page_width = layout_info['page_width']
        page_height = layout_info['page_height']
        
        # Page number
        page_text = f"- {page_num + 1} -"
        if font:
            bbox = draw.textbbox((0, 0), page_text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((page_width//2 - text_width//2, page_height - margin + 10), 
                     page_text, fill='black', font=font)
        
        # Footer line
        draw.line([(margin, page_height - margin + 5), 
                  (page_width - margin, page_height - margin + 5)], 
                 fill='gray', width=1)
    
    def _add_column_separators(self, draw: ImageDraw, layout_info: Dict):
        """Add visual separators between columns."""
        if layout_info['columns'] <= 1:
            return
            
        margin = layout_info['margin']
        header_height = layout_info['header_height']
        footer_height = layout_info['footer_height']
        page_height = layout_info['page_height']
        
        # Draw separators between columns
        for i in range(layout_info['columns'] - 1):
            sep_x = (layout_info['column_positions'][i] + layout_info['column_width'] + 
                    layout_info['column_positions'][i + 1]) // 2
            
            draw.line([(sep_x, margin + header_height), 
                      (sep_x, page_height - margin - footer_height)], 
                     fill='lightgray', width=1)
