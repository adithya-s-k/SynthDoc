import random
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple 
from ...languages import load_language_font
from .chart_generator import create_advanced_chart
from .table_generator import create_advanced_table  
from .ai_image_generator import generate_contextual_image
from .text_utils import wrap_text_to_width
# from .chart_generator import create_advanced_chart as create_simple_chart
# from .table_generator import create_advanced_table as create_simple_table

def create_multicolumn_document_image(text: str, config, page_num: int = 0, 
                                    used_graph_types: set = None, used_table_types: set = None,
                                    page_width: int = 800, page_height: int = 1000) -> Tuple[Image.Image, dict]:
    """Create document image with proper multi-column layouts"""
    
    if used_graph_types is None:
        used_graph_types = set()
    if used_table_types is None:
        used_table_types = set()
    
    # Create image
    img = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load language-appropriate fonts
    try:
        language_code = config.language.value if hasattr(config.language, 'value') else str(config.language)
        
       
        # Replace the massive font loading block with:
        font = load_language_font(language_code, 12)
        title_font = load_language_font(language_code, 16) 
        header_font = load_language_font(language_code, 20)



    except Exception as e:
        print(f"Font not loading; error: {e}")
        try:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        except:
            font = None
            title_font = None
            header_font = None
    
    # Layout calculations
    margin = 60
    header_height = 80
    footer_height = 60
    content_start_y = margin + header_height
    content_height = page_height - margin - footer_height - content_start_y
    
    # Determine layout type
    layout_type = getattr(config, 'layout_type', 'SINGLE_COLUMN')
    if hasattr(layout_type, 'value'):
        layout_type = layout_type.value
    
    # Calculate columns based on layout type
    if layout_type == "TWO_COLUMN":
        num_columns = 2
        column_gap = 30
        column_width = (page_width - 2 * margin - column_gap) // 2
    elif layout_type == "THREE_COLUMN":
        num_columns = 3
        column_gap = 20
        column_width = (page_width - 2 * margin - 2 * column_gap) // 3
    elif layout_type == "FOUR_COLUMN":
        num_columns = 4
        column_gap = 15
        column_width = (page_width - 2 * margin - 3 * column_gap) // 4
    elif layout_type == "NEWSLETTER":
        num_columns = 2
        column_gap = 40
        column_width = (page_width - 2 * margin - column_gap) // 2
    elif layout_type == "MAGAZINE":
        # Magazine style: 3 columns with wider gaps
        num_columns = 3
        column_gap = 25
        column_width = (page_width - 2 * margin - 2 * column_gap) // 3
    elif layout_type == "ACADEMIC":
        # Academic style: 2 columns with narrow gaps
        num_columns = 2
        column_gap = 20
        column_width = (page_width - 2 * margin - column_gap) // 2
    elif layout_type == "NEWSPAPER":
        # Newspaper style: 4 narrow columns
        num_columns = 4
        column_gap = 12
        column_width = (page_width - 2 * margin - 3 * column_gap) // 4
    elif layout_type == "BROCHURE":
        # Brochure style: 3 columns with medium gaps
        num_columns = 3
        column_gap = 18
        column_width = (page_width - 2 * margin - 2 * column_gap) // 3
    elif layout_type == "REPORT":
        # Report style: single column with wider margins
        num_columns = 1
        column_gap = 0
        margin = 80  # Wider margins for reports
        column_width = page_width - 2 * margin
    else:  # SINGLE_COLUMN
        num_columns = 1
        column_gap = 0
        column_width = page_width - 2 * margin
    
    # Draw header
    if header_font and hasattr(config, 'prompt') and config.prompt:
        draw.text((margin, 20), config.prompt[:50], fill='black', font=header_font)
        draw.line([(margin, 50), (page_width - margin, 50)], fill='black', width=2)
    
    # Draw footer
    if font:
        footer_text = f"Page {page_num + 1}"
        if hasattr(config, 'language'):
            # Handle both string and Language enum
            if hasattr(config.language, 'value'):
                footer_text += f" | {config.language.value.upper()}"
            else:
                footer_text += f" | {str(config.language).upper()}"
        draw.text((margin, page_height - 40), footer_text, fill='gray', font=font)
    
    current_y = content_start_y
    visual_elements = {'graphs': [], 'tables': [], 'images': [], 'equations': []}
    word_coords = []
    
    # Add visual elements strategically - GUARANTEE 1-2 per page with LOTS OF TEXT
    visual_elements_height = 0
    visual_elements_added = 0
    max_visual_elements = 2  # Limit to max 2 visual elements per page
    
    # Add AI-generated image - ALWAYS add one per page if enabled
    include_image = getattr(config, 'include_ai_images', False)
    if include_image and visual_elements_added < max_visual_elements:
        try:
            # Handle Language enum properly
            language_code = getattr(config, 'language', 'en')
            if hasattr(language_code, 'value'):
                language_code = language_code.value
            ai_image = generate_contextual_image(text, language_code)
            if ai_image:
                # Make image smaller to leave MORE room for text
                max_width = min(250, column_width)
                if ai_image.width > max_width:
                    aspect_ratio = ai_image.height / ai_image.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    ai_image = ai_image.resize((target_width, target_height))
                
                # Position image
                img_x = (page_width - ai_image.width) // 2
                img.paste(ai_image, (img_x, current_y))
                current_y += ai_image.height + 15
                visual_elements_height += ai_image.height + 15
                visual_elements_added += 1
                
                visual_elements['images'].append({
                    'type': 'ai_generated',
                    'title': f"Figure {page_num + 1}",
                    'position': {'x': img_x, 'y': current_y - ai_image.height - 15},
                    'size': {'width': ai_image.width, 'height': ai_image.height},
                    'description': f"Illustration for page {page_num + 1}"
                })
                
                # Small caption
                if font:
                    caption = f"Figure {page_num + 1}"
                    draw.text((img_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
        except Exception as e:
            print(f"⚠️ Error adding AI image: {e}")
    
    # Add graph or table - add one more if we have room
    if visual_elements_added < max_visual_elements:
        # Alternate between graphs and tables
        if page_num % 2 == 0 and getattr(config, 'include_graphs', False):
            # Add graph on even pages
            try:
                available_graph_types = ["line", "bar", "pie"]
                available_types = [t for t in available_graph_types if t not in used_graph_types]
                
                if not available_types:
                    available_types = available_graph_types
                    used_graph_types.clear()
                
                graph_type = random.choice(available_types)
                used_graph_types.add(graph_type)
                
                graph_img = create_advanced_chart(graph_type)
                
                # Make graph smaller to leave MORE room for text
                max_width = min(300, column_width)
                if graph_img.width > max_width:
                    aspect_ratio = graph_img.height / graph_img.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    graph_img = graph_img.resize((target_width, target_height))
                
                # Position graph
                graph_x = (page_width - graph_img.width) // 2
                img.paste(graph_img, (graph_x, current_y))
                current_y += graph_img.height + 15
                visual_elements_height += graph_img.height + 15
                visual_elements_added += 1
                
                visual_elements['graphs'].append({
                    'type': graph_type,
                    'title': f'{graph_type.title()} Chart',
                    'position': {'x': graph_x, 'y': current_y - graph_img.height - 15},
                    'size': {'width': graph_img.width, 'height': graph_img.height},
                    'description': f"Data visualization chart"
                })
                
                # Small caption
                if font:
                    caption = f"Chart {page_num + 1}: {graph_type.title()}"
                    draw.text((graph_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
            except Exception as e:
                print(f"⚠️ Error adding graph: {e}")
                
        elif page_num % 2 == 1 and getattr(config, 'include_tables', False):
            # Add table on odd pages
            try:
                table_img = create_advanced_table()
                
                # Make table smaller to leave MORE room for text
                max_width = min(350, column_width)
                if table_img.width > max_width:
                    aspect_ratio = table_img.height / table_img.width
                    target_width = max_width
                    target_height = int(target_width * aspect_ratio)
                    table_img = table_img.resize((target_width, target_height))
                
                # Position table
                table_x = (page_width - table_img.width) // 2
                img.paste(table_img, (table_x, current_y))
                current_y += table_img.height + 15
                visual_elements_height += table_img.height + 15
                visual_elements_added += 1
                
                visual_elements['tables'].append({
                    'type': 'data_table',
                    'title': 'Data Table',
                    'position': {'x': table_x, 'y': current_y - table_img.height - 15},
                    'size': {'width': table_img.width, 'height': table_img.height},
                    'rows': 4,
                    'columns': 3,
                    'description': "Data overview table"
                })
                
                # Small caption
                if font:
                    caption = f"Table {page_num + 1}: Data"
                    draw.text((table_x, current_y), caption, fill='black', font=font)
                    current_y += 15
                    visual_elements_height += 15
            except Exception as e:
                print(f"⚠️ Error adding table: {e}")
    
    print(f"📊 Page {page_num}: Added {visual_elements_added} visual elements, focusing on text content")

    # Add text content in columns - DENSE TEXT LAYOUT
    remaining_height = content_height - visual_elements_height
    if remaining_height > 50:
        # Split text into columns
        words = text.split()
        words_per_column = len(words) // num_columns if num_columns > 1 else len(words)
        
        for col in range(num_columns):
            start_idx = col * words_per_column
            end_idx = start_idx + words_per_column if col < num_columns - 1 else len(words)
            
            if start_idx < len(words):
                column_text = ' '.join(words[start_idx:end_idx])
                
                # Calculate column position
                col_x = margin + col * (column_width + column_gap)
                
                # Add text with word wrapping - MUCH DENSER
                y_offset = current_y
                lines = wrap_text_to_width(column_text, column_width, font, draw)
                
                line_height = 16  # Reduced from 20 to 16 for denser text
                for i, line in enumerate(lines):
                    line_y = y_offset + i * line_height
                    if line_y + line_height < page_height - footer_height:
                        if font:
                            draw.text((col_x, line_y), line, fill='black', font=font)
                        
                        # Track word coordinates
                        x_offset = col_x
                        for word in line.split():
                            if font:
                                try:
                                    word_bbox = draw.textbbox((0, 0), word, font=font)
                                    word_width = word_bbox[2] - word_bbox[0]
                                except:
                                    word_width = len(word) * 7  # Reduced from 8 to 7
                            else:
                                word_width = len(word) * 7
                            
                            word_coords.append({
                                'word': word,
                                'x': x_offset,
                                'y': line_y,
                                'width': word_width,
                                'height': line_height,
                                'page': page_num,
                                'column': col
                            })
                            x_offset += word_width + 6  # Reduced spacing from 8 to 6
                    else:
                        break
    
    # Prepare metadata
    metadata = {
        'page_num': page_num,
        'layout_type': layout_type,
        'num_columns': num_columns,
        'visual_elements': visual_elements,
        'word_coords': word_coords,
        'layout_info': {
            'margin': margin,
            'header_height': header_height,
            'footer_height': footer_height,
            'column_width': column_width,
            'column_gap': column_gap
        },
        'content_height': content_height,
        'visual_elements_height': visual_elements_height,
        'page_dimensions': {'width': page_width, 'height': page_height}
    }
    
    return img, metadata