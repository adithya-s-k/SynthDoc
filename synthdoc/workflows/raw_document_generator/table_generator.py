from PIL import Image, ImageDraw, ImageFont

def create_advanced_table() -> Image.Image:
    """Create publication-quality data tables"""
    # Table data
    headers = ["Product", "Revenue ($M)", "Growth (%)", "Market Share", "Status"]
    data = [
        ["Alpha Platform", "125.4", "+12.3", "23.5%", "Growing"],
        ["Beta Services", "89.7", "+8.1", "18.2%", "Stable"], 
        ["Gamma Solutions", "156.2", "+15.7", "28.9%", "Expanding"],
        ["Delta Analytics", "73.8", "+5.4", "14.1%", "Stable"],
        ["Epsilon Tools", "91.3", "+9.8", "15.3%", "Growing"]
    ]
    
    # Calculate dimensions
    cell_width = 120
    cell_height = 35
    header_height = 45
    table_width = len(headers) * cell_width
    table_height = header_height + len(data) * cell_height
    
    # Create image with some padding
    padding = 20
    img = Image.new('RGB', (table_width + 2*padding, table_height + 2*padding), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load fonts - with fallback
    try:
        header_font = ImageFont.truetype("arial.ttf", 12)
        cell_font = ImageFont.truetype("arial.ttf", 10)
    except:
        try:
            header_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
        except:
            header_font = None
            cell_font = None
    
    # Color scheme
    header_color = '#2E86AB'
    alt_row_color = '#F8F9FA'
    border_color = '#DEE2E6'
    text_color = '#212529'
    
    # Draw table
    start_x, start_y = padding, padding
    
    # Draw headers
    for i, header in enumerate(headers):
        x = start_x + i * cell_width
        y = start_y
        
        # Header background
        draw.rectangle([x, y, x + cell_width, y + header_height], 
                      fill=header_color, outline=border_color, width=1)
        
        # Header text (centered)
        text_bbox = draw.textbbox((0, 0), header, font=header_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (cell_width - text_width) // 2
        text_y = y + (header_height - text_height) // 2
        
        draw.text((text_x, text_y), header, fill='white', font=header_font)
    
    # Draw data rows
    for row_idx, row in enumerate(data):
        y = start_y + header_height + row_idx * cell_height
        
        # Alternate row colors
        row_color = alt_row_color if row_idx % 2 == 1 else 'white'
        
        for col_idx, cell_data in enumerate(row):
            x = start_x + col_idx * cell_width
            
            # Cell background
            draw.rectangle([x, y, x + cell_width, y + cell_height], 
                          fill=row_color, outline=border_color, width=1)
            
            # Cell text
            text_bbox = draw.textbbox((0, 0), str(cell_data), font=cell_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Right align numeric columns (1, 2, 3), left align others
            if col_idx in [1, 2, 3]:
                text_x = x + cell_width - text_width - 8
            else:
                text_x = x + 8
            
            text_y = y + (cell_height - text_height) // 2
            
            # Color coding for status
            if col_idx == 4:  # Status column
                if cell_data == "Growing":
                    text_color_cell = '#28A745'
                elif cell_data == "Expanding":
                    text_color_cell = '#007BFF'
                else:
                    text_color_cell = '#6C757D'
            else:
                text_color_cell = text_color
            
            draw.text((text_x, text_y), str(cell_data), fill=text_color_cell, font=cell_font)
    
    return img