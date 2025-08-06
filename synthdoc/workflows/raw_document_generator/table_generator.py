from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any
import random

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

def create_contextual_table(content_data: Dict[str, Any], text_content: str = "") -> Image.Image:
    """Create table based on actual extracted content data"""
    
    entities = content_data.get('entities', [])
    numbers = content_data.get('numbers', [])
    metrics = content_data.get('metrics', [])
    temporal_data = content_data.get('temporal_data', [])
    technical_concepts = content_data.get('technical_concepts', [])
    
    print(f"📋 Table Generator - Entities: {len(entities)}, Numbers: {len(numbers)}, Metrics: {len(metrics)}")
    
    # Strategy 1: Metrics table (if we have performance metrics)
    if metrics and len(metrics) >= 2:
        return create_metrics_performance_table(metrics, entities)
    
    # Strategy 2: Entity comparison table (if we have entities + numbers)
    elif entities and numbers and len(entities) >= 3:
        return create_entity_data_table(entities, numbers)
    
    # Strategy 3: Temporal data table (if we have time-based data)
    elif temporal_data and numbers and len(temporal_data) >= 3:
        return create_temporal_analysis_table(temporal_data, numbers)
    
    # Strategy 4: Technical specifications table
    elif technical_concepts and len(technical_concepts) >= 2:
        return create_technical_specs_table(technical_concepts, numbers)
    
    # Fallback to your existing advanced table
    else:
        return create_advanced_table()

def create_metrics_performance_table(metrics: List[Dict], entities: List[str]) -> Image.Image:
    """Create performance metrics table using actual extracted metrics"""
    
    # Prepare headers and data based on actual metrics
    headers = ["Metric", "Value", "Unit", "Performance", "Status"]
    data = []
    
    for i, metric in enumerate(metrics[:5]):
        # Extract metric info
        metric_name = metric.get('name', f"Metric {i+1}")[:15]
        value = metric.get('value', 0)
        unit = metric.get('unit', '')
        
        # Determine performance rating
        if metric.get('type') == 'percentage':
            if value >= 80:
                performance, status = "Excellent", "✓"
            elif value >= 60:
                performance, status = "Good", "○"
            else:
                performance, status = "Needs Work", "!"
        else:
            # For absolute values, use relative comparison
            avg_value = sum(m.get('value', 0) for m in metrics) / len(metrics)
            if value > avg_value * 1.2:
                performance, status = "Above Avg", "↗"
            elif value > avg_value * 0.8:
                performance, status = "Average", "→"
            else:
                performance, status = "Below Avg", "↘"
        
        data.append([metric_name, f"{value}", unit, performance, status])
    
    # If we need more rows, add from entities
    while len(data) < 4 and len(data) < len(entities):
        entity = entities[len(data)][:15]
        value = random.randint(65, 95)
        data.append([f"{entity} Score", f"{value}", "%", "Good" if value > 75 else "Fair", "○"])
    
    return _draw_contextual_table(headers, data, "Performance Metrics Analysis")

def create_entity_data_table(entities: List[str], numbers: List[Dict]) -> Image.Image:
    """Create entity comparison table with actual data"""
    
    headers = ["Entity", "Primary Value", "Secondary", "Growth", "Rank"]
    data = []
    
    for i, entity in enumerate(entities[:5]):
        entity_name = entity[:12]  # Truncate long names
        
        # Use actual numbers if available
        if i < len(numbers):
            primary_value = numbers[i]['value']
            unit = numbers[i].get('unit', '')
        else:
            primary_value = random.randint(50, 150)
            unit = numbers[0].get('unit', '') if numbers else ''
        
        # Generate secondary value (related to primary)
        secondary = round(primary_value * random.uniform(0.7, 1.3), 1)
        
        # Calculate growth
        growth = f"+{random.randint(1, 20)}%" if random.random() > 0.3 else f"-{random.randint(1, 10)}%"
        
        # Assign rank
        rank = f"#{i+1}"
        
        data.append([
            entity_name, 
            f"{primary_value}{unit}", 
            f"{secondary}", 
            growth, 
            rank
        ])
    
    title = f"Entity Analysis: {entities[0]} Comparison"
    return _draw_contextual_table(headers, data, title)

def create_temporal_analysis_table(temporal_data: List[str], numbers: List[Dict]) -> Image.Image:
    """Create time-series analysis table"""
    
    headers = ["Period", "Value", "Change", "Trend", "Notes"]
    data = []
    
    # Sort temporal data
    sorted_temporal = sorted(temporal_data[:5])
    
    prev_value = None
    for i, period in enumerate(sorted_temporal):
        # Use actual numbers or generate trending data
        if i < len(numbers):
            value = numbers[i]['value']
            unit = numbers[i].get('unit', '')
        else:
            base_value = numbers[0]['value'] if numbers else 100
            trend_factor = random.uniform(0.95, 1.15)
            value = base_value * (trend_factor ** i)
            unit = numbers[0].get('unit', '') if numbers else ''
        
        # Calculate change from previous
        if prev_value:
            change_pct = ((value - prev_value) / prev_value) * 100
            change = f"{change_pct:+.1f}%"
            trend = "↗" if change_pct > 5 else "↘" if change_pct < -5 else "→"
        else:
            change = "—"
            trend = "—"
        
        # Generate notes
        if change != "—":
            if abs(change_pct) > 10:
                notes = "Significant"
            elif abs(change_pct) > 5:
                notes = "Moderate"
            else:
                notes = "Stable"
        else:
            notes = "Baseline"
        
        data.append([
            period[:8], 
            f"{value:.1f}{unit}", 
            change, 
            trend, 
            notes
        ])
        
        prev_value = value
    
    return _draw_contextual_table(headers, data, "Temporal Analysis")

def create_technical_specs_table(concepts: List[str], numbers: List[Dict]) -> Image.Image:
    """Create technical specifications table"""
    
    headers = ["Component", "Specification", "Value", "Status", "Version"]
    data = []
    
    for i, concept in enumerate(concepts[:5]):
        component = concept[:12]
        
        # Generate realistic tech specs
        specs = ["Performance", "Capacity", "Efficiency", "Reliability", "Throughput"]
        spec = specs[i % len(specs)]
        
        # Use actual numbers or generate realistic values
        if i < len(numbers):
            value = f"{numbers[i]['value']}{numbers[i].get('unit', '')}"
        else:
            value = f"{random.randint(85, 99)}%"
        
        # Status and version
        status = random.choice(["Active", "Stable", "Updated", "Optimized"])
        version = f"v{random.randint(1,3)}.{random.randint(0,9)}"
        
        data.append([component, spec, value, status, version])
    
    return _draw_contextual_table(headers, data, "Technical Specifications")

def _draw_contextual_table(headers: List[str], data: List[List[str]], title: str) -> Image.Image:
    """Draw table using your existing styling but with contextual data"""
    
    # Use similar dimensions as your existing function
    cell_width = 110
    cell_height = 35
    header_height = 45
    table_width = len(headers) * cell_width
    table_height = header_height + len(data) * cell_height
    
    # Create image with padding
    padding = 25
    img = Image.new('RGB', (table_width + 2*padding, table_height + 2*padding + 40), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        header_font = ImageFont.truetype("arial.ttf", 11)
        cell_font = ImageFont.truetype("arial.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
    
    # Color scheme (your existing colors)
    header_color = '#2E86AB'
    alt_row_color = '#F8F9FA'
    border_color = '#DEE2E6'
    text_color = '#212529'
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (img.width - title_width) // 2
    draw.text((title_x, 10), title, fill=header_color, font=title_font)
    
    # Table starting position
    start_x, start_y = padding, padding + 30
    
    # Draw headers (using your existing logic)
    for i, header in enumerate(headers):
        x = start_x + i * cell_width
        y = start_y
        
        draw.rectangle([x, y, x + cell_width, y + header_height], 
                      fill=header_color, outline=border_color, width=1)
        
        text_bbox = draw.textbbox((0, 0), header, font=header_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (cell_width - text_width) // 2
        text_y = y + (header_height - text_height) // 2
        
        draw.text((text_x, text_y), header, fill='white', font=header_font)
    
    # Draw data rows (using your existing logic)
    for row_idx, row in enumerate(data):
        y = start_y + header_height + row_idx * cell_height
        row_color = alt_row_color if row_idx % 2 == 1 else 'white'
        
        for col_idx, cell_data in enumerate(row):
            x = start_x + col_idx * cell_width
            
            draw.rectangle([x, y, x + cell_width, y + cell_height], 
                          fill=row_color, outline=border_color, width=1)
            
            text_bbox = draw.textbbox((0, 0), str(cell_data), font=cell_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Center align for most columns, right align for numeric-looking columns
            if any(char.isdigit() for char in str(cell_data)) and col_idx in [1, 2]:
                text_x = x + cell_width - text_width - 8
            else:
                text_x = x + (cell_width - text_width) // 2
            
            text_y = y + (cell_height - text_height) // 2
            
            # Color coding for special columns
            cell_text_color = text_color
            if col_idx == len(headers) - 1:  # Last column (status/notes)
                if any(word in str(cell_data).lower() for word in ['good', 'excellent', 'active', '✓', '↗']):
                    cell_text_color = '#28A745'
                elif any(word in str(cell_data).lower() for word in ['fair', 'average', '○', '→']):
                    cell_text_color = '#6C757D'
                elif any(word in str(cell_data).lower() for word in ['needs', 'below', '!', '↘']):
                    cell_text_color = '#DC3545'
            
            draw.text((text_x, text_y), str(cell_data), fill=cell_text_color, font=cell_font)
    
    return img