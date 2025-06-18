import random 
from PIL import Image, ImageDraw
import io 
from config import DocumentConfig
import  numpy as np
def generate_sample_data_for_graph(graph_type: str = "line"):
    """Generate contextual graph data based on content"""
    graph_types = [
        "line", "bar", "pie", "scatter", "area", "box", "heatmap", 
        "histogram", "radar", "treemap", "funnel", "waterfall"
    ]
    
    if not graph_type:
        graph_type = random.choice(graph_types[:6])  # Use simpler types for now
    
    # Different data patterns
    if graph_type == "line":
        return generate_trend_data(), graph_type
    elif graph_type == "bar":
        return generate_category_data(), graph_type
    elif graph_type == "pie":
        return generate_distribution_data(), graph_type
    elif graph_type == "scatter":
        return generate_correlation_data(), graph_type
    elif graph_type == "area":
        return generate_stacked_data(), graph_type
    elif graph_type == "heatmap":
        return generate_matrix_data(), graph_type
    else:
        return generate_trend_data(), "line"

def generate_trend_data() :
    """Generate trend/time series data"""
    years = list(range(2020, 2025))
    values = [random.randint(50, 200) + i*10 for i in range(len(years))]  # Upward trend
    return {"x": years, "y": values, "title": "Growth Trend Analysis 2020-2024"}

def generate_category_data() :
    """Generate categorical data"""
    categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    values = [random.randint(20, 100) for _ in categories]
    return {"categories": categories, "values": values, "title": "Performance Comparison by Category"}

def generate_distribution_data():
    """Generate distribution data"""
    labels = ["Segment 1", "Segment 2", "Segment 3", "Segment 4"]
    sizes = [random.randint(15, 40) for _ in labels]
    return {"labels": labels, "sizes": sizes, "title": "Market Share Distribution"}

def generate_correlation_data():
    """Generate scatter plot data"""
    x = [random.uniform(10, 100) for _ in range(50)]
    y = [val + random.uniform(-20, 20) for val in x]  # Correlated with noise
    return {"x": x, "y": y, "x_label": "Input Variable", "y_label": "Output Variable", "title": "Correlation Analysis"}

def generate_stacked_data():
    """Generate area/stacked data"""
    x = list(range(2020, 2025))
    y1 = [random.randint(20, 50) for _ in x]
    y2 = [random.randint(30, 60) for _ in x]
    return {"x": x, "y1": y1, "y2": y2, "title": "Stacked Area Analysis"}

def generate_matrix_data():
    """Generate heatmap matrix data"""
    matrix = np.random.rand(5, 5) * 100
    return {"matrix": matrix, "title": "Performance Heatmap"}

def create_advanced_graph (graph_data, graph_type: str):
        """Create diverse graph types"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
        
        # Color palettes
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
        
        if graph_type == "heatmap":
            if graph_type == "heatmap":  # seaborn available
                try:
                    import seaborn as sns
                    sns.heatmap(graph_data["matrix"], annot=True, cmap='viridis', ax=ax, fmt='.1f')
                except ImportError:
                    # Fallback to matplotlib imshow
                    im = ax.imshow(graph_data["matrix"], cmap='viridis', aspect='auto')
                    plt.colorbar(im)
            else:
                im = ax.imshow(graph_data["matrix"], cmap='viridis', aspect='auto')
                plt.colorbar(im)
                
        elif graph_type == "scatter":
            ax.scatter(graph_data["x"], graph_data["y"], alpha=0.6, c=colors[0], s=30, edgecolors='white')
            ax.set_xlabel(graph_data["x_label"])
            ax.set_ylabel(graph_data["y_label"])
            
        elif graph_type == "area":
            ax.fill_between(graph_data["x"], graph_data["y1"], alpha=0.7, color=colors[0], label="Series 1")
            ax.fill_between(graph_data["x"], graph_data["y1"], 
                          [y1 + y2 for y1, y2 in zip(graph_data["y1"], graph_data["y2"])], 
                          alpha=0.7, color=colors[1], label="Series 2")
            ax.legend()
            
        elif graph_type == "line":
            ax.plot(graph_data["x"], graph_data["y"], marker='o', linewidth=2.5, color=colors[0], 
                   markersize=6, markerfacecolor='white', markeredgecolor=colors[0], markeredgewidth=2)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            
        elif graph_type == "bar":
            bars = ax.bar(graph_data["categories"], graph_data["values"], color=colors[:len(graph_data["categories"])])
            ax.set_ylabel("Value")
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}', ha='center', va='bottom')
                       
        elif graph_type == "pie":
            wedges, texts, autotexts = ax.pie(graph_data["sizes"], labels=graph_data["labels"], 
                                             autopct='%1.1f%%', startangle=90, colors=colors)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            ax.axis('equal')
        
        ax.set_title(graph_data["title"], fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        buf.seek(0)
        graph_img = Image.open(buf)
        plt.close(fig)
        
        return graph_img


def create_graph_image (graph_data: dict, graph_type: str = "line", width: int = 400, height: int = 300):
        """Create a graph as PIL Image - now uses advanced graph creation"""
        return create_advanced_graph(graph_data, graph_type)



def generate_table_data  (language: str = "en"):
    """Generate sample table data"""
    if language == "hi":
        headers = ["क्रम", "नाम", "मूल्य", "स्थिति"]
        rows = [
            ["1", "उत्पाद A", "₹1,200", "उपलब्ध"],
            ["2", "उत्पाद B", "₹850", "स्टॉक में नहीं"],
            ["3", "उत्पाद C", "₹2,100", "उपलब्ध"],
            ["4", "उत्पाद D", "₹750", "उपलब्ध"]
        ]
    else:
        headers = ["ID", "Product", "Price", "Status"]
        rows = [
            ["1", "Product A", "$120", "Available"],
            ["2", "Product B", "$85", "Out of Stock"],
            ["3", "Product C", "$210", "Available"],
            ["4", "Product D", "$75", "Available"]
        ]
    
    return {"headers": headers, "rows": rows}


def generate_diverse_table_data  (content_type: str = "general", language: str = "en"):
        """Generate different table types"""
        table_types = ["comparison", "timeline", "statistics", "pricing", "features"]
        table_type = random.choice(table_types)
        
        table_generators = {
            "comparison": generate_comparison_table(language),
            "timeline": generate_timeline_table(language),
            "statistics": generate_stats_table(language),
            "pricing": generate_pricing_table(language),
            "features": generate_features_table(language)
        }
        
        return table_generators[table_type](language)

def generate_comparison_table  (language: str):
    """Generate comparison table"""
    if language == "hi":
        headers = ["विशेषता", "विकल्प A", "विकल्प B", "विकल्प C"]
        rows = [
            ["प्रदर्शन", "उच्च", "मध्यम", "निम्न"],
            ["मूल्य", "₹5,000", "₹3,000", "₹1,500"],
            ["गुणवत्ता", "उत्कृष्ट", "अच्छी", "औसत"],
            ["सहायता", "24/7", "व्यावसायिक समय", "ईमेल केवल"]
        ]
    else:
        headers = ["Feature", "Option A", "Option B", "Option C"]
        rows = [
            ["Performance", "High", "Medium", "Low"],
            ["Price", "$500", "$300", "$150"],
            ["Quality", "Excellent", "Good", "Average"],
            ["Support", "24/7", "Business Hours", "Email Only"]
        ]
    return {"headers": headers, "rows": rows}

def generate_timeline_table (language: str):
    """Generate timeline table"""
    if language == "hi":
        headers = ["चरण", "समयावधि", "गतिविधि", "परिणाम"]
        rows = [
            ["1", "जनवरी 2024", "योजना निर्माण", "पूर्ण"],
            ["2", "फरवरी 2024", "विकास शुरू", "चल रहा"],
            ["3", "मार्च 2024", "परीक्षण चरण", "लंबित"],
            ["4", "अप्रैल 2024", "तैनाती", "नियोजित"]
        ]
    else:
        headers = ["Phase", "Timeline", "Activity", "Status"]
        rows = [
            ["1", "Jan 2024", "Planning", "Complete"],
            ["2", "Feb 2024", "Development", "In Progress"],
            ["3", "Mar 2024", "Testing", "Pending"],
            ["4", "Apr 2024", "Deployment", "Planned"]
        ]
    return {"headers": headers, "rows": rows}

def generate_stats_table  (language: str):
    """Generate statistics table"""
    if language == "hi":
        headers = ["मेट्रिक", "Q1", "Q2", "Q3", "Q4"]
        rows = [
            ["बिक्री", "₹2.5L", "₹3.2L", "₹2.8L", "₹4.1L"],
            ["ग्राहक", "150", "180", "165", "220"],
            ["वृद्धि %", "12%", "15%", "8%", "18%"],
            ["संतुष्टि", "85%", "88%", "82%", "91%"]
        ]
    else:
        headers = ["Metric", "Q1", "Q2", "Q3", "Q4"]
        rows = [
            ["Revenue", "$25K", "$32K", "$28K", "$41K"],
            ["Customers", "150", "180", "165", "220"],
            ["Growth %", "12%", "15%", "8%", "18%"],
            ["Satisfaction", "85%", "88%", "82%", "91%"]
        ]
    return {"headers": headers, "rows": rows}

def generate_pricing_table  (language: str):
    """Generate pricing table"""
    if language == "hi":
        headers = ["योजना", "मूल्य", "सुविधाएं", "सहायता"]
        rows = [
            ["बेसिक", "₹999/माह", "5 उपयोगकर्ता", "ईमेल"],
            ["प्रो", "₹2999/माह", "25 उपयोगकर्ता", "फोन"],
            ["एंटरप्राइज", "₹9999/माह", "असीमित", "समर्पित"]
        ]
    else:
        headers = ["Plan", "Price", "Features", "Support"]
        rows = [
            ["Basic", "$99/month", "5 Users", "Email"],
            ["Pro", "$299/month", "25 Users", "Phone"],
            ["Enterprise", "$999/month", "Unlimited", "Dedicated"]
        ]
    return {"headers": headers, "rows": rows}

def generate_features_table  (language: str):
    """Generate features table"""
    if language == "hi":
        headers = ["सुविधा", "विवरण", "उपलब्धता", "रेटिंग"]
        rows = [
            ["सुरक्षा", "एंड-टू-एंड एन्क्रिप्शन", "सभी योजनाओं में", "⭐⭐⭐⭐⭐"],
            ["बैकअप", "स्वचालित दैनिक बैकअप", "प्रो और उपर", "⭐⭐⭐⭐"],
            ["एनालिटिक्स", "रियल-टाइम डैशबोर्ड", "एंटरप्राइज", "⭐⭐⭐⭐⭐"],
            ["मोबाइल ऐप", "iOS और Android", "सभी योजनाओं में", "⭐⭐⭐⭐"]
        ]
    else:
        headers = ["Feature", "Description", "Availability", "Rating"]
        rows = [
            ["Security", "End-to-end encryption", "All plans", "⭐⭐⭐⭐⭐"],
            ["Backup", "Automated daily backups", "Pro and above", "⭐⭐⭐⭐"],
            ["Analytics", "Real-time dashboard", "Enterprise only", "⭐⭐⭐⭐⭐"],
            ["Mobile App", "iOS and Android", "All plans", "⭐⭐⭐⭐"]
        ]
    return {"headers": headers, "rows": rows}

def create_table_image  (table_data: dict, width: int = 500, height: int = 200):
    """Create a table as PIL Image with Hindi support"""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load Hindi-capable font
    from font import load_font
    font = load_font(12)
    
    # Table dimensions
    margin = 20
    col_width = (width - 2 * margin) // len(table_data["headers"])
    row_height = 30  # Increased for Hindi text
    
    # Draw table border
    draw.rectangle([margin, margin, width-margin, height-margin], outline='black', width=2)
    
    # Draw headers
    y = margin
    for i, header in enumerate(table_data["headers"]):
        x = margin + i * col_width
        # Header background
        draw.rectangle([x, y, x + col_width, y + row_height], fill='lightgray', outline='black')
        # Header text with better positioning
        try:
            # Center text vertically
            text_y = y + (row_height - 16) // 2
            draw.text((x + 5, text_y), str(header), fill='black', font=font)
        except Exception as e:
            print(f"⚠️ Header rendering error: {e}")
            draw.text((x + 5, y + 5), "[Header Error]", fill='red', font=font)
    
    # Draw rows
    y += row_height
    for row_idx, row in enumerate(table_data["rows"]):
        for i, cell in enumerate(row):
            x = margin + i * col_width
            # Cell border
            draw.rectangle([x, y, x + col_width, y + row_height], outline='black')
            # Cell text
            try:
                text_y = y + (row_height - 16) // 2
                draw.text((x + 5, text_y), str(cell), fill='black', font=font)
            except Exception as e:
                print(f"⚠️ Cell rendering error at row {row_idx}, col {i}: {e}")
                draw.text((x + 5, y + 5), "[Error]", fill='red', font=font)
        y += row_height
    
    return img


def text_to_markdown  (text: str, config: DocumentConfig, page_num: int = 0) -> str:
        """Convert text content to markdown format"""
        lines = text.split('\n')
        markdown_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                markdown_content.append("")
                continue
                
            # First line is usually title (make it H1)
            if i == 0 and not line.startswith('#'):
                markdown_content.append(f"# {line}")
            # Page headers
            elif line.startswith(f"Page {page_num + 1}"):
                markdown_content.append(f"## {line}")
            # Section headers (lines that end with colon or are short and capitalized)
            elif (line.endswith(':') or 
                (len(line.split()) <= 4 and line.istitle() and 
                not any(word in line.lower() for word in ['the', 'a', 'an', 'and', 'or']))):
                markdown_content.append(f"### {line}")
            # Regular paragraphs
            else:
                markdown_content.append(line)
        
        # Add page metadata at the end
        markdown_content.extend([
            "",
            "---",
            f"*Page {page_num + 1} | Language: {config.language} | Generated Document*"
        ])
        
        return '\n'.join(markdown_content)

def text_to_html(text: str, config: DocumentConfig, page_num: int = 0) -> str:
    """Convert text content to HTML format"""
    lines = text.split('\n')
    html_content = []
    
    # HTML document start
    html_content.extend([
        '<!DOCTYPE html>',
        '<html lang="{}">'.format(config.language),
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'    <title>Document Page {page_num + 1}</title>',
        '    <style>',
        '        body { font-family: Georgia, serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }',
        '        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }',
        '        h2 { color: #34495e; margin-top: 30px; }',
        '        h3 { color: #7f8c8d; }',
        '        p { margin: 15px 0; text-align: justify; }',
        '        .page-meta { font-style: italic; color: #95a5a6; border-top: 1px solid #ecf0f1; padding-top: 20px; margin-top: 30px; }',
        '    </style>',
        '</head>',
        '<body>'
    ])
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # First line is usually title
        if i == 0 and not line.startswith('<'):
            html_content.append(f'    <h1>{line}</h1>')
        # Page headers
        elif line.startswith(f"Page {page_num + 1}"):
            html_content.append(f'    <h2>{line}</h2>')
        # Section headers
        elif (line.endswith(':') or 
            (len(line.split()) <= 4 and line.istitle() and 
            not any(word in line.lower() for word in ['the', 'a', 'an', 'and', 'or']))):
            html_content.append(f'    <h3>{line}</h3>')
        # Regular paragraphs
        else:
            html_content.append(f'    <p>{line}</p>')
    
    # Add page metadata and close HTML
    html_content.extend([
        f'    <div class="page-meta">Page {page_num + 1} | Language: {config.language} | Generated Document</div>',
        '</body>',
        '</html>'
    ])
    
    return '\n'.join(html_content)
