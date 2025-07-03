import matplotlib.pyplot as plt
import random
from PIL import Image
from typing import Dict, List, Any
import io 

def create_advanced_chart(chart_type: str = "bar") -> Image.Image:
    """Create advanced, publication-quality charts"""
    # Set professional styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    
    # Professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5A189A', '#2D6A4F']
    
    if chart_type == "bar":
        categories = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024"]
        values = [random.randint(75, 125) for _ in range(5)]
        bars = ax.bar(categories, values, color=colors[:5], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title("Quarterly Performance Analysis", fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Performance Index", fontsize=11)
        ax.set_xlabel("Quarter", fontsize=11)
        
    elif chart_type == "line":
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        revenue = [random.randint(80, 120) + i*5 for i in range(6)]
        costs = [random.randint(60, 90) + i*3 for i in range(6)]
        
        ax.plot(months, revenue, marker='o', linewidth=3, label='Revenue', color=colors[0])
        ax.plot(months, costs, marker='s', linewidth=3, label='Costs', color=colors[1])
        
        ax.fill_between(months, revenue, alpha=0.3, color=colors[0])
        ax.fill_between(months, costs, alpha=0.3, color=colors[1])
        
        ax.set_title("Revenue vs Costs Trend", fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Amount ($K)", fontsize=11)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
    elif chart_type == "pie":
        labels = ["Technology", "Healthcare", "Finance", "Education", "Other"]
        sizes = [30, 25, 20, 15, 10]
        explode = (0.05, 0, 0, 0, 0)  # Slightly separate first slice
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         startangle=90, colors=colors[:5], 
                                         explode=explode, shadow=True)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title("Market Segment Distribution", fontsize=14, fontweight='bold', pad=20)
    
    # Common styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if chart_type != "pie":
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def create_contextual_chart(content_data: Dict[str, Any], text_content: str = "") -> Image.Image:
    """
    Create chart based on actual extracted data from content analysis.
    Selects appropriate chart type based on available data structures.
    
    Args:
        content_data: Dictionary containing extracted entities, numbers, temporal data, etc.
        text_content: Original text content for fallback analysis
        
    Returns:
        PIL Image containing the generated chart
    """
    numbers = content_data.get('numbers', [])
    entities = content_data.get('entities', [])
    temporal_data = content_data.get('temporal_data', [])
    technical_concepts = content_data.get('technical_concepts', [])
    
    print(f"Chart Generator - Numbers: {len(numbers)}, Entities: {len(entities)}, Temporal: {len(temporal_data)}")
    
    # Strategy 1: Time series chart for temporal data with numbers
    if temporal_data and numbers and len(temporal_data) >= 3:
        return create_timeline_chart(temporal_data, numbers, technical_concepts)
    
    # Strategy 2: Comparison bar chart for entities with numbers
    elif entities and numbers and len(entities) >= 3:
        return create_entity_comparison_chart(entities, numbers)
    
    # Strategy 3: Data distribution chart for multiple numbers
    elif numbers and len(numbers) >= 4:
        return create_data_distribution_chart(numbers, entities)
    
    # Strategy 4: Entity breakdown pie chart
    elif entities and len(entities) >= 3:
        return create_entity_pie_chart(entities)
    
    # Fallback to existing random chart generation
    else:
        chart_types = ["bar", "line", "pie"]
        return create_advanced_chart(random.choice(chart_types))

def create_timeline_chart(temporal_data: List[str], numbers: List[Dict], concepts: List[str]) -> Image.Image:
    """
    Create time-series line chart using actual temporal data and numbers.
    Shows trends over time periods found in the content.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    
    # Sort temporal data chronologically and limit to 6 points
    sorted_temporal = sorted(temporal_data[:6])
    
    # Extract values from numbers data or generate realistic trending data
    values = []
    for i, period in enumerate(sorted_temporal):
        if i < len(numbers):
            value = numbers[i]['value']
            # Normalize very large or very small values for better visualization
            if value > 1000:
                value = value / 1000
            elif value < 1:
                value = value * 100
        else:
            # Generate realistic trending data based on first available value
            base_value = numbers[0]['value'] if numbers else 75
            trend_factor = random.uniform(0.95, 1.15)  # Plus or minus 15% variance
            value = base_value * (trend_factor ** i)
        values.append(round(value, 1))
    
    # Create line chart with professional styling
    line_color = '#2E86AB'
    ax.plot(sorted_temporal, values, marker='o', linewidth=3, 
           markersize=8, color=line_color, markerfacecolor='white', 
           markeredgewidth=2, markeredgecolor=line_color)
    
    # Add area fill for visual appeal
    ax.fill_between(sorted_temporal, values, alpha=0.3, color=line_color)
    
    # Add value labels on data points
    for i, (period, value) in enumerate(zip(sorted_temporal, values)):
        ax.annotate(f'{value}', (i, value), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold')
    
    # Set title based on content concepts
    concept_title = concepts[0] if concepts else "Performance"
    unit = numbers[0]['unit'] if numbers else ''
    ax.set_title(f"{concept_title} Trend Over Time", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(f"Value {unit}", fontsize=11)
    ax.set_xlabel("Time Period", fontsize=11)
    
    # Apply professional styling
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return _convert_matplotlib_to_pil(fig)

def create_entity_comparison_chart(entities: List[str], numbers: List[Dict]) -> Image.Image:
    """
    Create horizontal bar chart comparing entities with their associated values.
    Uses actual entity names and numeric data from content.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    
    # Prepare data with entity names and values
    chart_entities = [e[:12] for e in entities[:5]]  # Truncate long names for display
    values = []
    
    for i, entity in enumerate(chart_entities):
        if i < len(numbers):
            value = numbers[i]['value']
        else:
            # Generate related values with controlled variance
            base_value = numbers[0]['value'] if numbers else 75
            variance = random.uniform(0.7, 1.4)
            value = base_value * variance
        values.append(round(value, 1))
    
    # Use color gradient for visual appeal
    colors = ['#2E86AB', '#3E96BB', '#4EA6CB', '#5EB6DB', '#6EC6EB']
    
    # Create horizontal bar chart for better label readability
    bars = ax.barh(chart_entities, values, color=colors[:len(chart_entities)], alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
               f'{value}', ha='left', va='center', fontweight='bold')
    
    # Set labels and title
    unit = numbers[0]['unit'] if numbers else ''
    ax.set_title(f"Comparison: {entities[0]} vs Others", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f"Value {unit}", fontsize=11)
    
    # Clean up chart appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return _convert_matplotlib_to_pil(fig)

def create_data_distribution_chart(numbers: List[Dict], entities: List[str]) -> Image.Image:
    """
    Create chart showing distribution of numerical values.
    Chooses between bar chart and line chart based on value spread.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    
    values = [num['value'] for num in numbers[:6]]
    labels = []
    
    # Create meaningful labels from context or entities
    for i, num in enumerate(numbers[:6]):
        if num.get('context') and num['context'] != 'extracted_number':
            labels.append(num['context'][:10])
        elif i < len(entities):
            labels.append(entities[i][:10])
        else:
            labels.append(f"Data {i+1}")
    
    # Choose chart type based on value distribution
    value_range = max(values) - min(values)
    
    if value_range > 50:  # Wide range suggests bar chart is better
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5A189A', '#2D6A4F']
        bars = ax.bar(labels, values, color=colors[:len(values)], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel("Values", fontsize=11)
        plt.xticks(rotation=45)
    else:  # Narrow range suggests line chart is better
        ax.plot(labels, values, marker='o', linewidth=3, markersize=8, 
               color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(range(len(labels)), values, alpha=0.3, color='#2E86AB')
        
        # Add value labels on points
        for i, value in enumerate(values):
            ax.annotate(f'{value}', (i, value), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
    
    # Set title with unit information
    unit = numbers[0]['unit'] if numbers else ''
    ax.set_title(f"Data Distribution Analysis {unit}", fontsize=14, fontweight='bold', pad=20)
    
    # Apply clean styling
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return _convert_matplotlib_to_pil(fig)

def create_entity_pie_chart(entities: List[str]) -> Image.Image:
    """
    Create pie chart showing distribution among entities.
    Generates realistic proportions for entity representation.
    """
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    
    # Prepare entity names and generate realistic distribution
    chart_entities = [e[:15] for e in entities[:5]]
    
    # Generate realistic distribution values that sum to 100
    sizes = []
    total = 100
    for i in range(len(chart_entities)-1):
        size = random.randint(10, total//2)
        sizes.append(size)
        total -= size
    sizes.append(max(5, total))  # Ensure last slice is at least 5%
    
    # Professional color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5A189A']
    
    # Explode the first slice for emphasis
    explode = [0.05 if i == 0 else 0 for i in range(len(chart_entities))]
    
    # Create pie chart with professional styling
    wedges, texts, autotexts = ax.pie(sizes, labels=chart_entities, autopct='%1.1f%%',
                                     startangle=90, colors=colors[:len(chart_entities)],
                                     explode=explode, shadow=True)
    
    # Enhance text formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f"Entity Distribution: {entities[0]} Analysis", 
                fontsize=14, fontweight='bold', pad=20)
    
    return _convert_matplotlib_to_pil(fig)

def _convert_matplotlib_to_pil(fig) -> Image.Image:
    """
    Convert matplotlib figure to PIL Image for consistency with other generators.
    Handles proper cleanup and formatting.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, 
               facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Important: clean up matplotlib resources
    return img