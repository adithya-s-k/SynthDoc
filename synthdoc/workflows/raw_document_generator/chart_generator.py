import matplotlib.pyplot as plt
import random
from PIL import Image
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