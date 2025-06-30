import random
import os
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from ..base import BaseWorkflow
from ...models import HandwritingGenerationConfig, WorkflowResult
from ...languages import load_language_font


class HandwritingGenerator(BaseWorkflow):
    """Generate handwritten documents with realistic handwriting simulation."""

    def __init__(self, save_dir: str = "handwriting_output"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def process(self, config: HandwritingGenerationConfig) -> WorkflowResult:
        """Generate handwritten documents based on configuration."""
        print(f"✍️ Starting handwriting generation ({config.num_samples} samples)...")
        
        samples = []
        
        for i in range(config.num_samples):
            print(f"Generating handwriting sample {i + 1}/{config.num_samples}")
            
            # Use provided content or generate sample content
            content = config.text_content or self._generate_sample_content(config.language, i)
            
            # Generate handwritten image
            image = self._render_handwriting(
                text=content,
                language=config.language,
                style=config.handwriting_style
            )
            
            # Save image
            img_filename = f"handwriting_{i}_{config.handwriting_style}.png"
            img_path = os.path.join(self.save_dir, img_filename)
            image.save(img_path)
            
            # Apply augmentations if specified
            if config.augmentations:
                image = self._apply_augmentations(image, config.augmentations)
            
            sample = {
                "id": f"handwriting_{i}",
                "image_path": img_path,
                "text_content": content,
                "handwriting_style": config.handwriting_style,
                "language": config.language.value if hasattr(config.language, 'value') else str(config.language),
                "augmentations": config.augmentations or [],
                "image_width": image.width,
                "image_height": image.height,
                "metadata": {
                    "writing_style": config.handwriting_style,
                    "paper_type": "lined",  # Default paper type
                    "font_variation": random.choice(["regular", "bold", "italic"]),
                    "pen_color": random.choice(["blue", "black", "red"]),
                    "margin_left": 60,
                    "line_spacing": 25
                }
            }
            samples.append(sample)

        dataset = self._create_hf_dataset(
            samples, 
            {
                "workflow": "handwriting_generation", 
                "config": config.dict(),
                "total_samples": len(samples)
            }
        )

        output_files = [sample["image_path"] for sample in samples if "image_path" in sample]
        
        return WorkflowResult(
            dataset=dataset,
            metadata={
                "workflow_type": "handwriting_generation",
                "total_samples": len(samples),
                "styles_used": [config.handwriting_style],
                "languages": [config.language.value if hasattr(config.language, 'value') else str(config.language)],
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            },
            num_samples=len(samples),
            output_files=output_files
        )

    def _generate_sample_content(self, language, sample_idx: int) -> str:
        """Generate sample content for handwriting in different languages."""
        sample_contents = {
            "en": [
                "Dear friend,\nI hope this letter finds you well. Today was a beautiful day, and I spent most of it reading in the garden. The flowers are blooming, and the birds are singing their morning songs.",
                "Today's shopping list:\n• Milk and bread\n• Fresh vegetables\n• Coffee beans\n• Notebook for writing\n• Stamps for letters",
                "Meeting Notes - Project Planning\n\nDiscussed timeline for next quarter\nAssigned tasks to team members\nScheduled follow-up meetings\nReviewed budget requirements",
                "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for handwriting practice.",
                "Dear diary,\nToday I learned something new about typography and document design. The way we arrange text on a page can greatly influence how readers understand and engage with the content."
            ],
            "hi": [
                "प्रिय मित्र,\nआशा है आप स्वस्थ और प्रसन्न होंगे। आज मैंने एक अच्छी किताब पढ़ी और बहुत कुछ सीखा।",
                "आज की खरीदारी की सूची:\n• दूध और ब्रेड\n• ताजी सब्जियां\n• चाय पत्ती\n• नोटबुक\n• कलम",
                "बैठक के नोट्स\nआज की चर्चा के मुख्य बिंदु\nअगली योजना पर विचार\nकार्य का विभाजन\nसमय सीमा का निर्धारण",
                "शिक्षा सबसे शक्तिशाली हथियार है जिससे आप दुनिया बदल सकते हैं। ज्ञान ही सच्ची संपत्ति है।",
                "प्रिय डायरी,\nआज मैंने देखा कि कैसे अच्छा लेखन और सुंदर हस्तलेखन मिलकर एक बेहतरीन दस्तावेज बनाते हैं।"
            ],
            "zh": [
                "亲爱的朋友，\n希望你一切都好。今天是美好的一天，我在花园里读书，花儿正在盛开。",
                "今日购物清单：\n• 牛奶和面包\n• 新鲜蔬菜\n• 咖啡豆\n• 笔记本\n• 邮票",
                "会议记录 - 项目规划\n讨论了下季度时间表\n分配团队任务\n安排后续会议\n审查预算需求",
                "书法练习：静以修身，俭以养德。学而时习之，不亦说乎。",
                "今天学习了文档设计的重要性。好的排版和清晰的手写体能够帮助读者更好地理解内容。"
            ]
        }
        
        language_key = language.value if hasattr(language, 'value') else str(language)
        contents = sample_contents.get(language_key, sample_contents["en"])
        return contents[sample_idx % len(contents)]

    def _render_handwriting(self, text: str, language, style: str) -> Image.Image:
        """Render text as handwriting with realistic variations."""
        # Create lined paper background
        width, height = 800, 1000
        image = self._create_lined_paper(width, height)
        draw = ImageDraw.Draw(image)
        
        # Load language-appropriate font
        try:
            language_code = language.value if hasattr(language, 'value') else str(language)
            if style == "cursive":
                font = load_language_font(language_code, 16, style="italic")
            else:
                font = load_language_font(language_code, 14)
        except Exception as e:
            print(f"⚠️ Font loading error: {e}")
            try:
                if style == "cursive":
                    font = ImageFont.truetype("Arial", 16)
                else:
                    font = ImageFont.truetype("Arial", 14)
            except:
                font = ImageFont.load_default()
        
        # Simulate handwriting with variations
        lines = text.split('\n')
        y_offset = 80
        line_height = 25
        margin_left = 60
        
        # Choose pen color
        pen_colors = ["blue", "black", "darkblue", "darkgreen"]
        pen_color = random.choice(pen_colors)
        
        for line_idx, line in enumerate(lines):
            if y_offset + line_height > height - 50:
                break
            
            # Add line variation (slight slant)
            base_x = margin_left + random.randint(-5, 5)
            y_position = y_offset + random.randint(-3, 3)
            
            # Handle different writing styles
            if style == "print":
                # Block letters with more spacing
                char_spacing = 0
                for char_idx, char in enumerate(line):
                    char_x = base_x + char_idx * 8 + char_spacing + random.randint(-2, 2)
                    char_y = y_position + random.randint(-2, 2)
                    draw.text((char_x, char_y), char, fill=pen_color, font=font)
                    char_spacing += random.randint(1, 3)
            elif style == "cursive":
                # Connected letters with flow
                word_spacing = 0
                words = line.split(' ')
                for word_idx, word in enumerate(words):
                    word_x = base_x + word_spacing + random.randint(-3, 3)
                    word_y = y_position + random.randint(-2, 2)
                    
                    # Add slight slant for cursive
                    draw.text((word_x, word_y), word, fill=pen_color, font=font)
                    word_spacing += len(word) * 7 + 15  # Space between words
            else:
                # Default handwriting style
                draw.text((base_x, y_position), line, fill=pen_color, font=font)
            
            y_offset += line_height + random.randint(-2, 5)
        
        # Add some realistic imperfections
        self._add_handwriting_artifacts(image, draw)
        
        return image

    def _create_lined_paper(self, width: int, height: int) -> Image.Image:
        """Create realistic lined paper background."""
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw horizontal lines
        line_color = (200, 220, 255)  # Light blue
        for y in range(80, height - 50, 25):
            # Add slight line variation
            start_x = 50 + random.randint(-2, 2)
            end_x = width - 50 + random.randint(-2, 2)
            draw.line([(start_x, y), (end_x, y)], fill=line_color, width=1)
        
        # Draw margin line
        margin_color = (255, 200, 200)  # Light red
        margin_x = 80 + random.randint(-1, 1)
        draw.line([(margin_x, 50), (margin_x, height - 50)], fill=margin_color, width=2)
        
        # Add paper texture (very subtle)
        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height)
            noise_color = (250, 250, 250)
            draw.point((x, y), fill=noise_color)
        
        return image

    def _add_handwriting_artifacts(self, image: Image.Image, draw: ImageDraw.Draw):
        """Add realistic handwriting artifacts like ink blots and pressure variations."""
        width, height = image.size
        
        # Add a few small ink spots
        for _ in range(random.randint(1, 3)):
            x = random.randint(60, width - 60)
            y = random.randint(80, height - 80)
            spot_size = random.randint(1, 2)
            draw.ellipse([x-spot_size, y-spot_size, x+spot_size, y+spot_size], fill='darkblue')
        
        # Add very subtle smudges
        for _ in range(random.randint(2, 5)):
            x = random.randint(60, width - 60)
            y = random.randint(80, height - 80)
            smudge_color = (240, 240, 240)
            draw.point((x, y), fill=smudge_color)

    def _apply_augmentations(self, image: Image.Image, augmentations: List[str]) -> Image.Image:
        """Apply augmentations to the handwritten image."""
        for aug in augmentations:
            if aug == "rotation":
                # Small rotation to simulate paper tilt
                angle = random.uniform(-3, 3)
                image = image.rotate(angle, expand=True, fillcolor='white')
            elif aug == "noise":
                # Add slight noise
                import numpy as np
                np_image = np.array(image)
                noise = np.random.normal(0, 5, np_image.shape).astype(np.uint8)
                np_image = np.clip(np_image.astype(int) + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(np_image)
            elif aug == "brightness":
                # Slight brightness variation
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(image)
                factor = random.uniform(0.9, 1.1)
                image = enhancer.enhance(factor)
            # Add more augmentations as needed
        
        return image 