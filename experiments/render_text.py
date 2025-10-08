import cairocffi as cairo
import pangocffi as pango
import pangocairocffi as pangocairo
from pathlib import Path


class KannadaTextRenderer:
    def __init__(self, output_dir="kannada_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # High contrast color combinations
        self.color_combinations = [
            {"text": (0, 0, 0), "bg": (1, 1, 1)},  # Black on White
            {
                "text": (0.13, 0.15, 0.16),
                "bg": (0.97, 0.98, 0.98),
            },  # Dark Gray on Light Gray
            {"text": (0, 0, 0), "bg": (1, 0.95, 0.80)},  # Black on Light Yellow
            {"text": (0.1, 0.53, 0.33), "bg": (1, 1, 1)},  # Green on White
            {"text": (0.05, 0.43, 0.99), "bg": (1, 1, 1)},  # Blue on White
            {"text": (0.40, 0.06, 0.95), "bg": (0.97, 0.98, 0.98)},  # Purple on Gray
            {"text": (0.86, 0.20, 0.27), "bg": (1, 1, 1)},  # Red on White
            {"text": (0, 0, 0), "bg": (0.82, 0.96, 0.92)},  # Black on Light Green
        ]

    def render_text(
        self,
        text,
        font="Noto Sans Kannada 32",
        text_color=(0, 0, 0),
        bg_color=(1, 1, 1),
        max_width=1400,
        filename="output.png",
    ):
        """Render Kannada text with proper shaping using PangoCairo"""

        # Create surface (temporary size, will adjust later)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, max_width, 5000)
        ctx = cairo.Context(surface)

        # Set background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        PANGO_SCALE = 1024
        # Create Pango layout
        layout = pangocairo.create_layout(ctx)
        layout._set_width(max_width * PANGO_SCALE)  # max width
        layout._set_wrap(pango.WrapMode.WORD_CHAR)  # wrap at words/chars
        layout._set_text(text)

        # Set font
        desc = pango.FontDescription(font)
        layout.set_font_description(desc)

        # Set text color
        ctx.set_source_rgb(*text_color)

        # Render
        pangocairo.show_layout(ctx, layout)

        # Get extents
        ink_rect, logical_rect = layout.get_extents()
        width = logical_rect.width // pango.SCALE
        height = logical_rect.height // pango.SCALE

        # Create a new surface with proper size
        surface_final = cairo.ImageSurface(cairo.FORMAT_ARGB32, width + 60, height + 60)
        ctx_final = cairo.Context(surface_final)

        # Background again
        ctx_final.set_source_rgb(*bg_color)
        ctx_final.paint()

        # Text again
        layout_final = pangocairo.create_layout(ctx_final)
        layout_final.set_width(max_width * pango.SCALE)
        layout_final.set_wrap(pango.WrapMode.WORD_CHAR)
        layout_final.set_text(text)
        layout_final.set_font_description(desc)

        ctx_final.set_source_rgb(*text_color)
        ctx_final.move_to(30, 30)
        pangocairo.show_layout(ctx_final, layout_final)

        # Save
        surface_final.write_to_png(str(filename))
        print(f"✅ Saved: {filename} ({width}x{height})")

    def generate_images(self, texts, font="Noto Sans Kannada 48"):
        """Generate one image per text in Kannada"""
        print(f"Starting generation of {len(texts)} images...")
        for idx, text in enumerate(texts):
            colors = self.color_combinations[idx % len(self.color_combinations)]
            filename = self.output_dir / f"kannada_{idx:03d}.png"
            self.render_text(
                text,
                font=font,
                text_color=colors["text"],
                bg_color=colors["bg"],
                filename=filename,
            )


if __name__ == "__main__":
    texts = [
        "ಭಾರತದಮಹೋದ್ದೇಶವಾದಏಕತೆಅವಧುಡೇಶಕವರೋಲಫ್ಇಂದಿಯಾ",
        "ಕನ್ನಡ ಭಾಷೆ ಅತ್ಯಂತ ಸುಂದರವಾದ ಭಾಷೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.",
        "ನಮ್ಮ ದೇಶದ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಅತಿ ವೈವಿಧ್ಯಮಯವಾಗಿದೆ.",
    ]

    renderer = KannadaTextRenderer()
    renderer.generate_images(texts, font="Noto Sans Kannada 48")
