"""
Command-line interface for SynthDoc.

This module provides a CLI for easy access to SynthDoc functionality.
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import List, Optional

from .core import SynthDoc
from .languages import LanguageSupport

app = typer.Typer(
    name="synthdoc",
    help="Generate synthetic documents for ML training",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    language: str = typer.Option("en", "--lang", "-l", help="Language code"),
    pages: int = typer.Option(1, "--pages", "-p", help="Number of pages"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", help="Custom generation prompt"
    ),
    augment: List[str] = typer.Option(
        [], "--augment", "-a", help="Augmentation techniques"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for content generation"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for LLM provider"
    ),
):
    """Generate synthetic documents."""
    console.print(
        f"[green]Generating {pages} documents in {language} using {model}[/green]"
    )

    synth = SynthDoc(output_dir=output, llm_model=model, api_key=api_key)

    try:
        documents = synth.generate_raw_docs(
            language=language, num_pages=pages, prompt=prompt, augmentations=augment
        )

        console.print(f"[green]✓ Generated {len(documents)} documents[/green]")
        console.print(f"[blue]Output saved to: {output}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def languages():
    """List supported languages."""
    table = Table(title="Supported Languages")

    table.add_column("Code", style="cyan")
    table.add_column("Language", style="green")
    table.add_column("Script", style="yellow")
    table.add_column("Category", style="magenta")

    for code, lang_info in LanguageSupport.LANGUAGES.items():
        table.add_row(code, lang_info.name, lang_info.script.value, lang_info.category)

    console.print(table)


@app.command()
def layout(
    input_dir: str = typer.Argument(..., help="Input directory with documents"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    languages: List[str] = typer.Option(
        ["en"], "--lang", "-l", help="Target languages"
    ),
    fonts: List[str] = typer.Option([], "--font", "-f", help="Font families"),
    augment: List[str] = typer.Option(
        [], "--augment", "-a", help="Augmentation techniques"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for content generation"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for LLM provider"
    ),
):
    """Apply layout augmentation to documents."""
    console.print(f"[green]Augmenting documents from {input_dir}[/green]")

    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]Error: Input directory {input_dir} does not exist[/red]")
        raise typer.Exit(1)

    synth = SynthDoc(output_dir=output, llm_model=model, api_key=api_key)

    try:
        # Find document files
        doc_files = (
            list(input_path.glob("*.pdf"))
            + list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
        )

        if not doc_files:
            console.print(f"[red]No documents found in {input_dir}[/red]")
            raise typer.Exit(1)

        dataset = synth.augment_layout(
            document_paths=doc_files,
            languages=languages,
            fonts=fonts,
            augmentations=augment,
        )

        console.print(f"[green]✓ Layout augmentation complete[/green]")
        console.print(f"[blue]Dataset saved to: {output}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def vqa(
    input_dir: str = typer.Argument(..., help="Input directory with documents"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    question_types: List[str] = typer.Option(
        ["factual", "reasoning"], "--type", "-t", help="Question types"
    ),
    difficulty: List[str] = typer.Option(
        ["easy", "medium"], "--difficulty", "-d", help="Difficulty levels"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model for VQA generation"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for LLM provider"
    ),
    hard_negative_ratio: float = typer.Option(
        0.2, "--hard-negative-ratio", help="Ratio of hard negative examples"
    ),
):
    """Generate VQA dataset from documents."""
    console.print(
        f"[green]Generating VQA dataset from {input_dir} using {model}[/green]"
    )

    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]Error: Input directory {input_dir} does not exist[/red]")
        raise typer.Exit(1)

    synth = SynthDoc(output_dir=output, llm_model=model, api_key=api_key)

    try:
        # Load documents from input directory
        # For now, create sample documents - in real implementation would load from files
        documents = [
            {
                "id": "sample_doc",
                "content": "Sample document content for VQA generation",
            }
        ]

        vqa_dataset = synth.generate_vqa(
            source_documents=documents,
            question_types=question_types,
            difficulty_levels=difficulty,
            hard_negative_ratio=hard_negative_ratio,
        )

        console.print(
            f"[green]✓ Generated {len(vqa_dataset['questions'])} VQA pairs[/green]"
        )
        console.print(f"[blue]Dataset saved to: {output}[/blue]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show SynthDoc information."""
    from . import __version__

    console.print(f"[bold blue]SynthDoc v{__version__}[/bold blue]")
    console.print("A comprehensive library for generating synthetic documents")
    console.print(
        f"Supports {len(LanguageSupport.get_supported_languages())} languages"
    )

    # Show quick stats
    lang_categories = {}
    for lang_info in LanguageSupport.LANGUAGES.values():
        category = lang_info.category
        lang_categories[category] = lang_categories.get(category, 0) + 1

    console.print("\n[bold]Language Distribution:[/bold]")
    for category, count in lang_categories.items():
        console.print(f"  {category}: {count} languages")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
