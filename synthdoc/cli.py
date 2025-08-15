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
from .models_manager import (
    download_model,
    list_available_models,
    get_model_info,
    is_model_downloaded,
    cleanup_models,
)
from .models import AugmentationConfig

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
    # Removed augment parameter - augmentation not implemented
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
            language=language, num_pages=pages, prompt=prompt
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


# Removed layout command - layout augmentation workflow not implemented


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


# Model management commands
@app.command()
def download_models(
    model_name: Optional[str] = typer.Argument(
        None, help="Model name to download (default: all)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
):
    """Download pre-trained models."""
    if model_name:
        if model_name not in list_available_models():
            console.print(f"[red]Error: Unknown model '{model_name}'[/red]")
            console.print("Available models:")
            list_models()
            return

        console.print(f"[blue]Downloading model: {model_name}[/blue]")
        try:
            path = download_model(model_name, force_download=force)
            console.print(f"[green]✅ Model downloaded to: {path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Download failed: {e}[/red]")
    else:
        # Download all models
        available_models = list_available_models()
        console.print(f"[blue]Downloading {len(available_models)} models...[/blue]")

        for model in available_models:
            console.print(f"\n[blue]Downloading {model}...[/blue]")
            try:
                path = download_model(model, force_download=force)
                console.print(f"[green]✅ {model} downloaded to: {path}[/green]")
            except Exception as e:
                console.print(f"[red]❌ {model} download failed: {e}[/red]")


@app.command()
def list_models():
    """List available models and their download status."""
    console.print("[bold]Available Models:[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Size (MB)", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    available_models = list_available_models()
    for model_name, config in available_models.items():
        try:
            info = get_model_info(model_name)
            status = "✅ Downloaded" if info["downloaded"] else "❌ Not downloaded"
            status_style = "green" if info["downloaded"] else "red"
        except Exception:
            status = "❓ Unknown"
            status_style = "yellow"

        table.add_row(
            model_name,
            config["description"],
            str(config["size_mb"]),
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)

    console.print("\n[bold]Usage:[/bold]")
    console.print("  synthdoc download-models <model_name>  # Download specific model")
    console.print("  synthdoc download-models               # Download all models")


@app.command()
def model_info(
    model_name: str = typer.Argument(..., help="Model name to get info about"),
):
    """Get detailed information about a specific model."""
    try:
        info = get_model_info(model_name)

        console.print(f"[bold]Model: {model_name}[/bold]\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Description", info["description"])
        table.add_row("Repository", info["repo_id"])
        table.add_row("Remote Size", f"{info['size_mb']} MB")
        table.add_row("Downloaded", "✅ Yes" if info["downloaded"] else "❌ No")

        if info["downloaded"]:
            table.add_row("Local Path", info["local_path"])
            table.add_row("Local Size", f"{info.get('local_size_mb', 'Unknown')} MB")

        console.print(table)

        if not info["downloaded"]:
            console.print(
                f"\n[yellow]To download: synthdoc download-models {model_name}[/yellow]"
            )

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def clean_models(
    model_name: Optional[str] = typer.Argument(
        None, help="Model name to remove (default: all)"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove downloaded models to free up space."""
    if not confirm:
        if model_name:
            if not typer.confirm(f"Remove model '{model_name}'?"):
                console.print("Cancelled.")
                return
        else:
            if not typer.confirm("Remove ALL downloaded models?"):
                console.print("Cancelled.")
                return

    try:
        cleanup_models(model_name)
        if model_name:
            console.print(f"[green]✅ Removed model: {model_name}[/green]")
        else:
            console.print("[green]✅ Removed all models[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@app.command()
def augment(
    input_path: str = typer.Argument(..., help="Input dataset folder or HuggingFace dataset path"),
    output: str = typer.Option("./augmented", "--output", "-o", help="Output directory"),
    augmentations: List[str] = typer.Option(
        ["brightness", "folding", "original"], 
        "--augmentations", "-a", 
        help="Augmentation types to apply"
    ),
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p", 
        help="Use preset: light, balanced, heavy, document_quality"
    ),
    original_ratio: float = typer.Option(
        0.3, "--original-ratio", help="Ratio of images to keep as original"
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", help="Maximum number of images to process"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducible results"
    ),
):
    """Apply augmentations to existing datasets or image folders."""
    console.print(f"[green]Applying augmentations to {input_path}[/green]")
    
    # Handle preset selection
    if preset:
        from .augmentation import (
            LIGHT_AUGMENTATIONS, 
            BALANCED_AUGMENTATIONS, 
            HEAVY_AUGMENTATIONS,
            DOCUMENT_QUALITY_AUGMENTATIONS
        )
        
        preset_map = {
            "light": LIGHT_AUGMENTATIONS,
            "balanced": BALANCED_AUGMENTATIONS, 
            "heavy": HEAVY_AUGMENTATIONS,
            "document_quality": DOCUMENT_QUALITY_AUGMENTATIONS,
        }
        
        if preset not in preset_map:
            console.print(f"[red]Unknown preset: {preset}[/red]")
            console.print(f"Available presets: {', '.join(preset_map.keys())}")
            raise typer.Exit(1)
        
        augmentations = preset_map[preset]
        console.print(f"[blue]Using preset '{preset}': {augmentations}[/blue]")
    
    try:
        synth = SynthDoc(output_dir=output)
        
        # Check if input is a folder or dataset
        input_path_obj = Path(input_path)
        if input_path_obj.is_dir():
            # Process image folder
            result = synth.apply_augmentations(
                input_data=input_path,
                augmentations=augmentations,
                original_ratio=original_ratio,
                max_samples=max_samples,
                random_seed=seed,
                output_folder=output
            )
        else:
            # Try to load as dataset (future enhancement)
            console.print("[yellow]Dataset input not yet supported. Use image folder for now.[/yellow]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓ Applied augmentations to {len(result)} images[/green]")
        console.print(f"[blue]Output saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
