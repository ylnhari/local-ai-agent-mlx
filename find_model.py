#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "rich"]
# ///
"""
find_model.py — Find the best MLX-optimized model for any base model name.

Usage:
    uv run find_model.py "Llama 3.2 3B"
    uv run find_model.py "Mistral 7B"
    uv run find_model.py "Qwen 2.5 7B"
"""
import sys

_missing = []
for _pkg, _name in [("huggingface_hub", "huggingface_hub"), ("rich", "rich")]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_name)

if _missing:
    print(f"\n  Missing: {', '.join(_missing)}.  Run:  uv run find_model.py\n")
    sys.exit(1)

from huggingface_hub import HfApi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Orgs known to publish MLX-compatible models
MLX_ORGS = ["mlx-community", "bartowski", "unsloth", "lmstudio-community"]

# Quantization priority (lower index = higher preference)
QUANT_PRIORITY = ["8bit", "4bit", "3bit", "2bit", "fp16", "bf16"]

def quant_score(repo_id: str) -> int:
    rid = repo_id.lower()
    for i, q in enumerate(QUANT_PRIORITY):
        if q in rid:
            return i
    return len(QUANT_PRIORITY)  # unknown quant goes last

def search_mlx_models(query: str) -> list[dict]:
    api = HfApi()
    results = []

    for org in MLX_ORGS:
        try:
            models = list(api.list_models(
                search=query,
                author=org,
                sort="downloads",
                direction=-1,
                limit=10,
                cardData=False,
            ))
            for m in models:
                results.append({
                    "id":        m.id,
                    "org":       org,
                    "downloads": m.downloads or 0,
                    "updated":   str(m.lastModified)[:10] if m.lastModified else "unknown",
                    "likes":     m.likes or 0,
                })
        except Exception:
            continue

    # Sort: prefer mlx-community, then by quant priority, then by downloads
    results.sort(key=lambda r: (
        0 if r["org"] == "mlx-community" else 1,
        quant_score(r["id"]),
        -r["downloads"],
    ))
    return results


def main():
    if len(sys.argv) < 2:
        console.print("\n  [bold red]Usage:[/bold red]  uv run find_model.py \"<model name>\"\n")
        console.print("  [dim]Examples:[/dim]")
        console.print("    uv run find_model.py \"Llama 3.2 3B\"")
        console.print("    uv run find_model.py \"Mistral 7B\"")
        console.print("    uv run find_model.py \"Qwen 2.5\"\n")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    console.print()
    console.print(f"[dim]  Searching Hugging Face for MLX-compatible variants of:[/dim] [cyan]{query}[/cyan]\n")

    results = search_mlx_models(query)

    if not results:
        console.print(Panel(
            f"[yellow]No MLX-format models found for '{query}'.[/yellow]\n\n"
            "You can still run the original model — mlx-lm will convert it on first load:\n\n"
            f"    python -m mlx_lm server --model <original-hf-id>\n\n"
            "To pre-convert and quantize:\n"
            f"    python -m mlx_lm convert --model <original-hf-id> -q",
            title="[bold yellow]  No MLX variants found[/bold yellow]",
            border_style="yellow",
        ))
        sys.exit(0)

    # Best match
    best = results[0]
    console.print(Panel(
        f"[bold green]Model ID:[/bold green]  [cyan]{best['id']}[/cyan]\n"
        f"[bold green]Downloads:[/bold green] {best['downloads']:,}   "
        f"[bold green]Likes:[/bold green] {best['likes']}   "
        f"[bold green]Updated:[/bold green] {best['updated']}\n\n"
        "[bold]Run it:[/bold]\n"
        "    uv run python start.py  →  use the custom ID entry\n\n"
        "    [dim]or directly:[/dim]\n"
        f"    python -m mlx_lm server --model {best['id']}",
        title="[bold cyan]  Best Match[/bold cyan]",
        border_style="green",
        padding=(1, 2),
    ))

    if len(results) > 1:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("Model ID",   style="cyan",  no_wrap=True)
        table.add_column("Org",        style="dim",   width=20)
        table.add_column("Downloads",  justify="right")
        table.add_column("Updated",    style="dim")

        for r in results[1:10]:
            table.add_row(r["id"], r["org"], f"{r['downloads']:,}", r["updated"])

        console.print("  [bold]Alternatives:[/bold]")
        console.print(table)

    console.print("[dim]  Tip: paste any model ID into the custom entry in start.py[/dim]\n")


if __name__ == "__main__":
    main()
