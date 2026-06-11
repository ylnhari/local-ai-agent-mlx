#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx-lm",
#     "rich",
#     "questionary",
#     "huggingface_hub",
# ]
# ///
"""
Local LLM Server — powered by MLX on Apple Silicon
Run a fully local, OpenAI-compatible LLM server on your Mac.

Usage:
    uv run start.py          # zero-setup: uv installs deps automatically
    python start.py          # inside an activated venv (after ./setup.sh)
"""
import sys
import os
import subprocess

# ── Dependency check (guards against running outside venv / uv) ───────────────
_missing = []
for _pkg, _name in [("rich", "rich"), ("questionary", "questionary"), ("mlx_lm", "mlx-lm")]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_name)

if _missing:
    print(f"\n  Missing packages: {', '.join(_missing)}")
    print("  Run:  uv run start.py   (or: ./setup.sh then python start.py)\n")
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
import questionary

console = Console()

# ── Model catalog ─────────────────────────────────────────────────────────────
# These are curated recommendations. Any Hugging Face model compatible with
# mlx-lm can be used — enter a custom ID at the bottom of the menu.
# MLX-format models (pre-converted, fastest to start):
#   https://huggingface.co/mlx-community
# Standard HuggingFace models also work — mlx-lm converts them on first load.
MODELS = [
    # ── MLX Community (pre-converted, recommended) ──────────────────────────
    {
        "id":      "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "name":    "Llama 3.2 1B",
        "size":    "~0.7 GB",
        "size_gb": 0.7,
        "desc":    "Ultra-fast, works on 8 GB RAM",
    },
    {
        "id":      "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "name":    "Llama 3.2 3B  ⭐",
        "size":    "~2 GB",
        "size_gb": 2.0,
        "desc":    "Best starter — fast and capable",
        "recommended": True,
    },
    {
        "id":      "mlx-community/Phi-3.5-mini-instruct-4bit",
        "name":    "Phi 3.5 Mini",
        "size":    "~2.2 GB",
        "size_gb": 2.2,
        "desc":    "Strong reasoning and coding for its size",
    },
    {
        "id":      "mlx-community/gemma-3-4b-it-4bit",
        "name":    "Gemma 3 4B",
        "size":    "~2.5 GB",
        "size_gb": 2.5,
        "desc":    "Google's compact instruction model",
    },
    {
        "id":      "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "name":    "Mistral 7B",
        "size":    "~4.1 GB",
        "size_gb": 4.1,
        "desc":    "Excellent general instruction following",
    },
    {
        "id":      "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "name":    "Llama 3.1 8B",
        "size":    "~4.5 GB",
        "size_gb": 4.5,
        "desc":    "High quality, highly versatile",
    },
    {
        "id":      "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "name":    "Qwen 2.5 7B",
        "size":    "~4.3 GB",
        "size_gb": 4.3,
        "desc":    "Great for coding and multilingual tasks",
    },
    {
        "id":      "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
        "name":    "DeepSeek R1 8B",
        "size":    "~5 GB",
        "size_gb": 5.0,
        "desc":    "Reasoning model with chain-of-thought",
    },
    {
        "id":      "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "name":    "Llama 3.3 70B",
        "size":    "~39 GB",
        "size_gb": 39.0,
        "desc":    "Flagship quality (needs 48 GB+ RAM)",
    },
    # ── Other popular sources (also MLX-compatible) ──────────────────────────
    {
        "id":      "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "name":    "Llama 3.2 3B (bartowski)",
        "size":    "~2 GB",
        "size_gb": 2.0,
        "desc":    "Popular community quantization",
    },
    {
        "id":      "unsloth/Llama-3.2-3B-Instruct",
        "name":    "Llama 3.2 3B (unsloth)",
        "size":    "~2 GB",
        "size_gb": 2.0,
        "desc":    "Unsloth fine-tuned variant",
    },
    {
        "id":      "microsoft/Phi-3.5-mini-instruct",
        "name":    "Phi 3.5 Mini (official)",
        "size":    "~7.6 GB",
        "size_gb": 7.6,
        "desc":    "Original bf16 weights — mlx-lm converts on first load",
    },
    {
        "id":      "google/gemma-3-4b-it",
        "name":    "Gemma 3 4B (official)",
        "size":    "~9.8 GB",
        "size_gb": 9.8,
        "desc":    "Original bf16 weights — mlx-lm converts on first load",
    },
]

CATALOG_IDS = {m["id"] for m in MODELS}


# ── Hardware detection ─────────────────────────────────────────────────────────────
def get_mac_specs() -> tuple[int | None, str | None]:
    """Return (ram_gb, chip_name) by reading sysctl values."""
    try:
        ram_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        return ram_bytes // (1024 ** 3), chip
    except Exception:
        return None, None


def _fit_indicator(ram_gb: int, size_gb: float) -> str:
    """Return a coloured fit symbol for a model given available RAM."""
    if size_gb <= ram_gb * 0.60:
        return "[green]✓[/green]"
    elif size_gb <= ram_gb * 0.80:
        return "[yellow]~[/yellow]"
    else:
        return "[red]✗[/red]"


def show_hardware_panel(ram_gb: int, chip: str):
    if ram_gb <= 8:
        sweet_spot  = "1B – 3B models"
        warning     = "[red]Avoid anything larger than ~3 GB — it may crash or swap heavily.[/red]"
    elif ram_gb <= 16:
        sweet_spot  = "3B – 8B models"
        warning     = "7–8B models run well. Avoid 13B+."
    elif ram_gb <= 24:
        sweet_spot  = "7B – 13B models"
        warning     = "8B models will be fast. 13B possible but slower."
    elif ram_gb <= 36:
        sweet_spot  = "8B – 13B models (30B possible)"
        warning     = "30B models will run but expect reduced speed."
    elif ram_gb <= 48:
        sweet_spot  = "up to 34B models"
        warning     = "70B will likely be too slow on this amount of RAM."
    else:
        sweet_spot  = "any model including 70B"
        warning     = "You have enough RAM to run the full catalog comfortably."

    console.print(Panel(
        f"[bold]Chip  [/bold]  {chip}\n"
        f"[bold]Memory[/bold]  {ram_gb} GB unified\n\n"
        f"[bold]Sweet spot[/bold]  {sweet_spot}\n"
        f"{warning}\n\n"
        "[dim]Legend in the model list below — "
        "[green]✓[/green] fits comfortably  "
        "[yellow]~[/yellow] fits but may be slower  "
        "[red]✗[/red] likely too large[/dim]",
        title="[bold]  Your Mac[/bold]",
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_local_models() -> list[str]:
    """Return all model IDs found in the local HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return sorted(
            r.repo_id for r in info.repos
            if r.repo_type == "model"
        )
    except Exception:
        return []


def build_choices(local: list[str], ram_gb: int | None = None) -> list:
    local_set = set(local)
    choices = []

    downloaded_catalog = [m for m in MODELS if m["id"] in local_set]
    extra_local = [lid for lid in local if lid not in CATALOG_IDS]

    if downloaded_catalog or extra_local:
        choices.append(questionary.Separator("  ── Downloaded (ready to run) ──────────────────────────"))
        for m in downloaded_catalog:
            fit = (" " + _fit_indicator(ram_gb, m["size_gb"])) if ram_gb else ""
            choices.append(questionary.Choice(
                title=f"  ✓{fit}  {m['name']:<24} {m['size']:<10}  {m['desc']}",
                value=m["id"],
            ))
        for lid in extra_local:
            choices.append(questionary.Choice(title=f"  ✓    {lid}", value=lid))

    not_downloaded = [m for m in MODELS if m["id"] not in local_set]
    if not_downloaded:
        choices.append(questionary.Separator("  ── Download & Run ─────────────────────────────────────"))
        for m in not_downloaded:
            fit = (" " + _fit_indicator(ram_gb, m["size_gb"])) if ram_gb else ""
            choices.append(questionary.Choice(
                title=f"  ↓{fit}  {m['name']:<24} {m['size']:<10}  {m['desc']}",
                value=m["id"],
            ))

    choices.append(questionary.Separator("  ────────────────────────────────────────────────────────"))
    choices.append(questionary.Choice(
        title="  ✎  Enter any Hugging Face model ID  (mlx-community, official orgs, etc.)…",
        value="__custom__",
    ))
    return choices


def select_model(local: list[str], ram_gb: int | None = None) -> str | None:
    choices = build_choices(local, ram_gb)
    model_id = questionary.select(
        "Select a model to run:",
        choices=choices,
        use_shortcuts=False,
        style=questionary.Style([
            ("separator", "fg:#555555"),
            ("highlighted", "bold fg:cyan"),
            ("pointer",     "fg:cyan"),
            ("selected",    "fg:green"),
        ]),
    ).ask()

    if model_id == "__custom__":
        model_id = questionary.text(
            "Hugging Face model ID  (e.g. mistralai/Mistral-7B-Instruct-v0.3"
            " or mlx-community/Llama-3.2-3B-Instruct-4bit):",
            validate=lambda v: True if v.strip() else "Please enter a model ID",
        ).ask()

    return model_id.strip() if model_id else None


def show_server_panel(model_id: str, port: int):
    console.print(Panel(
        f"[bold green]URL  →[/bold green]  [cyan]http://localhost:{port}/v1[/cyan]\n"
        f"[bold green]Model →[/bold green]  {model_id}\n\n"
        "[bold]── How to connect from any tool ────────────────────────────────[/bold]\n\n"
        "[yellow]Python  (openai SDK)[/yellow]\n"
        "    from openai import OpenAI\n"
        f"    client = OpenAI(base_url=\"http://localhost:{port}/v1\", api_key=\"local\")\n"
        "    resp = client.chat.completions.create(\n"
        f"        model=\"{model_id}\",\n"
        "        messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]\n"
        "    )\n\n"
        "[yellow]LangChain[/yellow]\n"
        "    from langchain_openai import ChatOpenAI\n"
        f"    llm = ChatOpenAI(base_url=\"http://localhost:{port}/v1\",\n"
        f"                     api_key=\"local\", model=\"{model_id}\")\n\n"
        "[yellow]CrewAI / AutoGen / LlamaIndex / any OpenAI-compatible client[/yellow]\n"
        f"    base_url = \"http://localhost:{port}/v1\"\n"
        f"    api_key  = \"local\"\n\n"
        "[dim]See README.md for full examples.  Press Ctrl+C to stop the server.[/dim]",
        title="[bold cyan]  LLM Server Ready[/bold cyan]",
        border_style="green",
        padding=(1, 2),
    ))


def start_server(model_id: str, port: int, is_new: bool):
    if is_new:
        console.print(
            f"\n[yellow]  Model not found locally — it will be downloaded before starting.[/yellow]\n"
            "[dim]  This is a one-time download. The model is cached for all future runs.[/dim]\n"
        )
    else:
        console.print("[dim]  Model is cached locally — loading from disk…[/dim]")

    show_server_panel(model_id, port)

    env = os.environ.copy()
    if not is_new:
        env["HF_HUB_OFFLINE"] = "1"

    cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", model_id, "--port", str(port)]
    os.execvpe(sys.executable, cmd, env)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]  Local LLM Server[/bold cyan]\n"
        "[dim]  MLX · Apple Silicon · OpenAI-compatible API[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    console.print("[dim]  Scanning for local models…[/dim]")
    local = find_local_models()

    ram_gb, chip = get_mac_specs()
    if ram_gb and chip:
        show_hardware_panel(ram_gb, chip)

    if local:
        console.print(f"[green]  Found {len(local)} downloaded model(s).[/green]\n")
    else:
        console.print("[dim]  No local models yet — select one below to download it.\n[/dim]")

    model_id = select_model(local, ram_gb)
    if not model_id:
        console.print("\n[red]  No model selected. Exiting.[/red]\n")
        sys.exit(0)

    port_str = questionary.text("  Port:", default="8080").ask()
    port = int(port_str) if port_str and port_str.isdigit() else 8080

    is_new = model_id not in set(local)
    start_server(model_id, port, is_new)


if __name__ == "__main__":
    main()
