#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["psutil", "rich"]
# ///
"""
monitor.py — Live CPU, GPU and memory monitor for Apple Silicon.
Open in a second terminal while the LLM server is running.

Usage:
    uv run monitor.py
"""
import sys, time, subprocess, re, socket, urllib.request

_missing = []
for _p, _n in [("psutil", "psutil"), ("rich", "rich")]:
    try:
        __import__(_p)
    except ImportError:
        _missing.append(_n)
if _missing:
    print(f"\n  Missing: {', '.join(_missing)}.  Run:  uv run monitor.py\n")
    sys.exit(1)

import psutil
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box

console = Console()


def _bar(pct: float, width: int = 30) -> str:
    n = int(min(pct, 100) / 100 * width)
    return "█" * n + "░" * (width - n)

def _c(pct: float) -> str:
    return "green" if pct < 60 else "yellow" if pct < 85 else "red"

def gpu_utilization() -> int | None:
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def memory_free_pct() -> int | None:
    """System-wide free-memory percentage from macOS `memory_pressure`,
    or None if unavailable."""
    try:
        out = subprocess.check_output(["memory_pressure"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=2)
        m = re.search(r'System-wide memory free percentage:\s*(\d+)%', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None

def llm_process() -> psutil.Process | None:
    """Find the mlx_lm server process. Among matches, prefer the one with the
    largest RSS — that's the worker actually holding the model, not a wrapper."""
    best = None
    best_rss = -1
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "mlx_lm" in cmd and "server" in cmd:
                rss = p.memory_info().rss
                if rss > best_rss:
                    best, best_rss = p, rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return best

def proc_tree_rss(proc: psutil.Process) -> int:
    """Sum RSS across the process and all its children.
    NOTE: RSS does NOT include Metal/IOGPU wired memory where MLX maps model
    weights — watch the 'Wired RAM' row for the real model footprint."""
    total = 0
    try:
        total += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total

def proc_tree_cpu(proc: psutil.Process) -> float:
    """Sum cpu_percent across the process and all its children."""
    total = 0.0
    try:
        total += proc.cpu_percent(interval=None)
        for child in proc.children(recursive=True):
            try:
                total += child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total

def server_status(port: int = 8080) -> str:
    """Distinguish 'ready' (server serving HTTP) from 'loading' (port open but
    not yet answering) from 'offline'. A bare TCP check can't tell these apart,
    so we hit the cheap /health endpoint (returns {"status":"ok"} instantly —
    unlike /v1/models, which rescans the whole HF cache on every call)."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            pass
    except OSError:
        return "offline"
    # Port is open — confirm the HTTP server is actually responding.
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return "ready"
    except Exception:
        pass
    return "loading"


def build_table() -> Panel:
    mem   = psutil.virtual_memory()
    cpu   = psutil.cpu_percent(interval=None)
    gpu   = gpu_utilization()
    proc  = llm_process()
    status = server_status()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    t.add_column("label", style="bold dim", width=22)
    t.add_column("bar",   min_width=30)
    t.add_column("value", justify="right", width=18)

    def section(title: str):
        t.add_row(f"[bold cyan]{title}[/bold cyan]", "", "")

    # ════ SECTION 1: the whole Mac ════════════════════════════════════════════
    section("SYSTEM  ·  whole machine")

    # CPU across all cores
    t.add_row("  CPU  (all cores)", f"[{_c(cpu)}]{_bar(cpu)}[/{_c(cpu)}]",
              f"[{_c(cpu)}]{cpu:.1f}%[/{_c(cpu)}]")

    # GPU (Metal)
    if gpu is not None:
        t.add_row("  GPU  (Metal)", f"[{_c(gpu)}]{_bar(gpu)}[/{_c(gpu)}]",
                  f"[{_c(gpu)}]{gpu}%[/{_c(gpu)}]")
    else:
        t.add_row("  GPU  (Metal)", "[dim]" + "─" * 30 + "[/dim]", "[dim]no data[/dim]")

    # RAM total, with Wired shown as a subset
    rp = mem.percent
    used = mem.used / 1024**3;  total = mem.total / 1024**3
    t.add_row("  RAM  (total used)", f"[{_c(rp)}]{_bar(rp)}[/{_c(rp)}]",
              f"[{_c(rp)}]{used:.1f} / {total:.0f} GB[/{_c(rp)}]")

    wired = getattr(mem, "wired", None)
    if wired is not None:
        wp = wired / mem.total * 100
        t.add_row("  └ of which Wired", f"[{_c(wp)}]{_bar(wp)}[/{_c(wp)}]",
                  f"[{_c(wp)}]{wired/1024**3:.1f} GB[/{_c(wp)}]")
        t.add_row("", "", "[dim](locked in RAM, non-pageable)[/dim]")

    # Memory pressure: bar shows how "full" RAM is (100 - free%), colored by level
    free = memory_free_pct()
    if free is not None:
        usedp = 100 - free
        col = _c(usedp)
        level = "normal" if free > 40 else "warning" if free > 15 else "critical"
        t.add_row("  Memory pressure", f"[{col}]{_bar(usedp)}[/{col}]",
                  f"[{col}]{level} · {free}% free[/{col}]")
    else:
        t.add_row("  Memory pressure", "[dim]" + "─" * 30 + "[/dim]", "[dim]unknown[/dim]")
    t.add_row("", "", "")

    # ════ SECTION 2: the LLM server process ═══════════════════════════════════
    section("LLM SERVER  ·  mlx_lm process")

    status_display = {
        "ready":   "[green]● ready[/green]",
        "loading": "[yellow]◐ loading…[/yellow]",
        "offline": "[red]○ offline[/red]",
    }[status]
    t.add_row("  Status (:8080)", "", status_display)

    if proc:
        try:
            pm  = proc_tree_rss(proc)
            pmp = pm / mem.total * 100
            pc  = proc_tree_cpu(proc)
            t.add_row("  Process RAM", f"[{_c(pmp)}]{_bar(pmp)}[/{_c(pmp)}]",
                      f"[{_c(pmp)}]{pm/1024**3:.1f} GB[/{_c(pmp)}]")
            t.add_row("  Process CPU", f"[{_c(pc)}]{_bar(pc)}[/{_c(pc)}]",
                      f"[{_c(pc)}]{pc:.1f}%[/{_c(pc)}]")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            t.add_row("  Process", "[dim]gone[/dim]", "")
    else:
        t.add_row("  Process", "[dim]─[/dim]", "[dim]not running[/dim]")

    return Panel(t,
                 title="[bold cyan]  Apple Silicon Monitor[/bold cyan]",
                 subtitle="[dim]Ctrl+C to quit · refreshes every second[/dim]",
                 border_style="cyan", padding=(0, 1))


def main():
    # Prime cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)
    proc = llm_process()
    if proc:
        try: proc_tree_cpu(proc)
        except Exception: pass

    console.print()
    console.print("[dim]  For ANE + per-core breakdown install:[/dim]  "
                  "[cyan]brew install asitop[/cyan]  [dim]then[/dim]  [cyan]sudo asitop[/cyan]\n")
    console.print("[bold cyan]  LEGEND[/bold cyan]  [dim](Apple Silicon shares ONE unified-memory pool between CPU and GPU)[/dim]\n")

    console.print("[bold]  SYSTEM[/bold] [dim]— the whole Mac (everything running, not just the LLM)[/dim]")
    console.print("    [bold]CPU (all cores)[/bold]    [dim]%[/dim]   how busy all CPU cores are, combined")
    console.print("    [bold]GPU (Metal)[/bold]        [dim]%[/dim]   how busy the Apple GPU is (does the real LLM math)")
    console.print("    [bold]RAM (total used)[/bold]   [dim]GB[/dim]  physical memory used by everything / your total RAM")
    console.print("    [bold]└ of which Wired[/bold]   [dim]GB[/dim]  the slice of RAM that is LOCKED and can't be paged out")
    console.print("                            [dim](non-pageable; includes GPU/Metal memory), as reported by macOS[/dim]")
    console.print("    [bold]Memory pressure[/bold]         normal / warning / critical (macOS's own verdict)\n")

    console.print("[bold]  LLM SERVER[/bold] [dim]— just the mlx_lm process[/dim]")
    console.print("    [bold]Status (:8080)[/bold]          ready = serving · loading = starting up · offline = not running")
    console.print("    [bold]Process RAM[/bold]        [dim]GB[/dim]  memory used by the server process, as reported by macOS")
    console.print("    [bold]Process CPU[/bold]        [dim]%[/dim]   CPU used by the server process (+ its children)\n")

    console.print("[dim]  For total memory use, read RAM and Wired (system-wide).[/dim]")
    console.print("[dim]  Process RAM/CPU are per-process numbers as reported by macOS.[/dim]\n")

    try:
        with Live(build_table(), refresh_per_second=1, screen=False) as live:
            while True:
                time.sleep(1)
                live.update(build_table())
    except KeyboardInterrupt:
        console.print("\n[yellow]  Monitor stopped.[/yellow]\n")


if __name__ == "__main__":
    main()
