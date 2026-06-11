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
import sys, time, subprocess, re, socket

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

def memory_pressure() -> str:
    try:
        out = subprocess.check_output(["memory_pressure"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=2)
        m = re.search(r'System-wide memory free percentage:\s*(\d+)%', out)
        if m:
            f = int(m.group(1))
            return "[green]normal[/green]" if f > 40 else "[yellow]warning[/yellow]" if f > 15 else "[red]critical[/red]"
    except Exception:
        pass
    return "[dim]unknown[/dim]"

def llm_process() -> psutil.Process | None:
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "mlx_lm" in cmd and "server" in cmd:
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def server_online(port: int = 8080) -> bool:
    """TCP connect check — no HTTP request, nothing in server logs."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def build_table() -> Panel:
    mem   = psutil.virtual_memory()
    cpu   = psutil.cpu_percent(interval=None)
    gpu   = gpu_utilization()
    proc  = llm_process()
    up    = server_online()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    t.add_column("label", style="bold dim", width=18)
    t.add_column("bar",   min_width=32)
    t.add_column("value", justify="right", width=18)

    # ── CPU ───────────────────────────────────────────────────────────────────
    t.add_row("CPU", f"[{_c(cpu)}]{_bar(cpu)}[/{_c(cpu)}]",
              f"[{_c(cpu)}]{cpu:.1f}%[/{_c(cpu)}]")

    # ── GPU (Metal) ───────────────────────────────────────────────────────────
    if gpu is not None:
        t.add_row("GPU  Metal", f"[{_c(gpu)}]{_bar(gpu)}[/{_c(gpu)}]",
                  f"[{_c(gpu)}]{gpu}%[/{_c(gpu)}]")
    else:
        t.add_row("GPU  Metal", "[dim]" + "─" * 32 + "[/dim]", "[dim]no data[/dim]")

    # ── RAM ───────────────────────────────────────────────────────────────────
    rp = mem.percent
    used = mem.used / 1024**3;  total = mem.total / 1024**3
    t.add_row("RAM", f"[{_c(rp)}]{_bar(rp)}[/{_c(rp)}]",
              f"[{_c(rp)}]{used:.1f} / {total:.0f} GB[/{_c(rp)}]")

    t.add_row("Memory pressure", "", memory_pressure())
    t.add_row("", "", "")

    # ── LLM server process ────────────────────────────────────────────────────
    if proc:
        try:
            pm  = proc.memory_info().rss
            pmp = pm / mem.total * 100
            pc  = proc.cpu_percent(interval=None)
            t.add_row("LLM  RAM", f"[{_c(pmp)}]{_bar(pmp)}[/{_c(pmp)}]",
                      f"[{_c(pmp)}]{pm/1024**3:.1f} GB[/{_c(pmp)}]")
            t.add_row("LLM  CPU", f"[{_c(pc)}]{_bar(pc)}[/{_c(pc)}]",
                      f"[{_c(pc)}]{pc:.1f}%[/{_c(pc)}]")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            t.add_row("LLM  process", "[dim]gone[/dim]", "")
    else:
        t.add_row("LLM  Server", "[dim]─[/dim]", "[dim]not running[/dim]")

    t.add_row("Server :8080", "",
              "[green]● online[/green]" if up else "[red]○ offline[/red]")

    return Panel(t,
                 title="[bold cyan]  Apple Silicon Monitor[/bold cyan]",
                 subtitle="[dim]Ctrl+C to quit · refreshes every second[/dim]",
                 border_style="cyan", padding=(0, 1))


def main():
    # Prime cpu_percent (first call always returns 0)
    psutil.cpu_percent(interval=None)
    proc = llm_process()
    if proc:
        try: proc.cpu_percent(interval=None)
        except Exception: pass

    console.print()
    console.print("[dim]  For ANE + per-core breakdown install:[/dim]  "
                  "[cyan]brew install asitop[/cyan]  [dim]then[/dim]  [cyan]sudo asitop[/cyan]\n")

    try:
        with Live(build_table(), refresh_per_second=1, screen=False) as live:
            while True:
                time.sleep(1)
                live.update(build_table())
    except KeyboardInterrupt:
        console.print("\n[yellow]  Monitor stopped.[/yellow]\n")


if __name__ == "__main__":
    main()
