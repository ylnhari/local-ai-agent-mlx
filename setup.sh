#!/bin/bash
set -e

echo ""
echo "  Local LLM Server — Setup"
echo "  ─────────────────────────────────────────────────"
echo ""

# ── Check Apple Silicon ────────────────────────────────────────────────────────
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "  ✗  Apple Silicon (M1 or later) is required."
  exit 1
fi
echo "  ✓  Apple Silicon detected"

# ── Check Python 3.10+ ────────────────────────────────────────────────────────
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
  echo "  ✗  Python 3.10+ is required."
  echo "     Install via: brew install python"
  exit 1
fi
echo "  ✓  Python $(python3 --version | awk '{print $2}')"

# ── Install uv if not present ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "  Installing uv (fast Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "  ✓  uv $(uv --version)"

# ── Sync dependencies ─────────────────────────────────────────────────────────
echo "  Syncing dependencies…"
uv sync

echo ""
echo "  ✓  Setup complete!"
echo ""
echo "  Two ways to start the server:"
echo ""
echo "    1. uv run python start.py           # recommended"
echo "    2. source .venv/bin/activate        # then: python start.py"
echo ""
