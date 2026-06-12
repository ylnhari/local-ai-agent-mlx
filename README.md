# Local LLM Server — MLX on Apple Silicon

Run any LLM **fully locally** on your Mac with a single command. This tool downloads, serves, and manages open-source models through an **OpenAI-compatible HTTP API** — no cloud, no subscriptions, complete privacy.

---

## Requirements

| | |
|---|---|
| **Mac** | Apple Silicon (M1 / M2 / M3 / M4 or later) |
| **macOS** | 14.0 Sonoma or later |
| **RAM** | 8 GB minimum · 16 GB recommended |
| **Python** | 3.10 or later (`brew install python`) |
| **uv** | Fast Python package manager — installed automatically by `setup.sh` |

---

## Setup (one-time)

> **Using an agentic IDE?** The [Local AI Setup agent](#local-ai-setup-agent) can run these steps for you — just attach the agent file and type `Set up this repo`.

```bash
git clone https://github.com/bby-corp/local-ai-agent-mlx.git
cd local-ai-agent-mlx
./setup.sh
```

`setup.sh` installs [uv](https://docs.astral.sh/uv/) if needed, then runs `uv sync` to create a lockfile-pinned virtual environment in `.venv`.

---

## Run

> **Using an agentic IDE?** The [Local AI Setup agent](#local-ai-setup-agent) will start the server, pick the right model for your RAM, and open the monitoring dashboard — all in one flow.

```bash
# Recommended — uv resolves the environment automatically
uv run python start.py

# Alternative — activate the venv manually
source .venv/bin/activate && python start.py
```

> **Zero-setup shortcut:** if you already have `uv` installed you can skip `setup.sh` entirely:
> ```bash
> uv run start.py
> ```
> uv reads the inline dependency metadata in `start.py` (PEP 723) and installs everything on the fly.

The interactive menu will:

1. Detect any models you have already downloaded.
2. Let you pick from a curated model list — or enter any `mlx-community/*` model ID from Hugging Face.
3. Download the model if needed *(one-time, cached forever after)*.
4. Start the OpenAI-compatible server and print the URL and integration snippets.

---

## IDE & Coding Tool Integrations

Once the server is running at `http://localhost:8080/v1`, connect any of these tools:

| Tool | How to connect | What you get |
|---|---|---|
| **Continue** (VS Code) | `apiBase: http://localhost:8080/v1` in `~/.continue/config.yaml` | Inline edit (`Cmd+I`), chat panel (`Cmd+L`) |
| **Cursor** | Settings → Models → OpenAI-compatible → `http://localhost:8080/v1` | AI editor backed by your local model |
| **Windsurf** | Settings → AI → Custom model → `http://localhost:8080/v1` | Agentic coding assistant, local model |
| **Open WebUI** | `docker run` with `OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1` | ChatGPT-style browser UI |
| **Any OpenAI client** | `base_url="http://localhost:8080/v1"`, `api_key="local"` | LangChain, LlamaIndex, CrewAI, AutoGen, curl |

### Continue setup (VS Code)

Install the [Continue extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue), then edit `~/.continue/config.yaml`:

```yaml
models:
  - name: Local MLX
    provider: openai
    model: <your-model-id>
    apiBase: http://localhost:8080/v1
    apiKey: local
    roles: [chat, edit, apply]
```

Reload VS Code — the Continue sidebar will now use your local server.

---

## Integrating with Agentic Tools

Once the server is running at `http://localhost:8080/v1`, **any tool that supports the OpenAI API works immediately** — just change `base_url`.

### Python — openai SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

Install: `pip install openai`

---

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
)
print(llm.invoke("What is the capital of France?").content)
```

Install: `pip install langchain-openai`

---

### LlamaIndex

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    api_base="http://localhost:8080/v1",
    api_key="local",
    is_chat_model=True,
)
response = llm.complete("What is the capital of France?")
print(response.text)
```

Install: `pip install llama-index-llms-openai-like`

---

### CrewAI

```python
from crewai import LLM

llm = LLM(
    model="openai/mlx-community/Llama-3.2-3B-Instruct-4bit",
    base_url="http://localhost:8080/v1",
    api_key="local",
)
```

Install: `pip install crewai`

---

### AutoGen

```python
llm_config = {
    "config_list": [{
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "base_url": "http://localhost:8080/v1",
        "api_key": "local",
    }]
}
```

Install: `pip install pyautogen`

---

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Available Models

### What works

`mlx-lm` can run **any model on Hugging Face** that uses a supported transformer architecture — you are not limited to `mlx-community`. Two categories:

| Category | How it works | Start time |
|---|---|---|
| **MLX-format** (pre-converted, e.g. `mlx-community/*`) | Downloaded and run directly | Fast |
| **Standard HF weights** (e.g. `mistralai/*`, `google/*`, `meta-llama/*`) | `mlx-lm` converts them on first load | Slower first run |

For best startup performance, prefer pre-converted MLX models from [mlx-community](https://huggingface.co/mlx-community) or [lmstudio-community](https://huggingface.co/lmstudio-community). Note that `mlx-lm` does **not** load GGUF files — use MLX weights or original safetensors checkpoints instead.

### Curated list (shown in the interactive menu)

The menu auto-detects your RAM and flags each model **[green]✓ fits[/green] / [yellow]~ tight[/yellow] / [red]✗ too large[/red]**. Quick pick by your Mac's unified memory:

| Your Mac (unified RAM) | Recommended model | On-disk | Why |
|---|---|---|---|
| **8 GB** | Llama 3.2 3B ⭐ | ~2 GB | Fast, capable lightweight starter |
| **16 GB** | Gemma 4 12B (OptiQ 4-bit) | ~7 GB | Multimodal, near-flagship quality |
| **24 GB** | Gemma 4 12B (8-bit) | ~13 GB | Higher-fidelity 8-bit weights |
| **32 GB** | Qwen3.6 27B (OptiQ 4-bit) | ~18 GB | Flagship dense reasoning & coding |
| **48 GB** | Qwen3.6 35B-A3B (OptiQ 4-bit) ⭐ | ~25 GB | MoE flagship: 3B active = fast, 256K+ context |
| **64 GB+** | Qwen3.6 35B-A3B (8-bit) | ~37 GB | Maximum-fidelity flagship |

#### MLX Community — pre-converted, recommended

| Model | Size | Notes |
|---|---|---|
| Llama 3.2 1B | ~0.7 GB | Ultra-fast, runs on any 8 GB Mac |
| **Llama 3.2 3B ⭐** | ~2 GB | **Best lightweight starter** |
| Qwen3 4B | ~2.4 GB | Strong small reasoning & coding, 256K context |
| Gemma 4 12B (OptiQ 4-bit) | ~7 GB | Multimodal — 16 GB sweet spot |
| Gemma 4 12B (8-bit) | ~13 GB | Higher-fidelity 8-bit |
| Qwen3.6 27B (OptiQ 4-bit) | ~18 GB | Flagship dense (needs 32 GB+) |
| Qwen3.6 35B-A3B (OptiQ 4-bit) | ~25 GB | MoE flagship (best on 48 GB) |

> **OptiQ** — calibrated mixed-precision quant (sensitive layers kept at higher precision); noticeably better quality than uniform 4-bit at a similar size.
> **A3B** — Mixture-of-Experts with ~3B active parameters per token, so it runs at small-model speed while keeping large-model quality and a small KV cache (great for long context).

#### Other sources — also fully supported

`mlx-lm` runs any MLX-format repo, plus standard Hugging Face safetensors weights (converted on first load). It does **not** load GGUF files.

| Model ID | Notes |
|---|---|
| `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` | Unsloth Dynamic (UD-MLX) MoE quant — pre-converted, no conversion needed |
| `Qwen/Qwen3-4B-Instruct-2507` | Official Qwen weights — mlx-lm converts on first load |
| `google/gemma-4-12b-it` | Official Google weights — mlx-lm converts on first load |

Use the **"Enter any Hugging Face model ID"** option in the menu to run models not in the list.

### Monitoring CPU, GPU and memory

Open a second terminal while the server is running:

```bash
uv run monitor.py
```

Shows a live dashboard (refreshes every second):

| Row | What it shows | How it's calculated |
|---|---|---|
| CPU | Overall system CPU across all cores | `psutil.cpu_percent()` |
| GPU Metal | Apple GPU utilization (no `sudo` needed) | `ioreg` "Device Utilization %" |
| RAM | Total unified memory used / available — the headline number | `psutil.virtual_memory()` used / total |
| Wired RAM | Memory locked in RAM (non-pageable; includes GPU/Metal), as reported by macOS | `psutil.virtual_memory().wired` |
| Memory pressure | `normal` / `warning` / `critical` | parses macOS `memory_pressure` |
| Server proc RAM | Memory used by the `mlx_lm server` process tree, as reported by macOS | sum of each process's RSS (Resident Set Size) over the process + children |
| Server proc CPU | CPU used by the server process tree | sum of `cpu_percent()` over the process + children |
| Server :8080 | `● ready` / `◐ loading…` / `○ offline` | TCP check + HTTP `GET /health` |

> **Unified memory.** Apple Silicon shares **one** memory pool between CPU and GPU. **RAM** is the whole machine; **Wired RAM** is the locked (non-pageable) portion of it. **Server proc RAM** is the memory macOS reports for the server process. For judging total memory use, read **RAM** and **Wired RAM** (system-wide); treat **Server proc RAM** as a per-process figure reported by macOS.

> **Server status:** `◐ loading…` means the port is open but `/health` isn't responding yet (still starting/loading); `● ready` means the HTTP server is answering. The check uses the cheap `/health` endpoint (not `/v1/models`, which rescans the Hugging Face cache on every call).


For deeper Apple Neural Engine (ANE) and per-core (E/P cluster) breakdown:

```bash
brew install asitop
sudo asitop
```

---

## Testing effective context length (`context_test.py`)

Every model advertises a maximum context window (e.g. 128K or 256K tokens), but the **usable** window is almost always smaller — limited by your RAM, the server's KV-cache budget, and the model's real ability to *recall* information buried in a long prompt. `context_test.py` measures both so you can pick a safe working size.

### What it does

It runs a **"needle-in-a-haystack"** sweep. For each target prompt size it:

1. Generates a random 8-character passcode (the *needle*).
2. Inserts it at **five depths** in the prompt — `start`, `25%`, `50%`, `75%`, `end` — padded with varied filler text up to the target size (the *haystack*).
3. Asks the model to recall the exact passcode.
4. Records the **actual token count** the server saw, whether each depth was recalled correctly, and how long it took.

This finds **two different limits**:

| Limit | Meaning |
|---|---|
| **Hard limit** | First size where the server returns an error (out of memory / KV cache exceeded). |
| **Effective limit** | First size where recall flips from ✓ to ✗ — the model "loses" the needle even though the request succeeds (the *lost-in-the-middle* effect). |

### How to run

Start your server first (`uv run start.py`), then in a second terminal:

```bash
# Test the first model the server reports, with the default size sweep
uv run context_test.py

# Pick a specific model (servers can host several)
uv run context_test.py --model mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit

# Custom sizes and port
uv run context_test.py --port 8080 --sizes 2000,8000,32000,64000,131072
```

| Flag | Default | Purpose |
|---|---|---|
| `--model` | first model from `/v1/models` | Which loaded model to test |
| `--sizes` | `1000…131072` | Comma-separated approximate prompt sizes (tokens) |
| `--port` / `--host` | `8080` / `localhost` | Where the server is listening |
| `--timeout` | `600` | Per-request timeout in seconds (large prompts are slow) |

### How to read the results

```
   real_tok   start    25%    50%    75%    end    secs
  ------------------------------------------------------
      2,626       ✓      ✓      ✓      ✓      ✓     30.9
     10,326       ✓      ✓      ✓      ✓      ✓     81.2
     40,999       ✓      ✓      ✓      ✗      ✗    392.5
```

- **`real_tok`** — the *actual* prompt token count the server reported (not the approximate target).
- **✓ / ✗** — whether the needle was recalled correctly at that depth. ✗ at deeper positions while ✓ at the start is the classic *lost-in-the-middle* pattern.
- **`secs`** — total time for all five depth probes at that size. This climbs fast — long prompts are usually **latency-bound, not memory-bound**.
- **`ERR`** — the server rejected the request (hard limit). The test stops and prints the error.

The final **REPORT** summarizes:

- **Reliable at ALL depths up to** — largest size where every depth passed.
- **First degradation** — where recall started failing, and which depths.
- **Hard server/RAM limit** — where the server errored, if reached.
- **RECOMMENDED CONTEXT SIZE** — ~80 % of the largest all-pass size (rounded to a clean power of two), plus a ready-to-use `--max-kv-size` launch example.

### What to do with the results

1. **Cap the KV cache** to a safe size when launching the server, e.g.:
   ```bash
   python -m mlx_lm server --model <model> --max-kv-size 32768
   ```
   This bounds memory use and avoids slow, unreliable giant prompts.
2. **Keep working prompts well under the effective limit** — always leave room for the model's reply; don't fill the entire window.
3. **For interactive use, stay small** (≈ ≤ 16K tokens). Recall may hold far higher, but per-request latency grows steeply.

> **Thinking-model caveat:** "reasoning" models (e.g. Qwen3.6) spend hidden tokens thinking before answering. If `max_tokens` is too small the visible answer can be truncated and show a false ✗. The tester already accounts for this (uses `max_tokens=256` and falls back to `reasoning_content` / `text`), but keep it in mind if results look noisy at one depth.

---

## Agentic IDE Setup Agents (recommended)

This repo ships two agents in `.github/agents/` that any **agentic IDE** (VS Code + GitHub Copilot, Cursor, Windsurf, Cline, or any tool that can reference files as context) can pick up and run. They are the **fastest and recommended way** to get started — they detect your hardware, run all commands for you, and guide you step by step.

### Local AI Setup agent

> **When to use:** setting up the repo for the first time, starting the server, starting the monitor, getting a model recommendation for your specific Mac.

In your agentic IDE, select the **Local AI Setup** agent (or attach `.github/agents/local-ai-setup.agent.md` as context) and type:

```
Set up this repo and start the server
```
```
Which model should I run on my 16 GB Mac?
```
```
Start the monitoring dashboard
```

The agent will detect your chip and RAM, run the right commands directly, verify success at each step, and hand off to the **MLX Model Finder** agent if you need help picking a model.

---

### MLX Model Finder agent

> **When to use:** you have a specific base model in mind and want to find the best pre-converted MLX variant for your RAM.

Select the **MLX Model Finder** agent (or attach `.github/agents/mlx-model-finder.agent.md` as context) and type the base model name:

```
Llama 3.2 3B
```
```
Best Mistral 7B for 16 GB RAM
```
```
Qwen 2.5 Coder 7B — which quantization?
```

The agent browses `mlx-community`, `bartowski`, `unsloth`, and `lmstudio-community` on Hugging Face in real time and returns a copy-pasteable model ID ranked by quality and download count.

You can also run it from the terminal without an agentic IDE:

```bash
uv run find_model.py "Llama 3.2 3B"
uv run find_model.py "Mistral 7B"
uv run find_model.py "Qwen 2.5"
```

---

## Troubleshooting

**Model is slow or you see a wired memory warning**

```bash
sudo sysctl iogpu.wired_limit_mb=<model_size_in_MB + 2048>
```

**Port already in use**  
The script lets you choose any port. Default is `8080`.

**Large models (70B+)**  
Requires macOS 15 or later for best performance.

---

## References

- [MLX LM](https://github.com/ml-explore/mlx-lm) — LLM engine powering this server
- [WWDC 2026 — Run local agentic AI on the Mac using MLX](https://developer.apple.com/videos/play/wwdc2026/232/)
- [MLX Community on Hugging Face](https://huggingface.co/mlx-community) — Model library
- [MLX Framework](https://ml-explore.github.io/mlx/) — Apple's ML framework for Apple Silicon
 
