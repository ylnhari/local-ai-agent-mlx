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

For best startup performance, prefer pre-converted MLX models from [mlx-community](https://huggingface.co/mlx-community) or [bartowski](https://huggingface.co/bartowski).

### Curated list (shown in the interactive menu)

#### MLX Community — pre-converted, recommended

| Model | Size | Notes |
|---|---|---|
| Llama 3.2 1B | ~0.7 GB | Ultra-fast, works on 8 GB RAM |
| **Llama 3.2 3B ⭐** | ~2 GB | **Best starter — recommended** |
| Phi 3.5 Mini | ~2.2 GB | Strong reasoning and coding |
| Gemma 3 4B | ~2.5 GB | Google's compact instruction model |
| Mistral 7B | ~4.1 GB | Excellent instruction following |
| Llama 3.1 8B | ~4.5 GB | High quality, highly versatile |
| Qwen 2.5 7B | ~4.3 GB | Coding and multilingual |
| DeepSeek R1 8B | ~5 GB | Chain-of-thought reasoning |
| Llama 3.3 70B | ~39 GB | Best quality (needs 48 GB+ RAM) |

#### Other sources — also fully supported

| Model ID | Notes |
|---|---|
| `bartowski/Llama-3.2-3B-Instruct-GGUF` | Popular community quantization |
| `unsloth/Llama-3.2-3B-Instruct` | Unsloth fine-tuned variant |
| `microsoft/Phi-3.5-mini-instruct` | Official bf16 weights — converted on first load |
| `google/gemma-3-4b-it` | Official bf16 weights — converted on first load |
| `mistralai/Mistral-7B-Instruct-v0.3` | Any standard HF checkpoint |

Use the **"Enter any Hugging Face model ID"** option in the menu to run models not in the list.

### Monitoring CPU, GPU and memory

Open a second terminal while the server is running:

```bash
uv run monitor.py
```

Shows a live dashboard (refreshes every second):

| Row | What it shows |
|---|---|
| CPU | Overall system CPU across all cores |
| GPU Metal | Apple GPU utilization via IOKit (no `sudo` needed) |
| RAM | Total unified memory used / available |
| Memory pressure | `normal` / `warning` / `critical` via `memory_pressure` |
| LLM RAM | RAM used by the `mlx_lm server` process specifically |
| LLM CPU | CPU used by the `mlx_lm server` process specifically |
| Server :8080 | Live ping — `● online` once the server is accepting requests |

For deeper Apple Neural Engine (ANE) and per-core (E/P cluster) breakdown:

```bash
brew install asitop
sudo asitop
```

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
