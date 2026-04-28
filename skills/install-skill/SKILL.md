---
name: install-skill
description: Use when installing, configuring, or verifying shein341/livestream-helper from GitHub with Docker, Ollama, Python, DeepSeek/OpenAI-compatible keys, README steps, or first-run environment checks.
---

# Install Skill

## Overview

Detect the local environment, then choose the optimal path: **Docker** or **local Python**. No Docker? No problem — the skill adapts. Only install what is actually missing.

## Core Rule

Always preflight before installing. Report every check result. Never assume. Never install what is already present.

Never write real API keys into tracked files. Use `.env` as a local untracked file only.

---

# Phase 1 — Environment Detection

Run ALL of these regardless of platform. Record every result.

## OS Detection

```bash
case "$(uname -s)" in
  CYGWIN*|MINGW*|MSYS*|Windows*) OS=Windows ;;
  Darwin*)                        OS=Mac ;;
  *)                              OS=Linux ;;
esac
echo "OS: $OS"
```

## Docker

```bash
docker --version 2>/dev/null && echo "Docker: found" || echo "Docker: NOT found"
docker compose version 2>/dev/null && echo "Docker Compose: found" || echo "Docker Compose: NOT found"
docker info >/dev/null 2>&1 && echo "Docker: running" || echo "Docker: NOT running"
```

## Ollama

```bash
ollama --version 2>/dev/null && echo "Ollama: found" || echo "Ollama: NOT found"
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && echo "Ollama: reachable" || echo "Ollama: NOT reachable"
```

## Ollama Models

```bash
ollama list 2>/dev/null
```

## Python

```bash
python --version 2>/dev/null && echo "Python: found" || echo "Python: NOT found"
pip --version 2>/dev/null && echo "pip: found" || echo "pip: NOT found"
```

---

# Phase 2 — Decision Tree

```
START
├─ Docker available + running?
│   ├─ YES → PATH A: Docker deployment
│   └─ NO
│       ├─ Python 3.11+ available?
│       │   ├─ YES → PATH B: Local Python deployment
│       │   └─ NO → BLOCK: Need Docker or Python 3.11+
└─ Ollama available + reachable?
    ├─ YES → Use local embedding/rerank
    └─ NO → BLOCK: Need Ollama running at 127.0.0.1:11434
```

---

# PATH A — Docker Deployment

## Step A1 — Model Check (conditional pull)

Check which models are actually missing:

```bash
ollama list | grep -q "bge-m3:latest" || echo "MISSING: bge-m3:latest"
ollama list | grep -q "qwen3.5:4b" || echo "MISSING: qwen3.5:4b (only needed if RAG_QUERY_REWRITE_PROVIDER=ollama)"
```

Pull only what's missing:

```bash
# Always needed
ollama list | grep -q "bge-m3:latest" || ollama pull bge-m3:latest

# Only if using local query rewrite
grep -q "RAG_QUERY_REWRITE_PROVIDER=ollama" .env && \
  ollama list | grep -q "qwen3.5:4b" || ollama pull qwen3.5:4b
```

Do NOT `ollama pull deepseek-v4-flash` — it is an OpenAI-compatible API model, not an Ollama model.

## Step A2 — Environment File

```bash
[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo ".env created from .env.example"; }
grep -q "your_answer_api_key_here\|your_deepseek_key_here" .env && \
  echo "ACTION REQUIRED: Open .env and fill in your API keys" || echo ".env: keys present"
```

## Step A3 — Build

```bash
# First time: full build
docker compose build

# Subsequent runs: reuse image cache
docker compose build --no-cache  # only if dependencies changed
```

## Step A4 — Start

```bash
docker compose up -d
```

## Step A5 — Verify

```bash
curl -s http://localhost:8000/health && echo "Health OK"
curl -s http://localhost:8000/swagger >/dev/null && echo "Swagger UI accessible"
docker compose logs -f --tail=20  # if health fails
```

---

# PATH B — Local Python Deployment

## Step B1 — Python Version Check

```bash
PYTHON_VERSION=$(python -c "import sys; print(sys.version_info.minor)")
[ "$PYTHON_VERSION" -ge 11 ] && echo "Python 3.$PYTHON_VERSION: OK" || echo "Python 3.$PYTHON_VERSION: too old, need 3.11+"
```

If Python is too old, offer to install via:

- Windows: https://www.python.org/downloads/ or `winget install Python.3.11`
- Mac: `brew install python@3.11`
- Linux: `apt install python3.11` / `yum install python311`

## Step B2 — Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
echo "Virtual environment activated"
```

## Step B3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `chroma-hnswlib` or `tokenizers` fail to install on Python 3.13+, the environment is unsupported — fall back to Docker or use Python 3.11.

## Step B4 — Ollama Model Check (conditional pull)

Same logic as Step A1, but only needed if not using API-based embedding/rerank.

## Step B5 — Environment File

Same as Step A2.

## Step B6 — Pre-load Embedding Cache (optional, speeds up first query)

```bash
python -c "
from rag_service.ingestion.embedder import OllamaEmbedder
e = OllamaEmbedder()
print('Embedding cache warm:', e.base_url)
"
```

## Step B7 — Start

```bash
uvicorn rag_service.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Step B8 — Verify

```bash
curl -s http://localhost:8000/health && echo "Health OK"
curl -s http://localhost:8000/swagger >/dev/null && echo "Swagger UI accessible"
```

---

# Phase 3 — First Run Verification

After startup succeeds, run a real query:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "主播提现前需要满足哪些条件？"}'
```

Expected: JSON with `answer`, `references`, `pipeline`, `rewritten_query`, `fallback`.

If `fallback: true`, the confidence was too low — check that documents were ingested:

```bash
curl -s http://localhost:8000/docs  # list ingested docs
```

---

# Completion Report Template

```
=== Install Complete ===

Path: [Docker / Local Python]
OS: <os>
Docker: [found+running / not found / not running]
Ollama: [found+reachable / not found / not reachable]
Models pulled: <list or "none">
Python: [3.x / N/A]
Virtual env: [created / N/A]
Dependencies: [installed / N/A]
.env: [created / existed with keys / existed without keys]
App: [running at http://localhost:8000 / FAILED]

Next: Open http://localhost:8000/swagger to try the API
```
