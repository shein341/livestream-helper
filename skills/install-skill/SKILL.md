---
name: install-skill
description: Use when installing, configuring, or verifying shein341/livestream-helper from GitHub with Docker, Ollama, Python, DeepSeek/OpenAI-compatible keys, README steps, or first-run environment checks.
---

# Install Skill

## Overview

Detect the local environment, let the user choose between Docker and local Python, then install only what is actually missing.

## Core Rule

Always preflight before installing. Report every check result. Never assume. Never install what is already present.

Never write real API keys into tracked files. Use `.env` as a local untracked file only.

---

# Phase 1 — Environment Detection (always run first)

Run all checks and report results before asking the user to choose.

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

# Phase 2 — Ask User to Choose

Report the detection results, then ask:

```
=== Environment Summary ===

OS: <os>
Docker: [found+running / found+not running / not found]
Ollama: [found+reachable / found+not reachable / not found]
Models: <list or "none">
Python: [3.x / N/A]

Choose deployment path:

  [D] Docker部署  — 需要较长时间构建镜像（约5-10分钟），
                    但环境隔离、一致性好、，一条命令启动
  [L] 本地Python  — 依赖本地环境，若依赖齐全则更快启动

请输入 D 或 L：
```

If user chooses **D**: go to PATH A.
If user chooses **L**: go to PATH B.
If user chooses neither: re-prompt.

---

# PATH A — Docker Deployment

> **注意：** Docker 构建首次需下载大量 Python/CUDA 依赖，耗时约 5-10 分钟（视网络而定）。

## Step A1 — Model Check

```bash
ollama list | grep -q "bge-m3:latest" || echo "MISSING: bge-m3:latest"
ollama list | grep -q "qwen3.5:4b" || echo "MISSING: qwen3.5:4b (only if RAG_QUERY_REWRITE_PROVIDER=ollama)"
```

Pull only what's missing:

```bash
ollama list | grep -q "bge-m3:latest" || ollama pull bge-m3:latest
grep -q "RAG_QUERY_REWRITE_PROVIDER=ollama" .env && \
  ollama list | grep -q "qwen3.5:4b" || ollama pull qwen3.5:4b
```

Do NOT `ollama pull deepseek-v4-flash` — it is an OpenAI-compatible API model, not an Ollama model.

## Step A2 — Environment File

```bash
[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo ".env created"; }
grep -q "your_answer_api_key_here\|your_deepseek_key_here" .env && \
  echo "ACTION REQUIRED: open .env and fill in your API keys" || echo ".env: keys present"
```

## Step A3 — Build (first time only)

```bash
docker compose build
```

> 预计耗时 5-10 分钟。后续启动使用 `docker compose up -d` 可跳过构建。

## Step A4 — Start

```bash
docker compose up -d
```

## Step A5 — Verify

```bash
curl -s http://localhost:8000/health && echo " Health OK"
curl -s http://localhost:8000/swagger >/dev/null && echo "Swagger UI accessible"
```

If health fails: `docker compose logs -f --tail=30`

---

# PATH B — Local Python Deployment

## Step B1 — Python Version Check

```bash
PYTHON_VERSION=$(python -c "import sys; print(sys.version_info.minor)")
[ "$PYTHON_VERSION" -ge 11 ] && echo "Python 3.$PYTHON_VERSION: OK" || echo "Python 3.$PYTHON_VERSION: too old (need 3.11+)"
```

If Python < 3.11: offer to install via:
- Windows: https://www.python.org/downloads/ or `winget install Python.3.11`
- Mac: `brew install python@3.11`
- Linux: `apt install python3.11`

## Step B2 — Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Step B3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If `chroma-hnswlib` or `tokenizers` fail on Python 3.13+: unsupported environment, fall back to Docker or use Python 3.11.

## Step B4 — Model Check

Same as Step A1.

## Step B5 — Environment File

Same as Step A2.

## Step B6 — Start

```bash
uvicorn rag_service.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Step B7 — Verify

```bash
curl -s http://localhost:8000/health && echo " Health OK"
curl -s http://localhost:8000/swagger >/dev/null && echo "Swagger UI accessible"
```

---

# Phase 3 — First Run Verification

After startup, run a real query:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "主播提现前需要满足哪些条件？"}'
```

Expected: JSON with `answer`, `references`, `pipeline`, `rewritten_query`, `fallback`.

If `fallback: true`: confidence was too low, check that docs were ingested:

```bash
curl -s http://localhost:8000/docs
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
Dependencies: [installed / failed]
.env: [created / existed with keys / existed without keys]
App: [running at http://localhost:8000 / FAILED]

Next: Open http://localhost:8000/swagger to try the API
```
