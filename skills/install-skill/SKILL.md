---
name: install-skill
description: Use when installing, configuring, or verifying shein341/livestream-helper from GitHub with Docker, Ollama, DeepSeek/OpenAI-compatible keys, README steps, or first-run environment checks.
---

# Install Skill

## Overview

Install `shein341/livestream-helper` by detecting the local environment, reporting missing dependencies, and performing only the necessary setup steps. Works on Windows (PowerShell) and macOS/Linux (bash/zsh).

## Core Rule

Always preflight before installing. Report what is already present, what is missing, and what will be downloaded before running expensive installs or model pulls.

Never write real API keys into tracked files, commit keys, or echo keys in logs. Use `.env` only as a local untracked file and ask the user to paste their own key there.

## Environment Detection

Run this detection once at the start:

```bash
# Detect OS and shell
IS_WINDOWS=false; IS_MAC=false; IS_LINUX=false
case "$(uname -s)" in
  CYGWIN*|MINGW*|MSYS*|Windows*) IS_WINDOWS=true ;;
  Darwin*)                        IS_MAC=true ;;
  *)                              IS_LINUX=true ;;
esac
echo "OS: $(uname -s), Windows=$IS_WINDOWS, Mac=$IS_MAC, Linux=$IS_LINUX"
```

## Step 1 — Check Prerequisites

Run all checks regardless of OS. Report each one individually.

### Docker

```bash
docker --version
docker compose version
docker info >/dev/null 2>&1 && echo "Docker: running" || echo "Docker: not running or not installed"
```

**If missing or not running:** Install Docker Desktop (Windows/macOS) or Docker Engine (Linux). Do not proceed until Docker is working.

### Ollama

```bash
ollama --version
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && echo "Ollama: reachable" || echo "Ollama: not reachable"
```

**If missing:** Install Ollama from https://ollama.com
**If not reachable:** Run `ollama serve` to start it.

## Step 2 — Check Required Models

```bash
ollama list
```

| Model | Required | Condition |
|---|---|---|
| `bge-m3:latest` | Yes | Always needed for embedding |
| `qwen3.5:4b` | Conditional | Only if `RAG_QUERY_REWRITE_PROVIDER=ollama` |

**If `bge-m3:latest` is missing:** Run `ollama pull bge-m3:latest`
**If `qwen3.5:4b` is missing and `RAG_QUERY_REWRITE_PROVIDER=ollama`:** Run `ollama pull qwen3.5:4b`

Do NOT pull `deepseek-v4-flash` with Ollama — it is an OpenAI-compatible API model, not an Ollama model.

## Step 3 — Prepare Environment File

```bash
if [ ! -f .env ]; then
  cp .env.example .env
  echo ".env created from .env.example — open it and fill in your API keys"
else
  echo ".env already exists"
fi
```

Ask the user to open `.env` and fill in at minimum:

```env
RAG_ANSWER_API_KEY=your_api_key_here
RAG_QUERY_REWRITE_API_KEY=your_api_key_here
```

## Step 4 — Build and Start

```bash
if $IS_WINDOWS; then
  .\run.ps1
else
  chmod +x ./run.sh && ./run.sh
fi
```

Use `--no-build` / `-NoBuild` flag on subsequent runs to reuse the Docker image layer cache.

## Step 5 — Verify

```bash
curl -s http://localhost:8000/health && echo "Health OK"
curl -s http://localhost:8000/swagger >/dev/null && echo "Swagger UI accessible"
```

If health check fails, check logs:

```bash
docker compose logs -f
```

## Completion Report

End every run with:

- tools found / installed
- models found / pulled (and why)
- whether `.env` was prepared
- startup URL
- verification results
- any steps that were skipped and why
