---
name: install-skill
description: Use when installing, configuring, or verifying shein341/livestream-helper from GitHub with Docker, Ollama, DeepSeek/OpenAI-compatible keys, README steps, or first-run environment checks.
---

# Install Skill

## Overview

Install `shein341/livestream-helper` by checking the user's machine first, reusing what already exists, then following the normal README startup path. Do not turn installation into a blind `docker build` or `pip install` run.

## Core Rule

Always preflight before installing. Report what is already present, what is missing, and what will be downloaded before running expensive installs or model pulls.

Never write real API keys into tracked files, commit keys, or echo keys in logs. Use `.env` only as a local untracked file and ask the user to paste their own key there.

## Workflow

1. Clone or open the project.
   - If the repo is absent, clone `https://github.com/shein341/livestream-helper`.
   - If the repo already exists, inspect it in place and do not overwrite user changes.

2. Read the install surface.
   - Read `README.md`, `.env.example`, `run.ps1`, `run.sh`, `docker-compose.yml`, `Dockerfile`, and `requirements.txt`.
   - Confirm whether `run.ps1`/`run.sh` only pull the query rewrite model when `RAG_QUERY_REWRITE_PROVIDER=ollama`.
   - If scripts would pull `deepseek-v4-flash` through Ollama, stop and fix or warn before running them.

3. Check local prerequisites and cache before installing:

   Windows PowerShell:
   ```powershell
   git --version
   docker --version
   docker compose version
   docker info
   ollama --version
   Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 10
   ollama list
   python --version
   docker images
   docker builder du
   ```

   macOS/Linux shell:
   ```bash
   git --version
   docker --version
   docker compose version
   docker info
   ollama --version
   curl -fsS http://127.0.0.1:11434/api/tags
   ollama list
   python --version
   docker images
   docker builder du
   ```

4. Decide what to install.
   - Install Docker Desktop only if Docker or Compose is missing or not running.
   - Install Ollama only if missing.
   - Start Ollama if installed but not reachable at `127.0.0.1:11434`.
   - Pull only missing Ollama models.
   - Required default model: `bge-m3:latest`.
   - Also pull `qwen3.5:4b` only when using local Ollama query rewrite.
   - Do not pull `deepseek-v4-flash` with Ollama; it is an OpenAI-compatible API model.

5. Prepare `.env`.
   - If `.env` is absent, copy `.env.example` to `.env`.
   - Ask the user to open `.env` and enter their own API key values.
   - For DeepSeek, use:

   ```env
   RAG_ANSWER_BASE_URL=https://api.deepseek.com
   RAG_ANSWER_MODEL=deepseek-v4-flash
   RAG_ANSWER_API_KEY=your_deepseek_key_here
   RAG_ANSWER_REASONING_SPLIT=false
   RAG_ANSWER_THINKING=disabled

   RAG_QUERY_REWRITE_PROVIDER=openai
   RAG_QUERY_REWRITE_BASE_URL=https://api.deepseek.com
   RAG_QUERY_REWRITE_API_KEY=your_deepseek_key_here
   RAG_QUERY_REWRITE_MODEL=deepseek-v4-flash
   RAG_QUERY_REWRITE_THINKING=disabled
   ```

6. Start normally.
   - Windows: `.\run.ps1`
   - macOS/Linux: `chmod +x ./run.sh && ./run.sh`
   - Reuse `--no-build` / `-NoBuild` only after a successful first build.

7. Verify.
   - Check `http://localhost:8000/health`.
   - Check `http://localhost:8000/swagger`.
   - Run a real `/chat` request after the app starts.
   - Run tests with `python -m pytest -q` only if local Python dependencies are installed; otherwise run the equivalent inside Docker.

## Slow Install Signals

`FlagEmbedding` depends on `torch`, `transformers`, and related ML packages. Docker builds may download large wheels, especially if pip resolves a GPU-enabled Torch build. Before changing dependencies, report the expected download size and whether the current Docker cache already has them.

On Windows with Python 3.13, local `pip install -r requirements.txt` may fail because `chroma-hnswlib` or `tokenizers` falls back to native compilation. Prefer Docker or Python 3.11 unless the user wants local Python setup.

## Completion Report

End with:

- installed/reused tools
- models already present or pulled
- whether `.env` was prepared without exposing keys
- startup URL
- verification commands and results
- any README or script gaps found
