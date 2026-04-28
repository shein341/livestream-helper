from __future__ import annotations

import os
import subprocess
import sys

import requests


def check_ollama(base_url: str) -> None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}: {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"Ollama health check failed ({response.status_code}): {response.text}")


def ensure_vector_index(base_url: str) -> None:
    if os.path.exists("chroma_db/chroma.sqlite3"):
        print("[app] existing chroma_db found", flush=True)
        return
    if not os.path.exists("rag_chunks_realistic.jsonl"):
        print("[app] no chroma_db and no rag_chunks_realistic.jsonl; upload docs through the UI before chatting", flush=True)
        return

    embedding_model = os.environ.get("RAG_EMBEDDING_MODEL", "bge-m3:latest")
    print("[app] chroma_db is missing; building vector index from rag_chunks_realistic.jsonl", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_service.ingestion.embedder",
            "--skip-pull",
            "--chunk-file",
            "rag_chunks_realistic.jsonl",
            "--chroma-dir",
            "chroma_db",
            "--ollama-url",
            base_url,
            "--embedding-model",
            embedding_model,
        ],
        check=True,
    )


def main() -> int:
    base_url = os.environ.get("RAG_OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    print(f"[app] checking Ollama at {base_url}", flush=True)
    check_ollama(base_url)
    ensure_vector_index(base_url)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rag_service.api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
