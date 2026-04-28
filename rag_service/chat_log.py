"""问答日志拆成三个 JSONL 文件，用 request_id + ts 关联。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_QUERY_REWRITE = LOG_DIR / "query_rewrite.jsonl"
LOG_RERANK_HITS = LOG_DIR / "rerank_hits.jsonl"
LOG_PROMPT_ANSWER = LOG_DIR / "prompt_answer.jsonl"
_lock = threading.Lock()


def _chunk_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "source": row.get("source"),
        "rerank_score": row.get("rerank_score"),
        "text": row.get("text", ""),
    }


def write_record(
    *,
    request_id: str,
    question: str,
    rewritten_query: str,
    retrieved_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    prompt: str,
    answer: str,
    fallback: bool,
    started_at: float,
) -> None:
    duration_ms = int((time.time() - started_at) * 1000)
    ts = datetime.now().isoformat(timespec="seconds")

    rec_rewrite = {
        "ts": ts,
        "request_id": request_id,
        "question": question,
        "rewritten_query": rewritten_query,
    }
    rec_hits = {
        "ts": ts,
        "request_id": request_id,
        "question": question,
        "fallback": fallback,
        "retrieved_chunks": [_chunk_record(r) for r in retrieved_rows],
        "context_chunks": [_chunk_record(r) for r in context_rows],
    }
    rec_run = {
        "ts": ts,
        "request_id": request_id,
        "duration_ms": duration_ms,
        "question": question,
        "prompt": prompt,
        "answer": answer,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        with LOG_QUERY_REWRITE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_rewrite, ensure_ascii=False) + "\n")
        with LOG_RERANK_HITS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_hits, ensure_ascii=False) + "\n")
        with LOG_PROMPT_ANSWER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_run, ensure_ascii=False) + "\n")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
