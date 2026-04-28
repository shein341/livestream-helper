"""问答日志，按类型分文件存储：
- logs/query.jsonl      原始问题 + 改写后 query + 耗时
- logs/chunks.jsonl    命中的 chunk（含各阶段分数）
- logs/prompt.jsonl    发给 LLM 的完整 prompt
- logs/answer.jsonl    LLM 返回内容
所有文件通过 request_id + ts 关联。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_QUERY = LOG_DIR / "query.jsonl"
_LOG_CHUNKS = LOG_DIR / "hit_chunk.jsonl"
_LOG_PROMPT = LOG_DIR / "llm_prompt.jsonl"
_LOG_ANSWER = LOG_DIR / "answer.jsonl"
_lock = threading.Lock()


def _chunk_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "source": row.get("source"),
        "heading_path": row.get("heading_path", ""),
        "char_count": row.get("char_count"),
        # 各阶段分数
        "dense_rank": row.get("dense_rank"),
        "bm25_rank": row.get("bm25_rank"),
        "rrf_score": row.get("rrf_score"),
        "distance": row.get("distance"),
        "bm25_score": row.get("bm25_score"),
        "rerank_score": row.get("rerank_score"),
        # chunk 原文
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
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1. query.jsonl：原始问题 + 改写 + 耗时
    rec_query = {
        "ts": ts,
        "request_id": request_id,
        "question": question,
        "rewritten_query": rewritten_query,
        "fallback": fallback,
        # 召回/过滤统计
        "retrieved_count": len(retrieved_rows),
        "context_count": len(context_rows),
    }

    # 2. chunks.jsonl：命中的 chunk（含各阶段分数）
    rec_chunks = {
        "ts": ts,
        "request_id": request_id,
        "question": question,
        "fallback": fallback,
        "chunks": [_chunk_record(r) for r in context_rows],
    }

    # 3. prompt.jsonl：发给 LLM 的完整 prompt
    rec_prompt = {
        "ts": ts,
        "request_id": request_id,
        "question": question,
        "prompt": prompt,
    }

    # 4. answer.jsonl：LLM 返回内容
    rec_answer = {
        "ts": ts,
        "request_id": request_id,
        "duration_ms": duration_ms,
        "answer": answer,
        "fallback": fallback,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        with _LOG_QUERY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_query, ensure_ascii=False) + "\n")
        with _LOG_CHUNKS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_chunks, ensure_ascii=False) + "\n")
        with _LOG_PROMPT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_prompt, ensure_ascii=False) + "\n")
        with _LOG_ANSWER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec_answer, ensure_ascii=False) + "\n")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
