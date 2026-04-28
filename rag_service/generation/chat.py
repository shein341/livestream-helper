import argparse
import json
from pathlib import Path

import requests

from rag_service.confidence import confidence_fallback_answer, filter_context_rows
from rag_service.config import ANSWER, MODELS, VECTOR_STORE
from rag_service.retrieval.hybrid import query_top_k


def build_context_blocks(rows: list[dict], max_context_chars: int) -> list[str]:
    blocks: list[str] = []
    used = 0
    for idx, row in enumerate(rows, start=1):
        source = row.get("source", "")
        heading = row.get("heading_path", "")
        chunk_id = row.get("chunk_id")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        block = (
            f"[{idx}] source={source} heading={heading} chunk_id={chunk_id}\n"
            f"{text}"
        )
        if blocks and used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)
    return blocks


def build_generation_prompt(query: str, context_blocks: list[str]) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context."
    return (
        "You are a strict policy QA assistant.\n"
        "Rules:\n"
        "1) Only use the provided context.\n"
        "2) Every conclusion must include citations like [1][2].\n"
        "3) If context is insufficient, explicitly say so.\n\n"
        f"User question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer in Chinese."
    )


def build_chat_completion_payload(
    *,
    model_name: str,
    prompt: str,
    stream: bool,
    reasoning_split: bool,
    thinking: str,
) -> dict:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    if reasoning_split:
        payload["reasoning_split"] = True
    if thinking:
        payload["thinking"] = {"type": thinking}
    return payload


def call_openai_compatible_generate(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    reasoning_split: bool = True,
    thinking: str = "",
) -> str:
    if not api_key:
        raise RuntimeError("RAG_ANSWER_API_KEY is required for answer generation.")
    if not base_url:
        raise RuntimeError("RAG_ANSWER_BASE_URL is required for answer generation.")
    if not model_name:
        raise RuntimeError("RAG_ANSWER_MODEL is required for answer generation.")
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_chat_completion_payload(
            model_name=model_name,
            prompt=prompt,
            stream=False,
            reasoning_split=reasoning_split,
            thinking=thinking,
        ),
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ChatCompletion API failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response schema from /chat/completions") from exc
    if not isinstance(content, str):
        raise RuntimeError("Unexpected response schema from /chat/completions: content must be string.")
    return content.strip()


def call_answer_generate(prompt: str) -> str:
    return call_openai_compatible_generate(
        base_url=ANSWER.base_url,
        api_key=ANSWER.api_key,
        model_name=ANSWER.model,
        prompt=prompt,
        reasoning_split=ANSWER.reasoning_split,
        thinking=ANSWER.thinking,
    )


def stream_answer_generate(prompt: str):
    if not ANSWER.api_key:
        raise RuntimeError("RAG_ANSWER_API_KEY is required for answer generation.")
    if not ANSWER.base_url:
        raise RuntimeError("RAG_ANSWER_BASE_URL is required for answer generation.")
    if not ANSWER.model:
        raise RuntimeError("RAG_ANSWER_MODEL is required for answer generation.")
    resp = requests.post(
        f"{ANSWER.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {ANSWER.api_key}", "Content-Type": "application/json"},
        json=build_chat_completion_payload(
            model_name=ANSWER.model,
            prompt=prompt,
            stream=True,
            reasoning_split=ANSWER.reasoning_split,
            thinking=ANSWER.thinking,
        ),
        stream=True,
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ChatCompletion stream failed ({resp.status_code}): {resp.text}")
    pending = ""
    for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
        if not chunk:
            continue
        pending += chunk
        while True:
            nl = pending.find("\n")
            if nl < 0:
                break
            raw_line = pending[:nl]
            pending = pending[nl + 1 :]
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                return
            data = json.loads(line)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not isinstance(delta, dict):
                continue
            token = delta.get("content")
            if isinstance(token, str) and token:
                yield token


def format_references(rows: list[dict]) -> list[str]:
    refs: list[str] = []
    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        refs.append(f"[{idx}] {text}")
    return refs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG chat with hybrid retrieval and mandatory reranker.")
    parser.add_argument("query", help="User query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=30)
    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--source", default=None)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--ollama-url", default=VECTOR_STORE.ollama_base_url)
    parser.add_argument("--embedding-model", default=MODELS.embedding)
    parser.add_argument("--reranker-model", default=MODELS.reranker)
    parser.add_argument("--chroma-dir", type=Path, default=VECTOR_STORE.chroma_dir)
    parser.add_argument("--collection", default=VECTOR_STORE.collection_name)
    parser.add_argument("--chunk-file", type=Path, default=VECTOR_STORE.chunk_file)
    parser.add_argument("--json", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_rag_chat(args: argparse.Namespace) -> dict:
    rows = query_top_k(
        query=args.query,
        top_k=args.top_k,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_file=args.chunk_file,
        ollama_url=args.ollama_url,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        rerank_top_n=args.rerank_top_n,
        source=args.source,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )
    fallback_answer = confidence_fallback_answer(rows)
    if fallback_answer is not None:
        return {"answer": fallback_answer, "references": [], "rows": rows}

    context_rows = filter_context_rows(rows)
    context_blocks = build_context_blocks(rows=context_rows, max_context_chars=args.max_context_chars)
    prompt = build_generation_prompt(query=args.query, context_blocks=context_blocks)
    answer = call_answer_generate(prompt=prompt)
    references = format_references(context_rows[: len(context_blocks)])
    return {"answer": answer, "references": references, "rows": rows}


def main() -> int:
    args = parse_args()
    result = run_rag_chat(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(result["answer"])
    print("")
    print("References:")
    for ref in result["references"]:
        print(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
