import re

import requests

from rag_service.config import QUERY_REWRITE, VECTOR_STORE


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_rewrite_prompt(question: str) -> str:
    return (
        "只改写【】内问题为一句中文陈述式检索句，40字以内。"
        "保留实体/业务对象/条件；不要拆成关键词；不回答不解释，只输出一句话。\n"
        f"问题：【{question}】\n改写："
    )


def _call_ollama_rewrite(prompt: str) -> str:
    resp = requests.post(
        f"{VECTOR_STORE.ollama_base_url.rstrip('/')}/api/chat",
        json={
            "model": QUERY_REWRITE.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 64},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Query rewrite failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("Unexpected query rewrite response schema from /api/chat")
    return _strip_think_tags(content)


def _call_openai_compatible_rewrite(prompt: str) -> str:
    if not QUERY_REWRITE.base_url:
        raise RuntimeError("RAG_QUERY_REWRITE_BASE_URL is required when RAG_QUERY_REWRITE_PROVIDER=openai.")
    if not QUERY_REWRITE.api_key:
        raise RuntimeError("RAG_QUERY_REWRITE_API_KEY is required when RAG_QUERY_REWRITE_PROVIDER=openai.")
    if not QUERY_REWRITE.model:
        raise RuntimeError("RAG_QUERY_REWRITE_MODEL is required when RAG_QUERY_REWRITE_PROVIDER=openai.")

    payload = {
        "model": QUERY_REWRITE.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "max_tokens": 48,
    }
    if QUERY_REWRITE.thinking:
        payload["thinking"] = {"type": QUERY_REWRITE.thinking}

    resp = requests.post(
        f"{QUERY_REWRITE.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {QUERY_REWRITE.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Query rewrite failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected query rewrite response schema from /chat/completions") from exc
    if not isinstance(content, str):
        raise RuntimeError("Unexpected query rewrite response schema from /chat/completions: content must be string.")
    return _strip_think_tags(content)


def _call_rewrite_model(prompt: str) -> str:
    provider = QUERY_REWRITE.provider.strip().lower()
    if provider == "ollama":
        return _call_ollama_rewrite(prompt)
    if provider == "openai":
        return _call_openai_compatible_rewrite(prompt)
    raise RuntimeError(f"Unsupported query rewrite provider: {QUERY_REWRITE.provider}")


def rewrite_query(question: str) -> str:
    original = question.strip()
    if not original:
        raise ValueError("question must not be empty")

    rewritten = _call_rewrite_model(build_rewrite_prompt(original))
    if not rewritten:
        raise RuntimeError("Query rewrite returned empty content.")
    return rewritten
