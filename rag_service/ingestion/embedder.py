import argparse
import json
import math
import subprocess
import unicodedata
from pathlib import Path

import chromadb
import requests

from rag_service.config import MODELS, VECTOR_STORE


def pull_ollama_model(model_name: str) -> None:
    print(f"Pulling ollama model: {model_name}")
    result = subprocess.run(
        ["ollama", "pull", model_name],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull model `{model_name}`: {result.stderr.strip()}")
    print(f"Model ready: {model_name}")


def read_jsonl_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            rows.append(row)
    return rows


def ollama_embed_batch(base_url: str, model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("texts must not be empty")

    embed_resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": model_name, "input": texts},
        timeout=120,
    )
    if embed_resp.status_code != 200:
        raise RuntimeError(f"Embedding API failed ({embed_resp.status_code}): {embed_resp.text}")

    data = embed_resp.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("Unexpected response schema from /api/embed")
    has_nan = any(
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
        for vec in embeddings
        if isinstance(vec, list)
        for v in vec
    )
    if has_nan:
        raise RuntimeError("Embedding API returned NaN/Inf values.")
    return embeddings


def chunked(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_retrieval_text(row: dict) -> str:
    heading_path = row.get("heading_path") or []
    if isinstance(heading_path, list):
        heading = " > ".join(str(item).strip() for item in heading_path if str(item).strip())
    else:
        heading = str(heading_path).strip()
    text = str(row.get("text", "")).strip()
    if heading and text:
        return f"标题：{heading}\n内容：{text}"
    return text or heading


def sanitize_text_for_embedding(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Remove control chars that can trigger backend encoding issues.
    normalized = "".join(ch for ch in normalized if ch >= " " or ch in "\n\t")
    return normalized.strip()


def embed_chunks(
    chunk_file: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    ollama_base_url: str = VECTOR_STORE.ollama_base_url,
) -> None:
    rows = read_jsonl_chunks(chunk_file)
    if not rows:
        raise ValueError(f"No chunks found in {chunk_file}.")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)

    total = len(rows)
    processed = 0
    print(f"Embedding {total} chunks into collection `{collection_name}` ...")

    for batch in chunked(rows, batch_size):
        texts = [sanitize_text_for_embedding(build_retrieval_text(row)) for row in batch]
        vectors = ollama_embed_batch(ollama_base_url, embedding_model, texts)
        ids = [
            f"{row.get('source', 'unknown')}::{row.get('chunk_id', idx)}"
            for idx, row in enumerate(batch)
        ]
        metadatas = [
            {
                "source": str(row.get("source", "")),
                "chunk_id": int(row.get("chunk_id", -1)),
                "char_count": int(row.get("char_count", len(str(row.get("text", ""))))),
                "heading_path": " > ".join(row.get("heading_path", [])),
            }
            for row in batch
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        processed += len(batch)
        print(f"Progress: {processed}/{total}")

    print(f"Done. Embedded {processed} chunks to `{chroma_dir}`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks to Chroma.")
    parser.add_argument("--chunk-file", type=Path, default=VECTOR_STORE.chunk_file)
    parser.add_argument("--chroma-dir", type=Path, default=VECTOR_STORE.chroma_dir)
    parser.add_argument("--collection", default=VECTOR_STORE.collection_name)
    parser.add_argument("--ollama-url", default=VECTOR_STORE.ollama_base_url)
    parser.add_argument("--embedding-model", default=MODELS.embedding)
    parser.add_argument("--batch-size", type=int, default=VECTOR_STORE.embed_batch_size)
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip `ollama pull` and directly start embedding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    if not args.skip_pull:
        pull_ollama_model(args.embedding_model)

    embed_chunks(
        chunk_file=args.chunk_file,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        ollama_base_url=args.ollama_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
