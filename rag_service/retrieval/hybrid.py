import argparse
import json
import math
import re
from pathlib import Path

import chromadb
import requests

from rag_service.config import MODELS, VECTOR_STORE
from rag_service.ingestion.embedder import build_retrieval_text

_FLAG_RERANKERS = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval (dense + BM25) with mandatory reranker."
    )
    parser.add_argument("query", help="User query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--dense-k", type=int, default=30, help="Dense recall depth.")
    parser.add_argument("--bm25-k", type=int, default=30, help="BM25 recall depth.")
    parser.add_argument("--chroma-dir", type=Path, default=VECTOR_STORE.chroma_dir)
    parser.add_argument("--collection", default=VECTOR_STORE.collection_name)
    parser.add_argument("--chunk-file", type=Path, default=VECTOR_STORE.chunk_file)
    parser.add_argument("--ollama-url", default=VECTOR_STORE.ollama_base_url)
    parser.add_argument("--embedding-model", default=MODELS.embedding)
    parser.add_argument("--reranker-model", default=MODELS.reranker)
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=10,
        help="How many fused coarse-ranked candidates go into reranker.",
    )
    parser.add_argument("--dense-weight", type=float, default=1.0, help="RRF weight for dense ranking.")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="RRF weight for BM25 ranking.")
    parser.add_argument("--source", default=None, help="Optional source file filter, e.g. a.md.")
    parser.add_argument("--json", action="store_true", help="Output full rows as JSON.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def ollama_embed_query(base_url: str, model_name: str, query: str) -> list[float]:
    if not query.strip():
        raise ValueError("query must not be empty")
    query = query.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    embed_resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": model_name, "input": [query]},
        timeout=120,
    )
    if embed_resp.status_code != 200:
        raise RuntimeError(f"Embedding API failed ({embed_resp.status_code}): {embed_resp.text}")
    data = embed_resp.json()
    embeddings = data.get("embeddings")
    if not (isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list)):
        raise RuntimeError("Unexpected response schema from /api/embed")
    vec = embeddings[0]
    return vec


def tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", text)
    han_chars = re.findall(r"[\u4e00-\u9fff]", text)
    han_bigrams = [han_chars[i] + han_chars[i + 1] for i in range(len(han_chars) - 1)]
    return latin_tokens + han_chars + han_bigrams


def bm25_retrieve(
    query: str,
    rows: list[dict],
    top_k: int,
    source: str | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    filtered = [row for row in rows if (not source or str(row.get("source", "")) == source)]
    if not filtered:
        return []

    tokenized_docs = [tokenize_for_bm25(build_retrieval_text(row)) for row in filtered]
    doc_lens = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

    df: dict[str, int] = {}
    for tokens in tokenized_docs:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    q_tokens = tokenize_for_bm25(query)
    if not q_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    n_docs = len(filtered)
    for idx, doc in enumerate(filtered):
        tokens = tokenized_docs[idx]
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        score = 0.0
        dl = len(tokens)
        for tok in q_tokens:
            tf_td = tf.get(tok, 0)
            if tf_td == 0:
                continue
            df_tok = df.get(tok, 0)
            idf = math.log((n_docs - df_tok + 0.5) / (df_tok + 0.5) + 1.0)
            denom = tf_td + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 0.0))
            score += idf * ((tf_td * (k1 + 1.0)) / denom)

        if score > 0.0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for rank, (score, doc) in enumerate(scored[:top_k], start=1):
        row = dict(doc)
        row["rank"] = rank
        row["bm25_score"] = score
        out.append(row)
    return out


def build_ranked_rows(query_result: dict) -> list[dict]:
    ids_batch = query_result.get("ids")
    distances_batch = query_result.get("distances")
    docs_batch = query_result.get("documents")
    metadatas_batch = query_result.get("metadatas")
    if not all(isinstance(x, list) and x for x in (ids_batch, distances_batch, docs_batch, metadatas_batch)):
        raise RuntimeError("Invalid Chroma query response: expected non-empty batched fields.")

    ids = ids_batch[0]
    distances = distances_batch[0]
    docs = docs_batch[0]
    metadatas = metadatas_batch[0]
    if not (len(ids) == len(distances) == len(docs) == len(metadatas)):
        raise RuntimeError("Invalid Chroma query response: field lengths are inconsistent.")

    rows: list[dict] = []
    for idx, chunk_id in enumerate(ids):
        metadata = metadatas[idx]
        if not isinstance(metadata, dict):
            raise RuntimeError("Invalid Chroma query response: metadata item must be dict.")
        rows.append(
            {
                "rank": idx + 1,
                "id": chunk_id,
                "distance": distances[idx],
                "source": metadata.get("source", ""),
                "chunk_id": metadata.get("chunk_id"),
                "char_count": metadata.get("char_count"),
                "heading_path": metadata.get("heading_path", ""),
                "text": docs[idx],
            }
        )
    return rows


def dense_retrieve(
    query: str,
    top_k: int,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    source: str | None = None,
    ollama_url: str = VECTOR_STORE.ollama_base_url,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top-k must be positive")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=collection_name)
    query_embedding = ollama_embed_query(ollama_url, embedding_model, query)

    where = {"source": source} if source else None
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    return build_ranked_rows(query_result)


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
            chunk_id = f"{row.get('source', 'unknown')}::{row.get('chunk_id', line_no)}"
            rows.append(
                {
                    "id": chunk_id,
                    "source": row.get("source", ""),
                    "chunk_id": row.get("chunk_id"),
                    "char_count": row.get("char_count"),
                    "heading_path": " > ".join(row.get("heading_path", [])),
                    "text": build_retrieval_text(row),
                }
            )
    return rows


def fuse_rrf(
    dense_rows: list[dict],
    bm25_rows: list[dict],
    top_k: int,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[dict]:
    fused: dict[str, dict] = {}

    for rank, row in enumerate(dense_rows, start=1):
        row_id = str(row.get("id"))
        item = fused.setdefault(row_id, dict(row))
        item["dense_rank"] = rank
        item["rrf_score"] = item.get("rrf_score", 0.0) + dense_weight * (1.0 / (rrf_k + rank))

    for rank, row in enumerate(bm25_rows, start=1):
        row_id = str(row.get("id"))
        item = fused.setdefault(row_id, dict(row))
        item["bm25_rank"] = rank
        item["rrf_score"] = item.get("rrf_score", 0.0) + bm25_weight * (1.0 / (rrf_k + rank))

    ranked = sorted(fused.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked[:top_k]


def call_rerank_model(
    *,
    model_name: str,
    query: str,
    docs: list[str],
    top_n: int,
) -> list[dict]:
    if not docs:
        return []

    pairs = [[query, doc] for doc in docs]
    raw_scores = get_flag_reranker(model_name).compute_score(pairs, normalize=True)
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, (int, float)):
        raw_scores = [float(raw_scores)]
    if not isinstance(raw_scores, list) or len(raw_scores) != len(docs):
        raise RuntimeError("Unexpected FlagEmbedding rerank score payload.")

    scored = [{"index": idx, "relevance_score": float(score)} for idx, score in enumerate(raw_scores)]
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    return scored[: min(top_n, len(scored))]


def get_flag_reranker(model_name: str):
    reranker = _FLAG_RERANKERS.get(model_name)
    if reranker is None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError("FlagEmbedding is required for rerank. Install it with `pip install FlagEmbedding`.") from exc
        reranker = FlagReranker(model_name, use_fp16=False)
        _FLAG_RERANKERS[model_name] = reranker
    return reranker


def rerank_candidates(
    *,
    rows: list[dict],
    query: str,
    top_k: int,
    rerank_top_n: int,
    model_name: str,
) -> list[dict]:
    rerank_input = rows[: min(rerank_top_n, len(rows))]
    rerank_results = call_rerank_model(
        model_name=model_name,
        query=query,
        docs=[str(row.get("text", "")) for row in rerank_input],
        top_n=min(top_k, len(rerank_input)),
    )
    return apply_rerank_scores(rerank_input, rerank_results)[:top_k]


def apply_rerank_scores(rows: list[dict], rerank_results: list[dict]) -> list[dict]:
    reranked: list[dict] = []
    used_indices: set[int] = set()
    for item in rerank_results:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(rows):
            raise RuntimeError("Invalid rerank response: index out of bounds.")
        if idx in used_indices:
            raise RuntimeError("Invalid rerank response: duplicate index.")
        used_indices.add(idx)
        row = dict(rows[idx])
        row["rerank_score"] = item.get("relevance_score")
        reranked.append(row)

    reranked.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
    for idx, row in enumerate(reranked, start=1):
        row["rank"] = idx
    return reranked


def query_top_k(
    query: str,
    top_k: int,
    dense_k: int,
    bm25_k: int,
    chroma_dir: Path,
    collection_name: str,
    chunk_file: Path,
    ollama_url: str,
    embedding_model: str,
    reranker_model: str,
    rerank_top_n: int,
    source: str | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    if rerank_top_n <= 0:
        raise ValueError("rerank-top-n must be positive")

    dense_rows = dense_retrieve(
        query=query,
        top_k=dense_k,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        source=source,
        ollama_url=ollama_url,
    )
    chunk_rows = read_jsonl_chunks(chunk_file)
    bm25_rows = bm25_retrieve(query=query, rows=chunk_rows, top_k=bm25_k, source=source)

    fused_rows = fuse_rrf(
        dense_rows=dense_rows,
        bm25_rows=bm25_rows,
        top_k=max(top_k, rerank_top_n),
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    if not fused_rows:
        return []

    return rerank_candidates(
        rows=fused_rows,
        query=query,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        model_name=reranker_model,
    )


def query_debug(
    query: str,
    top_k: int,
    dense_k: int,
    bm25_k: int,
    chroma_dir: Path,
    collection_name: str,
    chunk_file: Path,
    ollama_url: str,
    embedding_model: str,
    reranker_model: str,
    rerank_top_n: int,
    source: str | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> dict:
    if rerank_top_n <= 0:
        raise ValueError("rerank-top-n must be positive")

    dense_rows = dense_retrieve(
        query=query,
        top_k=dense_k,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        source=source,
        ollama_url=ollama_url,
    )
    chunk_rows = read_jsonl_chunks(chunk_file)
    bm25_rows = bm25_retrieve(query=query, rows=chunk_rows, top_k=bm25_k, source=source)
    fused_rows = fuse_rrf(
        dense_rows=dense_rows,
        bm25_rows=bm25_rows,
        top_k=max(top_k, rerank_top_n),
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    rerank_input = fused_rows[: min(rerank_top_n, len(fused_rows))]
    rerank_results = call_rerank_model(
        model_name=reranker_model,
        query=query,
        docs=[str(row.get("text", "")) for row in rerank_input],
        top_n=min(top_k, len(rerank_input)),
    )
    reranked_rows = apply_rerank_scores(rerank_input, rerank_results)[:top_k]
    return {
        "dense_rows": dense_rows,
        "bm25_rows": bm25_rows,
        "rerank_input_rows": rerank_input,
        "rerank_raw_results": rerank_results,
        "reranked_rows": reranked_rows,
    }


def print_plain(rows: list[dict]) -> None:
    if not rows:
        print("No result.")
        return

    for row in rows:
        print(f"[{row['rank']}] id={row['id']} distance={row.get('distance')}")
        print(f"source={row['source']} heading={row['heading_path']}")
        print(row["text"])
        print("")


def main() -> int:
    args = parse_args()
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

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_plain(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
