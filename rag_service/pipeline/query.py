from dataclasses import dataclass
from typing import Any

from rag_service.confidence import confidence_fallback_answer, filter_context_rows, top_rerank_score
from rag_service.config import MODELS, VECTOR_STORE
from rag_service.generation.chat import build_context_blocks, build_generation_prompt, format_references
from rag_service.pipeline.trace import PipelineTrace
from rag_service.retrieval.hybrid import query_top_k
from rag_service.retrieval.query_rewrite import rewrite_query


@dataclass
class QueryPipelineResult:
    rows: list[dict]
    """检索+rerank 后的 top_k 行（置信度过滤前）。"""
    context_rows: list[dict]
    """经 filter_context 后实际进入 prompt 的行（最终命中）。"""
    context_blocks: list[str]
    prompt: str
    references: list[str]
    pipeline: list[dict[str, Any]]
    rewritten_query: str
    fallback_answer: str | None = None


class QueryPipeline:
    def __init__(self) -> None:
        self.trace = PipelineTrace()

    def prepare(
        self,
        *,
        question: str,
        top_k: int,
        dense_k: int,
        bm25_k: int,
        rerank_top_n: int,
        dense_weight: float,
        bm25_weight: float,
        source: str | None,
        max_context_chars: int,
    ) -> QueryPipelineResult:
        rewritten_query = self._rewrite_query(question)
        rows = self._retrieve(
            question=rewritten_query,
            top_k=top_k,
            dense_k=dense_k,
            bm25_k=bm25_k,
            rerank_top_n=rerank_top_n,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            source=source,
        )
        fallback_answer = self._confidence_fallback(rows)
        if fallback_answer is not None:
            return QueryPipelineResult(
                rows=rows,
                context_rows=[],
                context_blocks=[],
                prompt="",
                references=[],
                pipeline=self.trace.as_dicts(),
                rewritten_query=rewritten_query,
                fallback_answer=fallback_answer,
            )

        context_rows = self._filter_context_rows(rows)
        context_blocks = self._build_context(rows=context_rows, max_context_chars=max_context_chars)
        prompt = self._build_prompt(question=question, context_blocks=context_blocks)
        references = self._format_references(rows=context_rows, context_blocks=context_blocks)

        return QueryPipelineResult(
            rows=rows,
            context_rows=context_rows,
            context_blocks=context_blocks,
            prompt=prompt,
            references=references,
            pipeline=self.trace.as_dicts(),
            rewritten_query=rewritten_query,
        )

    def _rewrite_query(self, question: str) -> str:
        rewritten_query = rewrite_query(question)
        self.trace.record("rewrite_query", original=question, rewritten=rewritten_query)
        return rewritten_query

    def _retrieve(
        self,
        *,
        question: str,
        top_k: int,
        dense_k: int,
        bm25_k: int,
        rerank_top_n: int,
        dense_weight: float,
        bm25_weight: float,
        source: str | None,
    ) -> list[dict]:
        rows = query_top_k(
            query=question,
            top_k=top_k,
            dense_k=dense_k,
            bm25_k=bm25_k,
            chroma_dir=VECTOR_STORE.chroma_dir,
            collection_name=VECTOR_STORE.collection_name,
            chunk_file=VECTOR_STORE.chunk_file,
            ollama_url=VECTOR_STORE.ollama_base_url,
            embedding_model=MODELS.embedding,
            reranker_model=MODELS.reranker,
            rerank_top_n=rerank_top_n,
            source=source,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
        self.trace.record("retrieve", retrieved_count=len(rows), top_k=top_k)
        return rows

    def _confidence_fallback(self, rows: list[dict]) -> str | None:
        top_score = top_rerank_score(rows)
        fallback_answer = confidence_fallback_answer(rows)
        self.trace.record(
            "confidence_gate",
            status="fallback" if fallback_answer is not None else "ok",
            top_score=top_score,
        )
        return fallback_answer

    def _filter_context_rows(self, rows: list[dict]) -> list[dict]:
        context_rows = filter_context_rows(rows)
        self.trace.record(
            "filter_context",
            kept_count=len(context_rows),
            dropped_count=len(rows) - len(context_rows),
        )
        return context_rows

    def _build_context(self, *, rows: list[dict], max_context_chars: int) -> list[str]:
        context_blocks = build_context_blocks(rows=rows, max_context_chars=max_context_chars)
        self.trace.record(
            "build_context",
            context_block_count=len(context_blocks),
            max_context_chars=max_context_chars,
        )
        return context_blocks

    def _build_prompt(self, *, question: str, context_blocks: list[str]) -> str:
        prompt = build_generation_prompt(query=question, context_blocks=context_blocks)
        self.trace.record("build_prompt", prompt_chars=len(prompt))
        return prompt

    def _format_references(self, *, rows: list[dict], context_blocks: list[str]) -> list[str]:
        references = format_references(rows[: len(context_blocks)])
        self.trace.record("format_references", reference_count=len(references))
        return references
