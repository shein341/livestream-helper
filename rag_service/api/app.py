import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_service.chat_log import new_request_id, write_record
from rag_service.confidence import confidence_fallback_answer, filter_context_rows, top_rerank_score
from rag_service.config import DATA, MODELS, VECTOR_STORE
from rag_service.generation.chat import call_answer_generate, stream_answer_generate
from rag_service.ingestion.converter import SUPPORTED_SUFFIXES
from rag_service.pipeline import IngestionPipeline, QueryPipeline
from rag_service.pipeline.trace import PipelineTrace
from rag_service.retrieval.hybrid import query_debug
from rag_service.retrieval.query_rewrite import rewrite_query

app = FastAPI(title="RAG Backend", docs_url="/swagger", redoc_url=None)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

_cors_origins = os.environ.get(
    "RAG_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)
    dense_k: int = Field(default=30, gt=0, le=200)
    bm25_k: int = Field(default=30, gt=0, le=200)
    rerank_top_n: int = Field(default=10, gt=0, le=200)
    dense_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    bm25_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    source: str | None = None
    max_context_chars: int = Field(default=6000, gt=0, le=50000)


def _safe_name(name: str) -> str:
    return Path(name).name


def _run_ingest_pipeline() -> dict:
    return IngestionPipeline().run()


def _run_query_pipeline(req: ChatRequest):
    return QueryPipeline().prepare(
        question=req.question,
        top_k=req.top_k,
        dense_k=req.dense_k,
        bm25_k=req.bm25_k,
        rerank_top_n=req.rerank_top_n,
        dense_weight=req.dense_weight,
        bm25_weight=req.bm25_weight,
        source=req.source,
        max_context_chars=req.max_context_chars,
    )


def _build_chat_answer(result) -> tuple[str, list[str], bool]:
    fallback_answer = getattr(result, "fallback_answer", None)
    is_fallback = fallback_answer is not None
    if is_fallback:
        return str(fallback_answer), [], True
    return call_answer_generate(prompt=result.prompt), result.references, False


def _write_chat_record(
    *,
    request_id: str,
    req: ChatRequest,
    result,
    answer: str,
    fallback: bool,
    started_at: float,
) -> None:
    write_record(
        request_id=request_id,
        question=req.question,
        rewritten_query=getattr(result, "rewritten_query", ""),
        retrieved_rows=getattr(result, "rows", []) or [],
        context_rows=getattr(result, "context_rows", []) or [],
        prompt=getattr(result, "prompt", "") or "",
        answer=answer,
        fallback=fallback,
        started_at=started_at,
    )


def _docs_status() -> dict:
    chunk_file = VECTOR_STORE.chunk_file
    sources: set[str] = set()
    chunk_count = 0
    if chunk_file.exists():
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                chunk_count += 1
                source = row.get("source")
                if isinstance(source, str) and source:
                    sources.add(source)

    return {
        "chunk_file": str(chunk_file),
        "chunk_count": chunk_count,
        "sources": sorted(sources),
        "index_exists": VECTOR_STORE.chroma_dir.exists(),
        "chroma_dir": str(VECTOR_STORE.chroma_dir),
        "collection_name": VECTOR_STORE.collection_name,
    }


def _run_chat_debug(req: ChatRequest) -> dict:
    trace = PipelineTrace()
    rewritten_query = rewrite_query(req.question)
    trace.record("rewrite_query", original=req.question, rewritten=rewritten_query)
    diagnostics = query_debug(
        query=rewritten_query,
        top_k=req.top_k,
        dense_k=req.dense_k,
        bm25_k=req.bm25_k,
        chroma_dir=VECTOR_STORE.chroma_dir,
        collection_name=VECTOR_STORE.collection_name,
        chunk_file=VECTOR_STORE.chunk_file,
        ollama_url=VECTOR_STORE.ollama_base_url,
        embedding_model=MODELS.embedding,
        reranker_model=MODELS.reranker,
        rerank_top_n=req.rerank_top_n,
        source=req.source,
        dense_weight=req.dense_weight,
        bm25_weight=req.bm25_weight,
    )
    rows = diagnostics["reranked_rows"]
    trace.record("retrieve", retrieved_count=len(rows), top_k=req.top_k)
    fallback_answer = confidence_fallback_answer(rows)
    trace.record(
        "confidence_gate",
        status="fallback" if fallback_answer is not None else "ok",
        top_score=top_rerank_score(rows),
    )
    context_rows = [] if fallback_answer is not None else filter_context_rows(rows)
    trace.record("filter_context", kept_count=len(context_rows), dropped_count=len(rows) - len(context_rows))
    return {
        "question": req.question,
        "rewritten_query": rewritten_query,
        "pipeline": trace.as_dicts(),
        **diagnostics,
        "context_rows": context_rows,
        "fallback_answer": fallback_answer,
    }


@app.post("/docs")
async def upload_docs(
    files: list[UploadFile] | None = File(default=None),
    text: str | None = Form(default=None),
    source_name: str | None = Form(default=None),
) -> dict:
    has_text = bool(text and text.strip())
    has_files = bool(files)
    if not has_text and not has_files:
        raise HTTPException(status_code=400, detail="Provide `files` or non-empty `text`.")

    DATA.raw_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if has_text:
        raw_name = source_name or f"inline_{int(time.time())}.md"
        safe_name = _safe_name(raw_name.strip()) or f"inline_{int(time.time())}.md"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".md", ".txt"}:
            safe_name = f"{Path(safe_name).stem}.md"
        dst = DATA.raw_dir / safe_name
        dst.write_text((text or "").strip() + "\n", encoding="utf-8")
        saved.append(str(dst))

    for f in files or []:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported suffix for `{f.filename}`. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
            )
        safe_name = _safe_name(f.filename or "upload.bin")
        dst = DATA.raw_dir / safe_name
        content = await f.read()
        dst.write_bytes(content)
        saved.append(str(dst))

    result = _run_ingest_pipeline()
    return {"saved_files": saved, **result}


@app.get("/docs/status")
def docs_status() -> dict:
    return _docs_status()


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    started_at = time.time()
    request_id = new_request_id()
    result = _run_query_pipeline(req)
    answer, references, is_fallback = _build_chat_answer(result)
    _write_chat_record(
        request_id=request_id,
        req=req,
        result=result,
        answer=answer,
        fallback=is_fallback,
        started_at=started_at,
    )
    return {
        "answer": answer,
        "references": references,
        "fallback": is_fallback,
        "rewritten_query": getattr(result, "rewritten_query", ""),
        "pipeline": getattr(result, "pipeline", []),
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    started_at = time.time()
    request_id = new_request_id()

    def event_gen():
        answer_buf: list[str] = []
        result = None
        is_fallback = False
        try:
            yield f"data: {json.dumps({'type': 'pipeline_step', 'name': 'prepare', 'status': 'running', 'details': {}}, ensure_ascii=False)}\n\n"

            result = _run_query_pipeline(req)
            fallback_answer = getattr(result, "fallback_answer", None)
            is_fallback = fallback_answer is not None
            references = [] if is_fallback else result.references

            for step in result.pipeline:
                payload = {"type": "pipeline_step", **step}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            token_iter = [str(fallback_answer)] if is_fallback else stream_answer_generate(prompt=result.prompt)
            for token in token_iter:
                answer_buf.append(token)
                payload = {"type": "token", "content": token}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            end_payload = {"type": "references", "references": references}
            yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\n\n"
        finally:
            if result is None:
                return
            try:
                _write_chat_record(
                    request_id=request_id,
                    req=req,
                    result=result,
                    answer="".join(answer_buf),
                    fallback=is_fallback,
                    started_at=started_at,
                )
            except Exception as e:
                print(f"[chat_log] write failed: {e}", flush=True)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/debug")
def chat_debug(req: ChatRequest) -> dict:
    return _run_chat_debug(req)
