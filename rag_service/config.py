from dataclasses import dataclass, field
import os
from pathlib import Path


def _load_dotenv() -> None:
    """从项目根目录的 .env 加载环境变量（零依赖、不覆盖已有变量）。"""
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ModelConfig:
    embedding: str = field(default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "bge-m3:latest"))
    reranker: str = field(default_factory=lambda: _env("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))


@dataclass
class AnswerConfig:
    base_url: str = field(default_factory=lambda: _env("RAG_ANSWER_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: _env("RAG_ANSWER_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("RAG_ANSWER_MODEL", ""))
    reasoning_split: bool = field(default_factory=lambda: _env_bool("RAG_ANSWER_REASONING_SPLIT", True))
    thinking: str = field(default_factory=lambda: _env("RAG_ANSWER_THINKING", ""))


@dataclass
class QueryRewriteConfig:
    provider: str = field(default_factory=lambda: _env("RAG_QUERY_REWRITE_PROVIDER", "ollama"))
    base_url: str = field(default_factory=lambda: _env("RAG_QUERY_REWRITE_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: _env("RAG_QUERY_REWRITE_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("RAG_QUERY_REWRITE_MODEL", "qwen3.5:4b"))
    thinking: str = field(default_factory=lambda: _env("RAG_QUERY_REWRITE_THINKING", ""))


@dataclass
class VectorStoreConfig:
    chunk_file: Path = Path("rag_chunks_realistic.jsonl")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "rag_chunks"
    ollama_base_url: str = field(default_factory=lambda: _env("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    embed_batch_size: int = 32


@dataclass
class DataConfig:
    raw_dir: Path = Path("raw_docs")
    processed_dir: Path = Path("processed_md")
    chunk_target_size: int = 400
    chunk_max_size: int = 700
    chunk_overlap: int = 15


MODELS = ModelConfig()
ANSWER = AnswerConfig()
QUERY_REWRITE = QueryRewriteConfig()
VECTOR_STORE = VectorStoreConfig()
DATA = DataConfig()
