from typing import Any

from rag_service.config import DATA, MODELS, VECTOR_STORE
from rag_service.ingestion.chunker import MarkdownChunker
from rag_service.ingestion.converter import DocumentConverter
from rag_service.ingestion.embedder import embed_chunks, pull_ollama_model
from rag_service.pipeline.trace import PipelineTrace


class IngestionPipeline:
    def __init__(self) -> None:
        self.trace = PipelineTrace()

    def run(self) -> dict[str, Any]:
        converted = self._convert_documents()
        rows = self._chunk_documents()
        self._pull_embedding_model()
        self._embed_chunks()

        return {
            "converted_count": len(converted),
            "chunk_count": len(rows),
            "pipeline": self.trace.as_dicts(),
        }

    def _convert_documents(self) -> list:
        converter = DocumentConverter()
        converted = converter.convert(
            input_path=DATA.raw_dir,
            output_dir=DATA.processed_dir,
            recursive=True,
            overwrite=True,
        )
        self.trace.record("convert_documents", converted_count=len(converted))
        return converted

    def _chunk_documents(self) -> list[dict]:
        chunker = MarkdownChunker(
            target_size=DATA.chunk_target_size,
            max_size=DATA.chunk_max_size,
            overlap=DATA.chunk_overlap,
        )
        rows = chunker.chunk_directory(
            input_dir=DATA.processed_dir,
            output_file=VECTOR_STORE.chunk_file,
            recursive=True,
        )
        self.trace.record("chunk_documents", chunk_count=len(rows), output_file=str(VECTOR_STORE.chunk_file))
        return rows

    def _pull_embedding_model(self) -> None:
        pull_ollama_model(MODELS.embedding)
        self.trace.record("pull_embedding_model", model=MODELS.embedding)

    def _embed_chunks(self) -> None:
        embed_chunks(
            chunk_file=VECTOR_STORE.chunk_file,
            chroma_dir=VECTOR_STORE.chroma_dir,
            collection_name=VECTOR_STORE.collection_name,
            embedding_model=MODELS.embedding,
            batch_size=VECTOR_STORE.embed_batch_size,
        )
        self.trace.record(
            "embed_chunks",
            chroma_dir=str(VECTOR_STORE.chroma_dir),
            collection_name=VECTOR_STORE.collection_name,
        )
