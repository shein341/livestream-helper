import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rag_service.ingestion.embedder import embed_chunks, ollama_embed_batch


class EmbedderTests(unittest.TestCase):
    def test_ollama_embed_batch_uses_local_embed_endpoint(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

        with patch("rag_service.ingestion.embedder.requests.post", return_value=resp) as post:
            vectors = ollama_embed_batch(
                base_url="http://127.0.0.1:11434",
                model_name="bge-m3:latest",
                texts=["a", "b"],
            )

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/embed")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "bge-m3:latest")
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["a", "b"])

    def test_ollama_embed_batch_fails_fast_on_http_error(self):
        resp = Mock()
        resp.status_code = 500
        resp.text = "boom"

        with patch("rag_service.ingestion.embedder.requests.post", return_value=resp) as post:
            with self.assertRaises(RuntimeError):
                ollama_embed_batch(
                    base_url="http://127.0.0.1:11434",
                    model_name="bge-m3:latest",
                    texts=["a"],
                )

        post.assert_called_once()


    def test_embed_chunks_sends_heading_path_with_text_to_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunk_file = Path(tmp) / "chunks.jsonl"
            chunk_file.write_text(
                '{"source":"rules.md","chunk_id":0,"char_count":12,'
                '"heading_path":["提现规则","前置条件"],"text":"主播需实名认证。"}\n',
                encoding="utf-8",
            )

            fake_collection = Mock()
            fake_client = Mock()
            fake_client.get_or_create_collection.return_value = fake_collection

            with (
                patch("rag_service.ingestion.embedder.chromadb.PersistentClient", return_value=fake_client),
                patch(
                    "rag_service.ingestion.embedder.ollama_embed_batch",
                    return_value=[[0.1, 0.2]],
                ) as embed,
            ):
                embed_chunks(
                    chunk_file=chunk_file,
                    chroma_dir=Path(tmp) / "chroma",
                    collection_name="chunks",
                    embedding_model="bge-m3:latest",
                    batch_size=32,
                    ollama_base_url="http://ollama",
                )

        texts = embed.call_args.args[2]
        self.assertEqual(len(texts), 1)
        self.assertIn("提现规则 > 前置条件", texts[0])
        self.assertIn("主播需实名认证。", texts[0])


if __name__ == "__main__":
    unittest.main()
