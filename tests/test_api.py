import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_service.api import app as app_module


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_docs_upload(self):
        with patch.object(
            app_module,
            "_run_ingest_pipeline",
            return_value={
                "converted_count": 1,
                "chunk_count": 2,
                "pipeline": [{"name": "convert_documents", "status": "ok", "details": {"converted_count": 1}}],
            },
        ):
            files = [("files", ("rules.md", io.BytesIO(b"# title\nrule"), "text/markdown"))]
            resp = self.client.post("/docs", files=files)
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["converted_count"], 1)
            self.assertIn("saved_files", body)
            self.assertEqual(body["pipeline"][0]["name"], "convert_documents")

    def test_docs_upload_text_only(self):
        with patch.object(
            app_module,
            "_run_ingest_pipeline",
            return_value={
                "converted_count": 1,
                "chunk_count": 1,
                "pipeline": [{"name": "convert_documents", "status": "ok", "details": {"converted_count": 1}}],
            },
        ):
            resp = self.client.post(
                "/docs",
                data={"text": "主播提现前需要实名认证", "source_name": "inline_rules.md"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["converted_count"], 1)
            self.assertIn("inline_rules.md", body["saved_files"][0])

    def test_chat_route_returns_non_stream_answer(self):
        with (
            patch.object(
                app_module.QueryPipeline,
                "prepare",
                return_value=type(
                    "Prepared",
                    (),
                    {
                        "pipeline": [{"name": "retrieve", "status": "ok", "details": {}}],
                        "prompt": "prompt",
                        "references": ["[1] ref"],
                        "rewritten_query": "rewritten",
                        "context_rows": [{"chunk_id": 1, "source": "a.md", "rerank_score": 0.9, "text": "ctx"}],
                        "rows": [{"chunk_id": 1, "source": "a.md", "rerank_score": 0.9, "text": "ctx"}],
                        "fallback_answer": None,
                    },
                )(),
            ) as prepare,
            patch.object(app_module, "call_answer_generate", return_value="答案") as answer_generate,
            patch.object(app_module, "write_record"),
        ):
            resp = self.client.post("/chat", json={"question": "hi"})

        self.assertEqual(resp.status_code, 200)
        prepare.assert_called_once()
        answer_generate.assert_called_once_with(prompt="prompt")
        body = resp.json()
        self.assertEqual(body["answer"], "答案")
        self.assertEqual(body["references"], ["[1] ref"])
        self.assertFalse(body["fallback"])

    def test_chat_stream(self):
        with (
            patch.object(app_module.QueryPipeline, "prepare",
                return_value=type(
                    "Prepared",
                    (),
                    {
                        "pipeline": [
                            {"name": "rewrite_query", "status": "ok", "details": {}},
                            {"name": "retrieve", "status": "ok", "details": {}},
                        ],
                        "prompt": "prompt",
                        "references": ["[1] source=a.md"],
                        "rewritten_query": "rewritten",
                        "context_rows": [{"chunk_id": 1, "source": "a.md", "rerank_score": 0.9, "text": "ctx"}],
                        "rows": [{"chunk_id": 1, "source": "a.md", "rerank_score": 0.9, "text": "ctx"}],
                        "fallback_answer": None,
                    },
                )(),
            ) as prepare,
            patch.object(app_module, "stream_answer_generate", return_value=iter(["a", "b"])) as stream_generate,
            patch.object(app_module, "call_answer_generate") as answer_generate,
            patch.object(app_module, "write_record"),
        ):
            resp = self.client.post("/chat/stream", json={"question": "hi"})
            self.assertEqual(resp.status_code, 200)
            prepare.assert_called_once()
            stream_generate.assert_called_once_with(prompt="prompt")
            answer_generate.assert_not_called()
            self.assertEqual(prepare.call_args.kwargs["top_k"], 5)
            self.assertEqual(prepare.call_args.kwargs["rerank_top_n"], 10)
            text = resp.text
            self.assertIn('"type": "pipeline_step"', text)
            self.assertIn('"name": "rewrite_query"', text)
            self.assertIn('"name": "retrieve"', text)
            self.assertIn('"type": "token"', text)
            self.assertIn('"type": "references"', text)
            payloads = [line.replace("data: ", "") for line in text.splitlines() if line.startswith("data: ")]
            token_text = "".join(json.loads(p).get("content", "") for p in payloads if "content" in json.loads(p))
            self.assertEqual(token_text, "ab")

    def test_chat_stream_returns_fixed_text_when_rerank_confidence_is_low(self):
        prepared = type(
            "Prepared",
            (),
            {
                "pipeline": [{"name": "confidence_gate", "status": "fallback", "details": {"top_score": 0.49}}],
                "prompt": "",
                "references": [],
                "fallback_answer": "无确信的依据，请问点别的问题吧~",
                "rewritten_query": "rewritten",
                "context_rows": [],
            },
        )()

        with (
            patch.object(app_module.QueryPipeline, "prepare", return_value=prepared),
            patch.object(app_module, "stream_answer_generate") as stream_generate,
            patch.object(app_module, "call_answer_generate") as answer_generate,
            patch.object(app_module, "write_record"),
        ):
            resp = self.client.post("/chat/stream", json={"question": "hi"})

        self.assertEqual(resp.status_code, 200)
        stream_generate.assert_not_called()
        answer_generate.assert_not_called()
        payloads = [json.loads(line.replace("data: ", "")) for line in resp.text.splitlines() if line.startswith("data: ")]
        token_text = "".join(p.get("content", "") for p in payloads if p.get("type") == "token")
        self.assertIn("无确信的依据，请问点别的问题吧~", token_text)
        self.assertEqual(payloads[-1], {"type": "references", "references": []})

    def test_docs_status_reports_chunk_sources_and_index_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunk_file = Path(tmp) / "chunks.jsonl"
            chroma_dir = Path(tmp) / "chroma_db"
            chroma_dir.mkdir()
            chunk_file.write_text(
                '\n'.join(
                    [
                        json.dumps({"source": "a.md", "chunk_id": 1}, ensure_ascii=False),
                        json.dumps({"source": "b.md", "chunk_id": 2}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.object(app_module.VECTOR_STORE, "chunk_file", chunk_file),
                patch.object(app_module.VECTOR_STORE, "chroma_dir", chroma_dir),
                patch.object(app_module.VECTOR_STORE, "collection_name", "rag_chunks"),
            ):
                resp = self.client.get("/docs/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["chunk_count"], 2)
        self.assertEqual(body["sources"], ["a.md", "b.md"])
        self.assertTrue(body["index_exists"])
        self.assertEqual(body["collection_name"], "rag_chunks")

    def test_chat_debug_returns_retrieval_diagnostics(self):
        debug_payload = {
            "question": "hi",
            "rewritten_query": "rewritten",
            "pipeline": [{"name": "rewrite_query", "status": "ok", "details": {}}],
            "dense_rows": [],
            "bm25_rows": [],
            "rerank_input_rows": [],
            "rerank_raw_results": [],
            "reranked_rows": [],
            "context_rows": [],
            "fallback_answer": None,
        }

        with patch.object(app_module, "_run_chat_debug", return_value=debug_payload) as run_debug:
            resp = self.client.post("/chat/debug", json={"question": "hi", "top_k": 3})

        self.assertEqual(resp.status_code, 200)
        run_debug.assert_called_once()
        self.assertEqual(resp.json()["rewritten_query"], "rewritten")
        self.assertIn("rerank_raw_results", resp.json())

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_root_serves_frontend_index(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("<!doctype html>", resp.text.lower())


if __name__ == "__main__":
    unittest.main()
