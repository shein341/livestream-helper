import unittest
from unittest.mock import patch

from rag_service.pipeline.query import QueryPipeline


class QueryPipelineTests(unittest.TestCase):
    def test_prepare_sets_fallback_answer_when_rerank_confidence_is_low(self):
        rows = [{"text": "irrelevant", "rerank_score": 0.49}]

        with (
            patch("rag_service.pipeline.query.rewrite_query", return_value="rewritten"),
            patch("rag_service.pipeline.query.query_top_k", return_value=rows),
        ):
            result = QueryPipeline().prepare(
                question="question",
                top_k=5,
                dense_k=30,
                bm25_k=30,
                rerank_top_n=20,
                dense_weight=1.0,
                bm25_weight=1.0,
                source=None,
                max_context_chars=6000,
            )

        self.assertEqual(result.fallback_answer, "无确信的依据，请问点别的问题吧~")
        self.assertEqual(result.prompt, "")
        self.assertEqual(result.references, [])
        self.assertEqual(result.context_rows, [])
        self.assertTrue(any(step["name"] == "confidence_gate" for step in result.pipeline))

    def test_prepare_builds_prompt_at_confidence_threshold(self):
        rows = [{"source": "a.md", "heading_path": "", "chunk_id": 1, "text": "ctx", "rerank_score": 0.5}]

        with (
            patch("rag_service.pipeline.query.rewrite_query", return_value="rewritten"),
            patch("rag_service.pipeline.query.query_top_k", return_value=rows),
        ):
            result = QueryPipeline().prepare(
                question="question",
                top_k=5,
                dense_k=30,
                bm25_k=30,
                rerank_top_n=20,
                dense_weight=1.0,
                bm25_weight=1.0,
                source=None,
                max_context_chars=6000,
            )

        self.assertIsNone(result.fallback_answer)
        self.assertIn("question", result.prompt)
        self.assertEqual(len(result.references), 1)
        self.assertEqual(len(result.context_rows), 1)
        self.assertEqual(result.context_rows[0]["chunk_id"], 1)

    def test_prepare_excludes_low_rerank_scores_from_prompt_context(self):
        rows = [
            {"source": "a.md", "heading_path": "", "chunk_id": 1, "text": "high confidence", "rerank_score": 0.8},
            {"source": "b.md", "heading_path": "", "chunk_id": 2, "text": "low confidence", "rerank_score": 0.09},
        ]

        with (
            patch("rag_service.pipeline.query.rewrite_query", return_value="rewritten"),
            patch("rag_service.pipeline.query.query_top_k", return_value=rows),
        ):
            result = QueryPipeline().prepare(
                question="question",
                top_k=5,
                dense_k=30,
                bm25_k=30,
                rerank_top_n=20,
                dense_weight=1.0,
                bm25_weight=1.0,
                source=None,
                max_context_chars=6000,
            )

        self.assertIn("high confidence", result.prompt)
        self.assertNotIn("low confidence", result.prompt)
        self.assertEqual(result.references, ["[1] high confidence"])
        self.assertEqual(len(result.context_blocks), 1)
        self.assertEqual(len(result.context_rows), 1)
        self.assertEqual(result.context_rows[0]["chunk_id"], 1)


if __name__ == "__main__":
    unittest.main()
