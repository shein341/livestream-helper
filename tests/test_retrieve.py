import unittest
from unittest.mock import Mock, patch

from rag_service.retrieval.hybrid import (
    apply_rerank_scores,
    bm25_retrieve,
    build_parser,
    build_ranked_rows,
    call_rerank_model,
    fuse_rrf,
    query_debug,
    rerank_candidates,
)


class RetrieveTests(unittest.TestCase):
    def test_build_parser_basic_args(self):
        parser = build_parser()
        args = parser.parse_args(["提现规则", "--top-k", "3"])
        self.assertEqual(args.query, "提现规则")
        self.assertEqual(args.top_k, 3)
        self.assertFalse(args.json)

    def test_build_parser_defaults_coarse_top10_and_final_top5(self):
        parser = build_parser()
        args = parser.parse_args(["提现规则"])
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.rerank_top_n, 10)

    def test_build_ranked_rows(self):
        query_result = {
            "ids": [["doc-a::1", "doc-b::2"]],
            "distances": [[0.11, 0.33]],
            "documents": [["第一条 提现说明", "第二条 结算说明"]],
            "metadatas": [[{"source": "a.md"}, {"source": "b.md", "heading_path": "章一 > 条二"}]],
        }
        rows = build_ranked_rows(query_result)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["id"], "doc-a::1")
        self.assertEqual(rows[0]["source"], "a.md")
        self.assertEqual(rows[1]["heading_path"], "章一 > 条二")

    def test_bm25_retrieve_prefers_term_match(self):
        rows = [
            {"id": "a::1", "source": "a.md", "text": "提现审核规则 T+1 到账"},
            {"id": "b::2", "source": "b.md", "text": "直播互动礼仪规范"},
        ]
        ranked = bm25_retrieve(query="提现 到账", rows=rows, top_k=2)
        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["id"], "a::1")

    def test_fuse_rrf_combines_two_rankings(self):
        dense_rows = [
            {"id": "a::1", "text": "A"},
            {"id": "b::2", "text": "B"},
        ]
        bm25_rows = [
            {"id": "b::2", "text": "B"},
            {"id": "c::3", "text": "C"},
        ]
        fused = fuse_rrf(dense_rows=dense_rows, bm25_rows=bm25_rows, top_k=3)
        self.assertEqual(len(fused), 3)
        ids = [x["id"] for x in fused]
        self.assertIn("b::2", ids)

    def test_apply_rerank_scores(self):
        rows = [
            {"id": "a::1", "text": "A"},
            {"id": "b::2", "text": "B"},
        ]
        rerank_results = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
        reranked = apply_rerank_scores(rows, rerank_results)
        self.assertEqual(reranked[0]["id"], "b::2")
        self.assertEqual(reranked[0]["rank"], 1)

    def test_flagembedding_rerank_scores_query_document_pairs(self):
        fake_reranker = Mock()
        fake_reranker.compute_score.return_value = [0.2, 0.9]

        with patch("rag_service.retrieval.hybrid.get_flag_reranker", return_value=fake_reranker):
            results = call_rerank_model(
                model_name="BAAI/bge-reranker-v2-m3",
                query="query",
                docs=["less relevant", "more relevant"],
                top_n=2,
            )

        self.assertEqual([item["index"] for item in results], [1, 0])
        self.assertGreater(results[0]["relevance_score"], results[1]["relevance_score"])
        fake_reranker.compute_score.assert_called_once_with(
            [["query", "less relevant"], ["query", "more relevant"]],
            normalize=True,
        )

    def test_rerank_candidates_sends_coarse_top20_and_returns_top5(self):
        rows = [{"id": str(i), "text": f"doc {i}"} for i in range(25)]
        rerank_results = [{"index": i, "relevance_score": float(100 - i)} for i in range(5)]
        with patch("rag_service.retrieval.hybrid.call_rerank_model", return_value=rerank_results) as rerank:
            reranked = rerank_candidates(
                rows=rows,
                query="query",
                top_k=5,
                rerank_top_n=20,
                model_name="BAAI/bge-reranker-v2-m3",
            )

        self.assertEqual([row["id"] for row in reranked], ["0", "1", "2", "3", "4"])
        self.assertEqual(len(reranked), 5)
        self.assertEqual(len(rerank.call_args.kwargs["docs"]), 20)
        self.assertEqual(rerank.call_args.kwargs["top_n"], 5)

    def test_rerank_candidates_uses_flagembedding_rerank_model(self):
        rows = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
        with patch(
            "rag_service.retrieval.hybrid.call_rerank_model",
            return_value=[{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.1}],
        ) as rerank:
            reranked = rerank_candidates(
                rows=rows,
                query="query",
                top_k=2,
                rerank_top_n=2,
                model_name="BAAI/bge-reranker-v2-m3",
            )

        self.assertEqual([row["id"] for row in reranked], ["b", "a"])
        self.assertEqual(rerank.call_args.kwargs["model_name"], "BAAI/bge-reranker-v2-m3")

    def test_query_debug_returns_each_retrieval_stage(self):
        dense_rows = [{"id": "a", "text": "dense"}]
        chunk_rows = [{"id": "b", "text": "bm25"}]
        bm25_rows = [{"id": "b", "text": "bm25"}]
        fused_rows = [{"id": "a", "text": "dense"}, {"id": "b", "text": "bm25"}]
        rerank_results = [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]

        with (
            patch("rag_service.retrieval.hybrid.dense_retrieve", return_value=dense_rows),
            patch("rag_service.retrieval.hybrid.read_jsonl_chunks", return_value=chunk_rows),
            patch("rag_service.retrieval.hybrid.bm25_retrieve", return_value=bm25_rows),
            patch("rag_service.retrieval.hybrid.fuse_rrf", return_value=fused_rows),
            patch("rag_service.retrieval.hybrid.call_rerank_model", return_value=rerank_results),
        ):
            result = query_debug(
                query="query",
                top_k=2,
                dense_k=10,
                bm25_k=10,
                chroma_dir="chroma",
                collection_name="chunks",
                chunk_file="chunks.jsonl",
                ollama_url="http://ollama",
                embedding_model="embed",
                reranker_model="rerank",
                rerank_top_n=20,
            )

        self.assertEqual(result["dense_rows"], dense_rows)
        self.assertEqual(result["bm25_rows"], bm25_rows)
        self.assertEqual(result["rerank_input_rows"], fused_rows)
        self.assertEqual(result["rerank_raw_results"], rerank_results)
        self.assertEqual([row["id"] for row in result["reranked_rows"]], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
