import unittest
from unittest.mock import patch

from rag_service.config import AnswerConfig, ModelConfig, QueryRewriteConfig, VectorStoreConfig


class ConfigTests(unittest.TestCase):
    def test_answer_config_reads_openai_compatible_options(self):
        with patch.dict(
            "os.environ",
            {
                "RAG_ANSWER_BASE_URL": "https://answer.example/v1",
                "RAG_ANSWER_MODEL": "answer-model",
                "RAG_ANSWER_API_KEY": "answer-key",
                "RAG_ANSWER_REASONING_SPLIT": "false",
                "RAG_ANSWER_THINKING": "disabled",
            },
            clear=True,
        ):
            cfg = AnswerConfig()

        self.assertEqual(cfg.base_url, "https://answer.example/v1")
        self.assertEqual(cfg.model, "answer-model")
        self.assertEqual(cfg.api_key, "answer-key")
        self.assertFalse(cfg.reasoning_split)
        self.assertEqual(cfg.thinking, "disabled")

    def test_answer_config_defaults_do_not_pin_vendor(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = AnswerConfig()

        self.assertEqual(cfg.base_url, "")
        self.assertEqual(cfg.model, "")
        self.assertEqual(cfg.api_key, "")
        self.assertTrue(cfg.reasoning_split)
        self.assertEqual(cfg.thinking, "")

    def test_small_model_config_is_ollama_only(self):
        with patch.dict(
            "os.environ",
            {
                "RAG_OLLAMA_BASE_URL": "http://127.0.0.1:11435",
                "RAG_QUERY_REWRITE_PROVIDER": "openai",
                "RAG_QUERY_REWRITE_BASE_URL": "https://rewrite.example",
                "RAG_QUERY_REWRITE_API_KEY": "rewrite-key",
                "RAG_QUERY_REWRITE_MODEL": "rewrite-model",
                "RAG_QUERY_REWRITE_THINKING": "disabled",
                "RAG_EMBEDDING_MODEL": "embed-model",
                "RAG_RERANK_MODEL": "rerank-model",
            },
            clear=True,
        ):
            ollama = VectorStoreConfig()
            models = ModelConfig()
            rewrite = QueryRewriteConfig()

        self.assertEqual(ollama.ollama_base_url, "http://127.0.0.1:11435")
        self.assertEqual(rewrite.model, "rewrite-model")
        self.assertEqual(rewrite.provider, "openai")
        self.assertEqual(rewrite.base_url, "https://rewrite.example")
        self.assertEqual(rewrite.api_key, "rewrite-key")
        self.assertEqual(rewrite.thinking, "disabled")
        self.assertEqual(models.embedding, "embed-model")
        self.assertEqual(models.reranker, "rerank-model")

    def test_local_retrieval_defaults_use_ollama_models(self):
        with patch.dict("os.environ", {}, clear=True):
            models = ModelConfig()
            rewrite = QueryRewriteConfig()
            ollama = VectorStoreConfig()

        self.assertEqual(ollama.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(rewrite.model, "qwen3.5:4b")
        self.assertEqual(models.embedding, "bge-m3:latest")
        self.assertEqual(models.reranker, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(rewrite.provider, "ollama")
        self.assertEqual(rewrite.base_url, "")
        self.assertEqual(rewrite.api_key, "")
        self.assertEqual(rewrite.thinking, "")
        self.assertFalse(hasattr(models, "answer"))
        self.assertFalse(hasattr(models, "query_rewrite"))
        self.assertFalse(hasattr(models, "query"))


if __name__ == "__main__":
    unittest.main()
