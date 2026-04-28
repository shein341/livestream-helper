import unittest
from unittest.mock import Mock, patch

from rag_service.retrieval.query_rewrite import build_rewrite_prompt, rewrite_query


class QueryRewriteTests(unittest.TestCase):
    def test_build_rewrite_prompt_uses_chinese_declarative_query_instruction(self):
        prompt = build_rewrite_prompt("主播提现多久到账？")

        self.assertIn("只输出", prompt)
        self.assertIn("陈述式检索句", prompt)
        self.assertIn("40字以内", prompt)
        self.assertIn("不要拆成关键词", prompt)
        self.assertIn("【主播提现多久到账？】", prompt)
        self.assertIn("主播提现多久到账？", prompt)
        self.assertLessEqual(len(prompt), 150)

    def test_ollama_rewrite_uses_chat_with_thinking_disabled(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "提现 到账 时间"}}

        with (
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.provider", "ollama"),
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.model", "qwen3.5:4b"),
            patch("rag_service.retrieval.query_rewrite.requests.post", return_value=resp) as post,
        ):
            rewritten = rewrite_query("主播提现多久到账？")

        self.assertEqual(rewritten, "提现 到账 时间")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen3.5:4b")
        self.assertFalse(payload["think"])

    def test_openai_compatible_rewrite_can_use_deepseek_flash(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "主播提现前需要满足的条件"}}]}

        with (
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.provider", "openai"),
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.base_url", "https://api.deepseek.com"),
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.api_key", "rewrite-key"),
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.model", "deepseek-v4-flash"),
            patch("rag_service.retrieval.query_rewrite.QUERY_REWRITE.thinking", "disabled"),
            patch("rag_service.retrieval.query_rewrite.requests.post", return_value=resp) as post,
        ):
            rewritten = rewrite_query("主播提现前需要满足哪些条件？")

        self.assertEqual(rewritten, "主播提现前需要满足的条件")
        self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 48)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIn("主播提现前需要满足哪些条件？", payload["messages"][0]["content"])

    def test_rewrite_query_fails_fast_on_empty_model_output(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "   "}}
        with patch("rag_service.retrieval.query_rewrite.requests.post", return_value=resp):
            with self.assertRaises(RuntimeError):
                rewrite_query("提现多久到账")

    def test_rewrite_query_rejects_blank_question(self):
        with self.assertRaises(ValueError):
            rewrite_query("   ")


if __name__ == "__main__":
    unittest.main()
