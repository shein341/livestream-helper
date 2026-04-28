import unittest
from unittest.mock import Mock, patch

from rag_service.generation.chat import (
    build_parser,
    build_context_blocks,
    build_generation_prompt,
    call_answer_generate,
    format_references,
    run_rag_chat,
)


class RagChatTests(unittest.TestCase):
    def test_build_context_blocks(self):
        rows = [
            {
                "source": "a.md",
                "heading_path": "章一 > 条一",
                "chunk_id": 1,
                "text": "第一条 提现T+1到账。",
            },
            {
                "source": "b.md",
                "heading_path": "章二 > 条三",
                "chunk_id": 2,
                "text": "第二条 禁止承诺收益。",
            },
        ]
        blocks = build_context_blocks(rows=rows, max_context_chars=200)
        self.assertEqual(len(blocks), 2)
        self.assertIn("[1]", blocks[0])
        self.assertIn("a.md", blocks[0])

    def test_build_generation_prompt(self):
        prompt = build_generation_prompt(
            query="提现多久到账",
            context_blocks=["[1] source=a.md\ntext"],
        )
        self.assertIn("提现多久到账", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("must include citations", prompt.lower())

    def test_format_references(self):
        rows = [
            {"source": "a.md", "heading_path": "章一 > 条一", "chunk_id": 1, "text": "第一条 提现T+1到账。"},
            {"source": "b.md", "heading_path": "章二 > 条三", "chunk_id": 2, "text": "第二条 禁止承诺收益。"},
        ]
        refs = format_references(rows)
        self.assertEqual(len(refs), 2)
        self.assertIn("[1]", refs[0])
        self.assertIn("第一条", refs[0])
        self.assertNotIn("source=", refs[0])

    def test_call_answer_generate_uses_answer_config(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "答案"}}]}

        with (
            patch("rag_service.generation.chat.ANSWER.base_url", "https://answer.example/v1"),
            patch("rag_service.generation.chat.ANSWER.api_key", "answer-key"),
            patch("rag_service.generation.chat.ANSWER.model", "answer-model"),
            patch("rag_service.generation.chat.ANSWER.reasoning_split", True),
            patch("rag_service.generation.chat.ANSWER.thinking", ""),
            patch("rag_service.generation.chat.requests.post", return_value=resp) as post,
        ):
            answer = call_answer_generate("prompt")

        self.assertEqual(answer, "答案")
        self.assertEqual(post.call_args.args[0], "https://answer.example/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "answer-model")
        self.assertTrue(post.call_args.kwargs["json"]["reasoning_split"])

    def test_call_answer_generate_can_disable_deepseek_thinking(self):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "答案"}}]}

        with (
            patch("rag_service.generation.chat.ANSWER.base_url", "https://api.deepseek.com"),
            patch("rag_service.generation.chat.ANSWER.api_key", "answer-key"),
            patch("rag_service.generation.chat.ANSWER.model", "deepseek-v4-flash"),
            patch("rag_service.generation.chat.ANSWER.reasoning_split", False),
            patch("rag_service.generation.chat.ANSWER.thinking", "disabled"),
            patch("rag_service.generation.chat.requests.post", return_value=resp) as post,
        ):
            answer = call_answer_generate("prompt")

        self.assertEqual(answer, "答案")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_split", payload)

    def test_chat_parser_has_no_query_model_option(self):
        args = build_parser().parse_args(["提现多久到账"])
        self.assertFalse(hasattr(args, "query_model"))

    def test_run_rag_chat_returns_fixed_text_when_rerank_confidence_is_low(self):
        args = build_parser().parse_args(["没依据的问题"])
        low_confidence_rows = [{"text": "irrelevant", "rerank_score": 0.49}]

        with (
            patch("rag_service.generation.chat.query_top_k", return_value=low_confidence_rows),
            patch("rag_service.generation.chat.call_answer_generate") as answer_generate,
        ):
            result = run_rag_chat(args)

        answer_generate.assert_not_called()
        self.assertEqual(result["answer"], "无确信的依据，请问点别的问题吧~")
        self.assertEqual(result["references"], [])
        self.assertEqual(result["rows"], low_confidence_rows)

    def test_run_rag_chat_calls_answer_model_at_confidence_threshold(self):
        args = build_parser().parse_args(["有依据的问题"])
        rows = [{"text": "relevant", "rerank_score": 0.5}]

        with (
            patch("rag_service.generation.chat.query_top_k", return_value=rows),
            patch("rag_service.generation.chat.call_answer_generate", return_value="答案") as answer_generate,
        ):
            result = run_rag_chat(args)

        answer_generate.assert_called_once()
        self.assertEqual(result["answer"], "答案")

    def test_run_rag_chat_excludes_low_rerank_scores_from_prompt_context(self):
        args = build_parser().parse_args(["有部分噪声的问题"])
        rows = [
            {"source": "a.md", "heading_path": "", "chunk_id": 1, "text": "high confidence", "rerank_score": 0.8},
            {"source": "b.md", "heading_path": "", "chunk_id": 2, "text": "low confidence", "rerank_score": 0.09},
        ]

        with (
            patch("rag_service.generation.chat.query_top_k", return_value=rows),
            patch("rag_service.generation.chat.call_answer_generate", return_value="答案") as answer_generate,
        ):
            result = run_rag_chat(args)

        prompt = answer_generate.call_args.kwargs["prompt"]
        self.assertIn("high confidence", prompt)
        self.assertNotIn("low confidence", prompt)
        self.assertEqual(result["references"], ["[1] high confidence"])


if __name__ == "__main__":
    unittest.main()
