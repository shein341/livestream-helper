import tempfile
import unittest
from pathlib import Path

from rag_service.ingestion.chunker import MarkdownChunker, split_markdown_sections


class ChunkMarkdownTests(unittest.TestCase):
    def test_split_markdown_sections_heading_path(self):
        text = "# 一级\n\n前言。\n\n## 二级\n\n正文段落。"
        sections = split_markdown_sections(text)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["heading_path"], ["一级"])
        self.assertEqual(sections[1]["heading_path"], ["一级", "二级"])

    def test_chunker_overlap_and_punctuation_split(self):
        text = (
            "# 规范\n\n"
            "第一条：主播应文明互动，避免争议表达。第二条：不得夸大宣传，不得承诺收益。"
            "第三条：涉及价格时应以系统展示为准，异常情况及时更正。"
            "第四条：对未成年人消费问题应优先保护并劝阻。"
        )
        chunker = MarkdownChunker(target_size=40, max_size=70, overlap=15)
        chunks = chunker.chunk_document("a.md", text)

        self.assertGreaterEqual(len(chunks), 2)
        for item in chunks:
            self.assertLessEqual(len(item["text"]), 85)

        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1]["text"][-15:]
            self.assertTrue(chunks[i]["text"].startswith(prev_tail))

    def test_legal_heading_split_without_markdown_headings(self):
        text = (
            "第一条 总则。平台建立规范。\n"
            "第二条 准入。主播需实名。\n"
            "第三条 处罚。违规将处理。"
        )
        chunker = MarkdownChunker(target_size=40, max_size=80, overlap=15)
        chunks = chunker.chunk_document("rules.md", text)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(any("第一条" in "".join(c["heading_path"]) for c in chunks))

    def test_chunk_dir_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "processed_md"
            raw.mkdir()
            (raw / "rules.md").write_text("# 标题\n\n这是第一句。这是第二句。", encoding="utf-8")

            out = Path(tmp) / "chunks.jsonl"
            chunker = MarkdownChunker(target_size=20, max_size=40, overlap=15)
            rows = chunker.chunk_directory(raw, out, recursive=True)

            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn('"source"', content)

    def test_heading_only_lines_do_not_become_standalone_chunks(self):
        text = (
            "一、账号与权限检查\n"
            "1. 主播账号需完成实名认证，且实名姓名与收款账户信息一致。\n"
            "2. 收款账户需为本人银行卡，不支持对公账户或他人代收。\n"
        )
        chunker = MarkdownChunker(target_size=200, max_size=400, overlap=0)
        chunks = chunker.chunk_document("rules.md", text)
        values = [c["text"].strip() for c in chunks]
        self.assertNotIn("一、账号与权限检查", values)
        self.assertTrue(any("实名认证" in v for v in values))


    def test_short_numbered_items_under_same_heading_are_merged(self):
        text = (
            "## 提现前置条件\n\n"
            "1. 主播账号需完成实名认证，且实名姓名与收款账户信息一致。\n"
            "2. 收款账户需为本人银行卡，不支持对公账户或他人代收。\n"
            "3. 账号当前不可处于封禁、冻结、限制提现等风控状态。\n"
            "4. 若存在未完结退款、投诉争议单，平台可临时延后提现审核。\n"
        )
        chunker = MarkdownChunker(target_size=220, max_size=400, overlap=0)
        chunks = chunker.chunk_document("rules.md", text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["heading_path"], ["提现前置条件"])
        self.assertIn("1. 主播账号需完成实名认证", chunks[0]["text"])
        self.assertIn("4. 若存在未完结退款", chunks[0]["text"])

    def test_leading_plain_title_becomes_heading_context(self):
        text = (
            "主播直播运营规范（主播侧）\n\n"
            "第一章 目的与原则\n"
            "第一条 主播应保持真实表达。\n"
        )
        chunker = MarkdownChunker(target_size=220, max_size=400, overlap=0)
        chunks = chunker.chunk_document("rules.md", text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["heading_path"], ["主播直播运营规范（主播侧）", "第一章 目的与原则"])
        self.assertIn("第一条 主播应保持真实表达", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
