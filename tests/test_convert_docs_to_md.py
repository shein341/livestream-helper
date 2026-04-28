import tempfile
import unittest
from pathlib import Path
import zipfile

from rag_service.ingestion.converter import (
    ConversionError,
    DocumentConverter,
    discover_source_files,
    extract_docx_text_via_zip,
    normalize_text,
)


class ConvertDocsTests(unittest.TestCase):
    def test_normalize_text(self):
        raw = "a\r\n\r\n\r\n b  \r\nc\r"
        self.assertEqual(normalize_text(raw), "a\n\n b\nc\n")

    def test_discover_source_files_non_recursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("x", encoding="utf-8")
            (root / "b.md").write_text("x", encoding="utf-8")
            (root / "c.docx").write_text("x", encoding="utf-8")
            (root / "d.png").write_text("x", encoding="utf-8")
            sub = root / "sub"
            sub.mkdir()
            (sub / "e.txt").write_text("x", encoding="utf-8")

            files = discover_source_files(root, recursive=False)
            names = sorted(p.name for p in files)
            self.assertEqual(names, ["a.txt", "b.md", "c.docx"])

    def test_convert_txt_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            out = Path(tmp) / "out"
            src.mkdir()
            (src / "rules.txt").write_text("主播直播规范\n\n禁止夸大宣传", encoding="utf-8")
            (src / "withdraw.md").write_text("# 提现规则\n\nT+1到账", encoding="utf-8")

            converter = DocumentConverter(word_text_extractor=lambda _: "")
            converted = converter.convert(src, out, recursive=False, overwrite=True)

            self.assertEqual(len(converted), 2)
            self.assertTrue((out / "rules.md").exists())
            self.assertTrue((out / "withdraw.md").exists())

            rules_content = (out / "rules.md").read_text(encoding="utf-8")
            self.assertIn("主播直播规范", rules_content)

    def test_doc_and_docx_use_word_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            out = Path(tmp) / "out"
            src.mkdir()
            (src / "a.doc").write_text("placeholder", encoding="utf-8")
            (src / "b.docx").write_text("placeholder", encoding="utf-8")

            calls = []

            def fake_extractor(path: Path) -> str:
                calls.append(path.name)
                return f"converted:{path.name}"

            converter = DocumentConverter(word_text_extractor=fake_extractor)
            converted = converter.convert(src, out, recursive=False, overwrite=True)

            self.assertEqual(len(converted), 2)
            self.assertEqual(sorted(calls), ["a.doc", "b.docx"])
            self.assertIn("converted:a.doc", (out / "a.md").read_text(encoding="utf-8"))
            self.assertIn("converted:b.docx", (out / "b.md").read_text(encoding="utf-8"))

    def test_convert_fails_fast_on_first_conversion_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            out = Path(tmp) / "out"
            src.mkdir()
            (src / "a.doc").write_text("placeholder", encoding="utf-8")

            def failing_extractor(_: Path) -> str:
                raise RuntimeError("word unavailable")

            converter = DocumentConverter(word_text_extractor=failing_extractor)
            with self.assertRaises(ConversionError):
                converter.convert(src, out, recursive=False, overwrite=True)

    def test_extract_docx_text_via_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.docx"
            document_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>第一行</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>第二行</w:t></w:r></w:p>"
                "</w:body>"
                "</w:document>"
            )
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("word/document.xml", document_xml)

            text = extract_docx_text_via_zip(path)
            self.assertIn("第一行", text)
            self.assertIn("第二行", text)


if __name__ == "__main__":
    unittest.main()
