import argparse
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable

from rag_service.config import DATA

SUPPORTED_SUFFIXES = {".doc", ".docx", ".txt", ".md"}
DEFAULT_RAW_DIR = DATA.raw_dir
DEFAULT_PROCESSED_DIR = DATA.processed_dir


class ConversionError(Exception):
    pass


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    compact_lines = []
    blank_streak = 0
    for line in lines:
        if line.strip() == "":
            blank_streak += 1
            if blank_streak <= 1:
                compact_lines.append("")
        else:
            blank_streak = 0
            compact_lines.append(line)

    normalized = "\n".join(compact_lines).lstrip("\ufeff")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def read_text_with_fallbacks(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ConversionError(f"Cannot decode text file: {path}")


def extract_docx_text_via_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml_payload = zf.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ConversionError(f"Cannot read DOCX payload from {path}: {exc}") from exc

    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as exc:
        raise ConversionError(f"Invalid DOCX XML schema in {path}: {exc}") from exc

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for p in root.findall(".//w:p", ns):
        texts: list[str] = []
        for node in p.findall(".//w:t", ns):
            if node.text:
                texts.append(node.text)
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def extract_doc_text_via_antiword(path: Path) -> str:
    if shutil.which("antiword") is None:
        raise ConversionError("antiword is not installed")
    result = subprocess.run(
        ["antiword", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise ConversionError(f"antiword failed for {path}: {stderr}")
    return result.stdout


def extract_word_text_via_soffice(path: Path) -> str:
    if shutil.which("soffice") is None:
        raise ConversionError("soffice is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                str(out_dir),
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ConversionError(f"soffice conversion failed for {path}: {stderr}")
        txt_files = sorted(out_dir.glob("*.txt"))
        if not txt_files:
            raise ConversionError(f"soffice conversion produced no .txt for {path}")
        return read_text_with_fallbacks(txt_files[0])


def extract_word_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            return extract_docx_text_via_zip(path)
        except ConversionError:
            return extract_word_text_via_soffice(path)
    if suffix == ".doc":
        try:
            return extract_doc_text_via_antiword(path)
        except ConversionError:
            return extract_word_text_via_soffice(path)
    raise ConversionError(f"Unsupported Word suffix: {suffix}")


def discover_source_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_SUFFIXES else []

    if recursive:
        candidates = input_path.rglob("*")
    else:
        candidates = input_path.glob("*")

    return sorted(
        p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


class DocumentConverter:
    def __init__(self, word_text_extractor: Callable[[Path], str] = extract_word_text):
        self.word_text_extractor = word_text_extractor

    def _to_markdown(self, source_file: Path) -> str:
        suffix = source_file.suffix.lower()
        if suffix in {".txt", ".md"}:
            raw = read_text_with_fallbacks(source_file)
        elif suffix in {".doc", ".docx"}:
            raw = self.word_text_extractor(source_file)
        else:
            raise ConversionError(f"Unsupported file type: {source_file}")

        return normalize_text(raw)

    def convert(
        self,
        input_path: Path,
        output_dir: Path,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> list[Path]:
        sources = discover_source_files(input_path, recursive=recursive)
        if not sources:
            raise ConversionError(f"No supported files found under {input_path}")

        converted = []

        output_dir.mkdir(parents=True, exist_ok=True)

        base_dir = input_path.parent if input_path.is_file() else input_path

        for src in sources:
            try:
                rel = src.relative_to(base_dir)
                out_path = output_dir / rel
                out_path = out_path.with_suffix(".md")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists() and not overwrite:
                    raise ConversionError(f"Output exists (use --overwrite): {out_path}")

                content = self._to_markdown(src)
                out_path.write_text(content, encoding="utf-8")
                converted.append(out_path)
            except Exception as exc:  # noqa: BLE001
                raise ConversionError(f"Failed converting {src}: {exc}") from exc

        return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert plain-text doc/docx/txt/md files into normalized markdown files."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input file or directory")
    parser.add_argument("output", nargs="?", type=Path, help="Output directory")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw document directory")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed markdown directory",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete processed directory before conversion",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipeline_mode = args.input is None and args.output is None
    if args.raw_dir is not None or args.processed_dir is not None:
        pipeline_mode = True

    if pipeline_mode:
        input_path = args.raw_dir or DEFAULT_RAW_DIR
        output_dir = args.processed_dir or DEFAULT_PROCESSED_DIR
    else:
        if args.input is None or args.output is None:
            print("ERROR: provide both input and output, or use --raw-dir/--processed-dir.", file=sys.stderr)
            return 2
        input_path = args.input
        output_dir = args.output

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    converter = DocumentConverter()
    try:
        converted = converter.convert(
            input_path=input_path,
            output_dir=output_dir,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Converted: {len(converted)}")
    for item in converted:
        print(f"  OK  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
