import argparse
import json
import re
from pathlib import Path

from rag_service.config import DATA, VECTOR_STORE

PUNCTUATION = "。！？；.!?;"
DEFAULT_INPUT_DIR = DATA.processed_dir
DEFAULT_OUTPUT_FILE = VECTOR_STORE.chunk_file

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+[章节篇]\s*.*")
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+条\s*.*")
CN_ITEM_RE = re.compile(r"^[一二三四五六七八九十]+、\s*.*")
NUM_ITEM_RE = re.compile(r"^\d+[\.、]\s*.*")

INLINE_MARKERS = [
    r"第[一二三四五六七八九十百千万零〇0-9]+[章节篇]",
    r"第[一二三四五六七八九十百千万零〇0-9]+条",
    r"[一二三四五六七八九十]+、",
    r"\d+[\.、]",
]


def _strip_clause_prefix(level: int, title: str) -> str:
    if level == 1:
        return re.sub(r"^第[一二三四五六七八九十百千万零〇0-9]+[章节篇]\s*", "", title).strip()
    if level == 2:
        return re.sub(r"^第[一二三四五六七八九十百千万零〇0-9]+条\s*", "", title).strip()
    if level == 3:
        return re.sub(r"^[一二三四五六七八九十]+、\s*", "", title).strip()
    if level == 4:
        return re.sub(r"^\d+[\.、]\s*", "", title).strip()
    return title.strip()


def should_keep_heading_line(level: int, title: str) -> bool:
    payload = _strip_clause_prefix(level, title)
    if not payload:
        return False
    if re.search(r"[。！？；.!?;，,：:]", payload):
        return True
    # Long payloads without terminal punctuation can still carry substantive content.
    return len(payload) >= 18


def split_markdown_sections(markdown_text: str) -> list[dict]:
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_stack: list[tuple[int, str]] = []
    sections: list[dict] = []
    current_lines: list[str] = []

    def flush_current() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append(
            {
                "heading_path": [title for _, title in heading_stack],
                "text": text,
            }
        )

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush_current()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            current_lines.append(line)

    flush_current()
    return sections


def detect_clause_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if CHAPTER_RE.match(stripped):
        return (1, stripped)
    if ARTICLE_RE.match(stripped):
        return (2, stripped)
    if CN_ITEM_RE.match(stripped):
        return (3, stripped)
    if NUM_ITEM_RE.match(stripped):
        return (4, stripped)
    return None


def split_legal_sections(text: str, base_heading_path: list[str]) -> list[dict]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    sections: list[dict] = []
    has_structural_parent = bool(base_heading_path)

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if not content:
            return
        sections.append(
            {
                "heading_path": base_heading_path + [title for _, title in stack],
                "text": content,
            }
        )

    found_clause_heading = False
    for line in lines:
        hit = detect_clause_heading(line)
        if hit is None:
            current_lines.append(line)
            continue

        level, title = hit
        is_parent_heading = level in {1, 3}
        is_standalone_clause_heading = level in {2, 4} and not has_structural_parent
        is_strong_heading = is_parent_heading or is_standalone_clause_heading
        if not is_strong_heading:
            current_lines.append(line.strip())
            continue

        found_clause_heading = True
        if is_parent_heading:
            has_structural_parent = True
        flush()
        current_lines = []

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        # Only keep heading line in text when it has substantive inline content.
        if should_keep_heading_line(level, title):
            current_lines.append(line.strip())

    flush()
    if found_clause_heading and sections:
        return sections

    return [{"heading_path": base_heading_path, "text": text.strip()}] if text.strip() else []


def pre_split_by_inline_markers(text: str) -> str:
    out = text
    for marker in INLINE_MARKERS:
        out = re.sub(rf"(?<!\n)\s*({marker})", r"\n\n\1", out)
    return out


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for idx, ch in enumerate(text):
        buf += ch
        if ch not in PUNCTUATION:
            continue
        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < len(text) else ""
        if ch == "." and prev_ch.isdigit() and (not next_ch or next_ch.isspace()):
            continue
        if ch in PUNCTUATION:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [x for x in out if x]


def split_long_sentence(sentence: str, max_size: int) -> list[str]:
    if len(sentence) <= max_size:
        return [sentence]

    comma_parts = re.split(r"([，,：:])", sentence)
    rebuilt: list[str] = []
    cur = ""
    for part in comma_parts:
        if not part:
            continue
        cur += part
        if part in "，,：:":
            rebuilt.append(cur.strip())
            cur = ""
    if cur.strip():
        rebuilt.append(cur.strip())

    if len(rebuilt) == 1 and len(rebuilt[0]) > max_size:
        text = rebuilt[0]
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    out: list[str] = []
    for part in rebuilt:
        if len(part) <= max_size:
            out.append(part)
        else:
            out.extend(split_long_sentence(part, max_size))
    return out


def chunk_section_text(text: str, target_size: int, max_size: int, overlap: int) -> list[str]:
    normalized = pre_split_by_inline_markers(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    segments: list[str] = []

    for para in paragraphs:
        sentences = split_sentences(para)
        if not sentences:
            continue
        for sen in sentences:
            segments.extend(split_long_sentence(sen, max_size=max_size))

    chunks_base: list[str] = []
    current = ""

    for seg in segments:
        candidate = seg if not current else f"{current}\n{seg}"
        if len(candidate) <= max_size:
            current = candidate
            continue

        if current:
            chunks_base.append(current)
        current = seg

        if len(current) > max_size:
            forced = split_long_sentence(current, max_size=max_size)
            chunks_base.extend(forced[:-1])
            current = forced[-1]

    if current:
        chunks_base.append(current)

    merged: list[str] = []
    for part in chunks_base:
        if merged and len(merged[-1]) < int(target_size * 0.6) and len(merged[-1]) + len(part) <= max_size:
            merged[-1] += "\n" + part
        else:
            merged.append(part)

    if overlap <= 0 or len(merged) <= 1:
        return merged

    with_overlap = [merged[0]]
    for idx in range(1, len(merged)):
        prev_plain = merged[idx - 1]
        tail = prev_plain[-overlap:] if len(prev_plain) >= overlap else prev_plain
        with_overlap.append(tail + merged[idx])
    return with_overlap


class MarkdownChunker:
    def __init__(self, target_size: int = 600, max_size: int = 900, overlap: int = 15):
        if target_size <= 0 or max_size <= 0:
            raise ValueError("target_size and max_size must be positive")
        if target_size > max_size:
            raise ValueError("target_size cannot be greater than max_size")
        self.target_size = target_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk_document(self, source_name: str, markdown_text: str) -> list[dict]:
        rows: list[dict] = []
        section_candidates = split_markdown_sections(markdown_text)
        expanded_sections: list[dict] = []
        for sec in section_candidates:
            expanded_sections.extend(split_legal_sections(sec["text"], sec["heading_path"]))

        if len(expanded_sections) > 1:
            first = expanded_sections[0]
            title = str(first.get("text", "")).strip()
            if not first.get("heading_path") and "\n" not in title and 0 < len(title) <= 80:
                expanded_sections = expanded_sections[1:]
                for sec in expanded_sections:
                    sec["heading_path"] = [title] + list(sec.get("heading_path") or [])

        chunk_id = 0
        for sec in expanded_sections:
            heading_path = sec["heading_path"]
            chunks = chunk_section_text(
                sec["text"],
                target_size=self.target_size,
                max_size=self.max_size,
                overlap=self.overlap,
            )
            for text in chunks:
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source_name,
                        "heading_path": heading_path,
                        "char_count": len(text),
                        "text": text,
                    }
                )
                chunk_id += 1
        return rows

    def chunk_directory(self, input_dir: Path, output_file: Path, recursive: bool = True) -> list[dict]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")

        files = sorted(input_dir.rglob("*.md") if recursive else input_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No markdown files found under {input_dir}")

        all_rows: list[dict] = []

        for file in files:
            text = file.read_text(encoding="utf-8")
            rows = self.chunk_document(file.name, text)
            all_rows.extend(rows)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk markdown docs by heading + recursive paragraph/sentence strategy for RAG."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--target-size", type=int, default=DATA.chunk_target_size)
    parser.add_argument("--max-size", type=int, default=DATA.chunk_max_size)
    parser.add_argument("--overlap", type=int, default=DATA.chunk_overlap)
    parser.add_argument("--non-recursive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunker = MarkdownChunker(
        target_size=args.target_size,
        max_size=args.max_size,
        overlap=args.overlap,
    )
    rows = chunker.chunk_directory(
        input_dir=args.input_dir,
        output_file=args.output_file,
        recursive=not args.non_recursive,
    )
    print(f"InputDir: {args.input_dir}")
    print(f"OutputFile: {args.output_file}")
    print(f"Chunks: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
