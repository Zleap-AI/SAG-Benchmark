"""Sentence-preserving chunks aligned with SAG-Desktop core/src/text.rs."""

import re
from pathlib import Path

DEFAULT_CHUNK_CHARS = 2200
CHUNK_POLICY = "desktop-sentences-v1"
CODE_SUFFIXES = set(
    ".rs .py .js .jsx .ts .tsx .mjs .cjs .go .java .c .h .cpp .hpp .cc .cs .rb .php "
    ".swift .kt .kts .scala .sh .bash .zsh .ps1 .sql .html .htm .css .scss .vue .svelte "
    ".r .lua .dart .json .jsonc .yaml .yml .toml .ini .cfg .conf .xml".split()
)
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
CLOSERS = "\"'”’）)]}"


def utf16_length(text):
    return len(text.encode("utf-16-le")) // 2


def sentence_ranges(source, start, end, code):
    cursor, position = start, start
    result = []
    while position < end:
        char = source[position]
        boundary = (
            char == "\n"
            if code
            else char in "。！？!?"
            or char == "."
            and (
                position + 1 == end
                or source[position + 1].isspace()
                or source[position + 1] in CLOSERS
            )
        )
        position += 1
        if not boundary:
            continue
        if not code:
            while position < end and (
                source[position].isspace() or source[position] in "。！？!?" + CLOSERS
            ):
                position += 1
        result.append((cursor, position))
        cursor = position
    if cursor < end:
        result.append((cursor, end))
    return result


def source_content(source, spans):
    result = ""
    for start, end in spans:
        part = source[start:end]
        if result and not result.endswith("\n") and not part.startswith("\n"):
            result += "\n\n"
        result += part
    return result


def chunk_document(source, target=DEFAULT_CHUNK_CHARS, path=None):
    """Return headings, exact source ranges and content; long sentences stay intact."""
    if target < 2:
        raise ValueError("Chunk target must be at least two characters")
    file = Path(path) if path else Path("document.md")
    markdown = file.suffix.lower() in {".md", ".markdown"}
    code = file.suffix.lower() in CODE_SUFFIXES or file.name in {
        "Dockerfile", "Containerfile", "Makefile", "CMakeLists.txt"
    }
    lines = []
    start, i = 0, 0
    while i < len(source):
        if source[i] in "\r\n":
            lines.append((start, i))
            if source[i : i + 2] == "\r\n":
                i += 1
            start = i + 1
        i += 1
    lines.append((start, len(source)))
    blocks, headings = [], []
    first, fence = None, None
    front = markdown and source[slice(*lines[0])].strip() == "---"

    def flush(last, is_code):
        nonlocal first
        if first is not None:
            start, end = lines[first][0], lines[last][1]
            if source[start:end].strip():
                blocks.append((" / ".join(h for h in headings if h), start, end, is_code))
            first = None

    if not markdown:
        if source.strip():
            blocks.append(("", 0, len(source), code))
    else:
        for i, (start, end) in enumerate(lines):
            line = source[start:end]
            if front:
                if i > 0 and line.strip() in {"---", "..."}:
                    front = False
                continue
            match = FENCE.match(line)
            if match:
                marker = match[1]
                if fence is None:
                    if i > 0:
                        flush(i - 1, False)
                    first, fence = i, marker
                elif fence.startswith(marker[0]) and len(marker) >= len(fence):
                    fence = None
                    flush(i, True)
                continue
            if fence is not None:
                continue
            match = HEADING.match(line)
            if match:
                if i > 0:
                    flush(i - 1, False)
                level = len(match[1])
                headings = (headings + [""] * level)[:level]
                headings[level - 1] = match[2]
            elif not line.strip():
                if i > 0:
                    flush(i - 1, False)
            elif first is None:
                first = i
        flush(len(lines) - 1, fence is not None)

    overlap = 80 if target >= 160 else 0
    limit = target - (overlap + 2 if overlap else 0)
    chunks, units, length = [], [], 0
    for heading, start, end, code in blocks:
        sentences = sentence_ranges(source, start, end, code)
        groups, cursor, used = [], start, 0
        for s, e in sentences:
            size = utf16_length(source[s:e])
            if used and used + size > limit:
                groups.append((cursor, s))
                cursor, used = s, 0
            used += size
        if cursor < end:
            groups.append((cursor, end))
        for s, e in groups:
            size = utf16_length(source[s:e])
            group_units = sentence_ranges(source, s, e, code)
            if not chunks or chunks[-1]["heading"] != heading or length + 2 + size > limit:
                chunks.append({"heading": heading, "spans": [(s, e)]})
                units.append(group_units)
                length = size
            else:
                chunks[-1]["spans"].append((s, e))
                units[-1].extend(group_units)
                length += 2 + size
    for i in range(len(chunks) - 1, 0, -1):
        if not overlap or chunks[i - 1]["heading"] != chunks[i]["heading"]:
            continue
        prefix, used = [], 0
        for start, end in reversed(units[i - 1]):
            size = utf16_length(source[start:end])
            if prefix and used + size > overlap:
                break
            prefix.append((start, end))
            used += size
            if used >= overlap:
                break
        chunks[i]["spans"] = list(reversed(prefix)) + chunks[i]["spans"]
    for chunk in chunks:
        merged = []
        for start, end in chunk["spans"]:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        chunk["spans"] = merged
        chunk["content"] = source_content(source, merged)
    return chunks
