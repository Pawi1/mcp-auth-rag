"""
MCP Auth Starter — RAG ingest: file parsing, hierarchical chunking, and the
orchestration that turns an uploaded file into embedded, stored chunks.

Chunking never crosses a section boundary (never mixes two topics into one
chunk), then splits each section into ~target_words windows with a small
overlap. Every chunk is tagged with the filename + section it came from
before embedding (embedding_input) — a cheap, deterministic stand-in for
Anthropic's LLM-generated Contextual Retrieval: that technique calls a
generation model per chunk to write a bespoke situating sentence, which is a
real quality upgrade but needs a generation-model integration this repo
doesn't have yet. This gets most of the benefit (chunks embed with document/
section context instead of in isolation) at zero extra cost or latency.
"""

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from config import RAG_CHUNK_OVERLAP_WORDS, RAG_CHUNK_TARGET_WORDS

logger = logging.getLogger("mcp-auth-starter")

SUPPORTED_FORMATS = {"pdf", "docx", "txt", "md"}

_HEADING_SIZE_RATIO = 1.2   # a line's max font size vs. document body size
_HEADING_MAX_WORDS = 15     # headings are short; longer lines are body text


@dataclass
class Paragraph:
    page: Optional[int]
    heading: Optional[str]  # section heading in force when this paragraph occurred
    text: str


@dataclass
class Chunk:
    ordinal: int
    page: Optional[int]
    section: Optional[str]
    text: str
    word_count: int


def sniff_format(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext in SUPPORTED_FORMATS else None


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Parsing — each parser returns (paragraphs, page_count). What counts as a
# "heading" is a different signal per format, so there's one detector each;
# downstream chunking only ever looks at Paragraph.heading.
# ---------------------------------------------------------------------------

def parse_pdf(data: bytes) -> tuple[list[Paragraph], int]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        body_size = _pdf_body_font_size(doc)
        lines: list[Paragraph] = []
        current_heading: Optional[str] = None

        for page_num, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s["text"].strip()]
                    if not spans:
                        continue
                    line_text = "".join(s["text"] for s in spans).strip()
                    max_size = max(s["size"] for s in spans)
                    if _looks_like_heading(line_text, max_size, body_size):
                        current_heading = line_text
                    else:
                        lines.append(Paragraph(page=page_num, heading=current_heading, text=line_text))

        return _merge_consecutive_lines(lines), doc.page_count
    finally:
        doc.close()


def _pdf_body_font_size(doc) -> float:
    """Median font size across the whole doc, used as the 'body text'
    baseline so heading detection isn't thrown off by one oddly-set page."""
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(span["size"])
    return median(sizes) if sizes else 11.0


def _looks_like_heading(text: str, size: float, body_size: float) -> bool:
    return (
        size >= body_size * _HEADING_SIZE_RATIO
        and len(text.split()) <= _HEADING_MAX_WORDS
        and not text.endswith((".", ",", ";", ":"))
    )


def _merge_consecutive_lines(lines: list[Paragraph]) -> list[Paragraph]:
    """PDF extraction yields one Paragraph per line; merge consecutive lines
    that share a page + heading into actual paragraphs."""
    merged: list[Paragraph] = []
    for p in lines:
        if merged and merged[-1].page == p.page and merged[-1].heading == p.heading:
            merged[-1] = Paragraph(page=p.page, heading=p.heading, text=f"{merged[-1].text} {p.text}")
        else:
            merged.append(Paragraph(page=p.page, heading=p.heading, text=p.text))
    return merged


def parse_docx(data: bytes) -> tuple[list[Paragraph], int]:
    doc = DocxDocument(io.BytesIO(data))
    paragraphs: list[Paragraph] = []
    current_heading: Optional[str] = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style is not None and para.style.name.startswith(("Heading", "Title")):
            current_heading = text
        else:
            paragraphs.append(Paragraph(page=None, heading=current_heading, text=text))
    return paragraphs, 0


def parse_text(data: bytes) -> tuple[list[Paragraph], int]:
    """Shared by .txt and .md — a Markdown '#' heading is detected, anything
    else is treated as flat prose split on blank lines."""
    text = data.decode("utf-8", errors="replace")
    current_heading: Optional[str] = None
    paragraphs: list[Paragraph] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", block)
        if heading:
            current_heading = heading.group(1).strip()
            continue
        paragraphs.append(Paragraph(page=None, heading=current_heading, text=block))
    return paragraphs, 0


def parse(data: bytes, fmt: str) -> tuple[list[Paragraph], int]:
    if fmt == "pdf":
        return parse_pdf(data)
    if fmt == "docx":
        return parse_docx(data)
    if fmt in ("txt", "md"):
        return parse_text(data)
    raise ValueError(f"Unsupported format: {fmt}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_paragraphs(
    paragraphs: list[Paragraph],
    target_words: int = RAG_CHUNK_TARGET_WORDS,
    overlap_words: int = RAG_CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0

    for section in _group_by_section(paragraphs):
        words: list[str] = []
        pages: list[Optional[int]] = []
        for p in section:
            for word in p.text.split():
                words.append(word)
                pages.append(p.page)
        if not words:
            continue

        step = max(target_words - overlap_words, 1)
        i = 0
        while True:
            window_words = words[i:i + target_words]
            chunks.append(Chunk(
                ordinal=ordinal,
                page=pages[i],
                section=section[0].heading,
                text=" ".join(window_words),
                word_count=len(window_words),
            ))
            ordinal += 1
            if i + target_words >= len(words):
                break
            i += step

    return chunks


def _group_by_section(paragraphs: list[Paragraph]) -> list[list[Paragraph]]:
    groups: list[list[Paragraph]] = []
    for p in paragraphs:
        if groups and groups[-1][-1].heading == p.heading:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def embedding_input(chunk: Chunk, filename: str) -> str:
    parts = [filename] + ([chunk.section] if chunk.section else [])
    return f"[{' > '.join(parts)}] {chunk.text}"


# ---------------------------------------------------------------------------
# Orchestration — parse, chunk, embed, store. Called by the worker for each
# claimed ingest job.
# ---------------------------------------------------------------------------

async def ingest_document(document_id, owner: str, filename: str, fmt: str, data: bytes) -> None:
    import rag_embed
    import rag_store

    try:
        paragraphs, page_count = parse(data, fmt)
        chunks = chunk_paragraphs(paragraphs)
        if not chunks:
            await rag_store.set_document_status(document_id, "error", error="No extractable text found")
            return

        embeddings = await rag_embed.embed_texts([embedding_input(c, filename) for c in chunks])
        chunk_dicts = [
            {"ordinal": c.ordinal, "page": c.page, "section": c.section, "text": c.text, "word_count": c.word_count}
            for c in chunks
        ]
        await rag_store.insert_chunks(document_id, owner, chunk_dicts, embeddings)
        await rag_store.set_document_status(document_id, "done", page_count=page_count)
        logger.info(f"Ingested {filename!r}: {len(chunks)} chunks, {page_count} pages")
    except Exception as e:
        logger.exception(f"Ingest failed for {filename!r}")
        await rag_store.set_document_status(document_id, "error", error=str(e))
        raise
