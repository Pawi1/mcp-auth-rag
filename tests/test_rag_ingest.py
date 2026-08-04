"""Tests for rag_ingest.py: format sniffing, PDF/DOCX/text parsing (heading
detection), hierarchical chunking (section boundaries + overlap), and the
ingest_document() orchestration (with rag_store/rag_embed mocked)."""

import io
from unittest.mock import AsyncMock

import fitz
import pytest
from docx import Document as DocxDocument

import rag_ingest as ri


def _make_pdf(sections: list[tuple[str, int, str]]) -> bytes:
    """sections: [(heading, n_words, word_prefix), ...] — one page per section."""
    doc = fitz.open()
    for heading, n_words, prefix in sections:
        page = doc.new_page()
        page.insert_text((72, 72), heading, fontsize=18)
        body = " ".join(f"{prefix}{i}" for i in range(n_words))
        rc = page.insert_textbox(fitz.Rect(72, 100, 523, 780), body, fontsize=9)
        assert rc >= 0, f"test fixture text overflowed the page (rc={rc})"
    data = doc.tobytes()
    doc.close()
    return data


class TestSniffFormat:
    def test_recognizes_supported_extensions(self):
        assert ri.sniff_format("paper.pdf") == "pdf"
        assert ri.sniff_format("paper.PDF") == "pdf"
        assert ri.sniff_format("notes.docx") == "docx"
        assert ri.sniff_format("readme.md") == "md"
        assert ri.sniff_format("plain.txt") == "txt"

    def test_rejects_unsupported_extensions(self):
        assert ri.sniff_format("virus.exe") is None
        assert ri.sniff_format("no_extension") is None


class TestContentHash:
    def test_stable_for_same_bytes(self):
        data = b"some pdf bytes"
        assert ri.content_hash(data) == ri.content_hash(data)

    def test_differs_for_different_bytes(self):
        assert ri.content_hash(b"a") != ri.content_hash(b"b")


class TestParsePdf:
    def test_detects_headings_by_font_size(self):
        pdf = _make_pdf([("Wprowadzenie", 200, "slowo"), ("Metodologia badan", 200, "inneslowo")])
        paragraphs, page_count = ri.parse_pdf(pdf)
        assert page_count == 2
        headings = {p.heading for p in paragraphs}
        assert headings == {"Wprowadzenie", "Metodologia badan"}

    def test_paragraphs_carry_their_page_number(self):
        pdf = _make_pdf([("Sekcja A", 50, "a"), ("Sekcja B", 50, "b")])
        paragraphs, _ = ri.parse_pdf(pdf)
        pages_for_a = {p.page for p in paragraphs if p.heading == "Sekcja A"}
        pages_for_b = {p.page for p in paragraphs if p.heading == "Sekcja B"}
        assert pages_for_a == {1}
        assert pages_for_b == {2}


class TestParseDocx:
    def _docx_bytes(self) -> bytes:
        d = DocxDocument()
        d.add_heading("Rozdzial 1", level=1)
        d.add_paragraph(" ".join(f"abc{i}" for i in range(50)))
        d.add_heading("Rozdzial 2", level=1)
        d.add_paragraph(" ".join(f"xyz{i}" for i in range(50)))
        buf = io.BytesIO()
        d.save(buf)
        return buf.getvalue()

    def test_heading_styles_become_section_boundaries(self):
        paragraphs, page_count = ri.parse_docx(self._docx_bytes())
        assert page_count == 0
        assert {p.heading for p in paragraphs} == {"Rozdzial 1", "Rozdzial 2"}

    def test_body_paragraphs_are_not_treated_as_headings(self):
        paragraphs, _ = ri.parse_docx(self._docx_bytes())
        assert all(p.text not in ("Rozdzial 1", "Rozdzial 2") for p in paragraphs)


class TestParseText:
    def test_markdown_headings_split_sections(self):
        md = b"# Wstep\n\nAkapit pierwszy.\n\n# Wyniki\n\nAkapit drugi z wynikami."
        paragraphs, page_count = ri.parse_text(md)
        assert page_count == 0
        assert [p.heading for p in paragraphs] == ["Wstep", "Wyniki"]

    def test_plain_text_without_headings_has_none_section(self):
        paragraphs, _ = ri.parse_text(b"Just one paragraph.\n\nAnd another.")
        assert all(p.heading is None for p in paragraphs)


class TestChunkParagraphs:
    def test_never_crosses_a_section_boundary(self):
        pdf = _make_pdf([("Wprowadzenie", 200, "slowo"), ("Metodologia badan", 200, "inneslowo")])
        paragraphs, _ = ri.parse_pdf(pdf)
        chunks = ri.chunk_paragraphs(paragraphs, target_words=50, overlap_words=10)
        sections = {c.section for c in chunks}
        assert sections == {"Wprowadzenie", "Metodologia badan"}
        # every word in a "Wprowadzenie" chunk actually came from that section
        for c in chunks:
            if c.section == "Wprowadzenie":
                assert all(w.startswith("slowo") for w in c.text.split())

    def test_consecutive_chunks_overlap_by_the_configured_amount(self):
        pdf = _make_pdf([("Sekcja", 200, "w")])
        paragraphs, _ = ri.parse_pdf(pdf)
        chunks = ri.chunk_paragraphs(paragraphs, target_words=50, overlap_words=10)
        assert len(chunks) >= 2
        assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]

    def test_empty_input_yields_no_chunks(self):
        assert ri.chunk_paragraphs([]) == []

    def test_ordinals_are_sequential(self):
        pdf = _make_pdf([("A", 100, "a"), ("B", 100, "b")])
        paragraphs, _ = ri.parse_pdf(pdf)
        chunks = ri.chunk_paragraphs(paragraphs, target_words=30, overlap_words=5)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestEmbeddingInput:
    def test_includes_filename_and_section(self):
        chunk = ri.Chunk(ordinal=0, page=1, section="Wyniki", text="tresc fragmentu", word_count=2)
        result = ri.embedding_input(chunk, "praca.pdf")
        assert result == "[praca.pdf > Wyniki] tresc fragmentu"

    def test_omits_section_when_none(self):
        chunk = ri.Chunk(ordinal=0, page=None, section=None, text="tresc", word_count=1)
        assert ri.embedding_input(chunk, "notes.txt") == "[notes.txt] tresc"


class TestParseDispatch:
    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            ri.parse(b"data", "exe")


class TestIngestDocument:
    async def test_happy_path_embeds_stores_and_marks_done(self, monkeypatch):
        import rag_embed
        import rag_store

        embed_mock = AsyncMock(return_value=[[0.1, 0.2]])
        insert_mock = AsyncMock()
        status_mock = AsyncMock()
        monkeypatch.setattr(rag_embed, "embed_texts", embed_mock)
        monkeypatch.setattr(rag_store, "insert_chunks", insert_mock)
        monkeypatch.setattr(rag_store, "set_document_status", status_mock)

        md = b"# Wstep\n\nJedno zdanie tresci."
        await ri.ingest_document("doc-1", "alice", "notes.md", "md", md)

        embed_mock.assert_awaited_once()
        insert_mock.assert_awaited_once()
        status_mock.assert_awaited_once_with("doc-1", "done", page_count=0)

    async def test_no_extractable_text_marks_error_without_embedding(self, monkeypatch):
        import rag_embed
        import rag_store

        embed_mock = AsyncMock()
        status_mock = AsyncMock()
        monkeypatch.setattr(rag_embed, "embed_texts", embed_mock)
        monkeypatch.setattr(rag_store, "set_document_status", status_mock)

        await ri.ingest_document("doc-1", "alice", "empty.txt", "txt", b"")

        embed_mock.assert_not_awaited()
        status_mock.assert_awaited_once_with("doc-1", "error", error="No extractable text found")

    async def test_embedding_failure_marks_error_and_reraises(self, monkeypatch):
        import rag_embed
        import rag_store

        monkeypatch.setattr(rag_embed, "embed_texts", AsyncMock(side_effect=RuntimeError("ollama down")))
        status_mock = AsyncMock()
        monkeypatch.setattr(rag_store, "set_document_status", status_mock)

        with pytest.raises(RuntimeError, match="ollama down"):
            await ri.ingest_document("doc-1", "alice", "notes.md", "md", b"# H\n\ntext")

        status_mock.assert_awaited_once_with("doc-1", "error", error="ollama down")
