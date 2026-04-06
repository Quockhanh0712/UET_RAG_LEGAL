from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.document_ingestion.extractors import DocxTextExtractor, PdfTextExtractor
from src.document_ingestion.ocr import DefaultOcrProvider
from src.document_ingestion.storage import ArtifactStore


fitz = pytest.importorskip("fitz")
docx = pytest.importorskip("docx")


def _create_pdf(path, text: str | None = None) -> None:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_pdf_text_extractor_extracts_text_layer_without_ocr(tmp_path):
    pdf_path = tmp_path / "digital.pdf"
    _create_pdf(pdf_path, "Dieu 1. Van ban nay co text layer ro rang.")

    extractor = PdfTextExtractor(min_text_chars=20)
    result = extractor.extract(pdf_path)

    assert result.parser_used == "pymupdf"
    assert result.needs_ocr is False
    assert "Dieu 1" in result.normalized_text
    assert result.page_count == 1


def test_pdf_text_extractor_flags_scanned_pdf_as_needs_ocr(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _create_pdf(pdf_path)

    extractor = PdfTextExtractor(min_text_chars=20)
    result = extractor.extract(pdf_path)

    assert result.parser_used == "pymupdf"
    assert result.needs_ocr is True
    assert result.text_chars == 0
    assert result.page_count == 1


def test_docx_text_extractor_preserves_paragraph_order(tmp_path):
    document = docx.Document()
    document.add_paragraph("Doan mot")
    document.add_paragraph("Doan hai")
    document.add_paragraph("Doan ba")
    docx_path = tmp_path / "sample.docx"
    document.save(docx_path)

    extractor = DocxTextExtractor()
    result = extractor.extract(docx_path)

    assert result.parser_used == "python-docx"
    assert result.normalized_text.split("\n\n") == ["Doan mot", "Doan hai", "Doan ba"]


def test_default_ocr_provider_returns_text_ready_for_chunking(tmp_path, monkeypatch):
    source_pdf = tmp_path / "input.pdf"
    _create_pdf(source_pdf)

    artifact_store = ArtifactStore(tmp_path / "artifacts")
    artifact_dir = artifact_store.document_dir("user-a", "session-a", "doc-a")
    provider = DefaultOcrProvider(artifact_store)

    monkeypatch.setattr("src.document_ingestion.ocr.shutil.which", lambda _: "ocrmypdf")

    def _fake_run(command, capture_output, text, check):
        output_pdf = command[-1]
        _create_pdf(output_pdf, "Noi dung OCR da du de chunk.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.document_ingestion.ocr.subprocess.run", _fake_run)

    result = provider.process(source_pdf, artifact_dir)

    assert result.ocr_provider == "default"
    assert result.parser_used == "pymupdf+ocrmypdf"
    assert result.needs_ocr is False
    assert "Noi dung OCR" in result.normalized_text
