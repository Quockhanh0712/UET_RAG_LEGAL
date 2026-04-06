from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.document_ingestion.chunking import ChunkRouter, GenericDocumentChunker
from src.document_ingestion.models import ExtractedDocument, ExtractedPage, IngestRequest, IngestedChunk


def _word_token_counter(text: str, **_: object) -> int:
    return len(str(text or "").split())


def _load_legal_chunker_with_stubbed_embeddings(monkeypatch):
    fake_embeddings = SimpleNamespace(
        DEFAULT_MODEL="truro7/vn-law-embedding",
        count_text_tokens=_word_token_counter,
    )
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_embeddings)
    module = importlib.import_module("src.legal_chunker_v2")
    importlib.reload(module)
    return module.LegalChunkerV2


def _build_request() -> IngestRequest:
    return IngestRequest(
        doc_id="user_test_doc",
        user_id="user-1",
        session_id="session-1",
        filename="sample.txt",
        file_type="txt",
        file_size=123,
        saved_file_path=Path(__file__),
        artifact_dir=Path(__file__).parent,
        upload_time="2026-04-04T10:00:00",
        content_mode="text",
        ocr_provider="default",
    )


def test_chunk_router_routes_legal_text_into_legal_chunker(monkeypatch):
    router = ChunkRouter(
        legal_chunker=SimpleNamespace(
            chunk_text=lambda *args, **kwargs: [],
            _build_encoder_profile=lambda: "test-encoder",
        ),
        generic_chunker=SimpleNamespace(chunk=lambda *args, **kwargs: []),
    )
    request = _build_request()
    extracted = ExtractedDocument(
        file_path=request.saved_file_path,
        file_type="txt",
        parser_used="plain-text",
        text="Chuong I\nDieu 1. Noi dung.\nDieu 2. Noi dung khac.",
        normalized_text="Chuong I\nDieu 1. Noi dung.\nDieu 2. Noi dung khac.",
        pages=[ExtractedPage(page_number=1, text="Dieu 1. Noi dung.\nDieu 2. Noi dung khac.", start_char=0, end_char=42)],
    )

    called = {"legal": 0, "generic": 0}

    def _fake_legal_chunk_text(text, metadata):
        called["legal"] += 1
        return [
            SimpleNamespace(
                to_dict=lambda: {
                    "chunk_id": "legal_chunk_1",
                    "content": "Dieu 1. Noi dung.",
                    "doc_id": request.doc_id,
                    "chunk_type": "ClauseChunk",
                    "segment_level": "clause",
                    "char_start": 0,
                    "char_end": 18,
                    "path": "Dieu 1",
                    "citation_base": metadata["citation_base"],
                    "citation_full": f"{metadata['citation_base']} | Dieu 1",
                }
            )
        ]

    def _fake_generic_chunk(*args, **kwargs):
        called["generic"] += 1
        return []

    monkeypatch.setattr(router.legal_chunker, "chunk_text", _fake_legal_chunk_text)
    monkeypatch.setattr(router.generic_chunker, "chunk", _fake_generic_chunk)

    chunks = router.chunk(extracted, request)

    assert called["legal"] == 1
    assert called["generic"] == 0
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "ClauseChunk"
    assert chunks[0].user_id == request.user_id
    assert chunks[0].page_number == 1


def test_chunk_router_routes_generic_text_into_generic_chunker(monkeypatch):
    router = ChunkRouter(
        legal_chunker=SimpleNamespace(
            chunk_text=lambda *args, **kwargs: [],
            _build_encoder_profile=lambda: "test-encoder",
        ),
        generic_chunker=SimpleNamespace(chunk=lambda *args, **kwargs: []),
    )
    request = _build_request()
    extracted = ExtractedDocument(
        file_path=request.saved_file_path,
        file_type="txt",
        parser_used="plain-text",
        text="Tai lieu thong thuong khong co cau truc dieu khoan.",
        normalized_text="Tai lieu thong thuong khong co cau truc dieu khoan.",
        pages=[ExtractedPage(page_number=1, text="Tai lieu thong thuong khong co cau truc dieu khoan.", start_char=0, end_char=50)],
    )

    expected_chunk = IngestedChunk(
        chunk_id="generic_chunk_1",
        content="Tai lieu thong thuong",
        doc_id=request.doc_id,
    )
    called = {"legal": 0, "generic": 0}

    monkeypatch.setattr(
        router.legal_chunker,
        "chunk_text",
        lambda *args, **kwargs: called.__setitem__("legal", called["legal"] + 1),
    )

    def _fake_generic_chunk(*args, **kwargs):
        called["generic"] += 1
        return [expected_chunk]

    monkeypatch.setattr(router.generic_chunker, "chunk", _fake_generic_chunk)

    chunks = router.chunk(extracted, request)

    assert called["generic"] == 1
    assert called["legal"] == 0
    assert chunks == [expected_chunk]


def test_chunk_router_prefers_legal_chunker_for_article_clause_structure(monkeypatch):
    router = ChunkRouter(
        legal_chunker=SimpleNamespace(
            chunk_text=lambda *args, **kwargs: [],
            _build_encoder_profile=lambda: "test-encoder",
        ),
        generic_chunker=SimpleNamespace(chunk=lambda *args, **kwargs: []),
    )
    request = _build_request()
    extracted = ExtractedDocument(
        file_path=request.saved_file_path,
        file_type="txt",
        parser_used="plain-text",
        text="Dieu 7. Nguyen tac ap dung\n1. Khoan mot.\n2. Khoan hai.",
        normalized_text="Dieu 7. Nguyen tac ap dung\n1. Khoan mot.\n2. Khoan hai.",
        pages=[
            ExtractedPage(
                page_number=1,
                text="Dieu 7. Nguyen tac ap dung\n1. Khoan mot.\n2. Khoan hai.",
                start_char=0,
                end_char=58,
            )
        ],
    )

    called = {"legal": 0, "generic": 0}
    monkeypatch.setattr(
        router.legal_chunker,
        "chunk_text",
        lambda *args, **kwargs: called.__setitem__("legal", called["legal"] + 1) or [],
    )
    monkeypatch.setattr(
        router.generic_chunker,
        "chunk",
        lambda *args, **kwargs: called.__setitem__("generic", called["generic"] + 1) or [],
    )

    router.chunk(extracted, request)

    assert called["legal"] == 1
    assert called["generic"] == 0


def test_chunk_router_preserves_legal_chunk_metadata_contract(monkeypatch):
    LegalChunkerV2 = _load_legal_chunker_with_stubbed_embeddings(monkeypatch)

    router = ChunkRouter(legal_chunker=LegalChunkerV2(), generic_chunker=SimpleNamespace(chunk=lambda *args, **kwargs: []))
    request = _build_request()
    request.filename = "luat-giao-thong.txt"
    extracted = ExtractedDocument(
        file_path=request.saved_file_path,
        file_type="txt",
        parser_used="plain-text",
        text="Dieu 1. Pham vi dieu chinh.\n1. Quy dinh thu nhat.\n2. Quy dinh thu hai.",
        normalized_text="Dieu 1. Pham vi dieu chinh.\n1. Quy dinh thu nhat.\n2. Quy dinh thu hai.",
        pages=[
            ExtractedPage(
                page_number=1,
                text="Dieu 1. Pham vi dieu chinh.\n1. Quy dinh thu nhat.\n2. Quy dinh thu hai.",
                start_char=0,
                end_char=73,
            )
        ],
    )

    chunks = router.chunk(extracted, request)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "ProvisionChunk"
    assert chunks[0].segment_level == "article"
    assert chunks[0].citation_full
    assert chunks[0].provision_id
    assert chunks[0].overflow_index == 1
    assert chunks[0].overflow_total == 1
    assert chunks[0].encoder_profile.startswith("truro7_vn_law_embedding|norm=")


def test_generic_document_chunker_enforces_cap_overlap_and_stable_ids(monkeypatch):
    monkeypatch.setattr("src.document_ingestion.chunking.count_text_tokens", _word_token_counter)

    chunker = GenericDocumentChunker(
        target_chunk_tokens=20,
        hard_cap_tokens=24,
        overlap_tokens=4,
    )
    request = _build_request()
    blocks = [
        "alpha1 mot hai ba bon nam sau bay tam chin",
        "alpha2 mot hai ba bon nam sau bay tam chin",
        "alpha3 mot hai ba bon nam sau bay tam chin",
        "alpha4 mot hai ba bon nam sau bay tam chin",
    ]
    extracted = ExtractedDocument(
        file_path=request.saved_file_path,
        file_type="txt",
        parser_used="plain-text",
        text="\n\n".join(blocks),
        normalized_text="\n\n".join(blocks),
        pages=[
            ExtractedPage(
                page_number=1,
                text="\n\n".join(blocks[:2]),
                blocks=blocks[:2],
                start_char=0,
                end_char=len("\n\n".join(blocks[:2])),
            ),
            ExtractedPage(
                page_number=2,
                text="\n\n".join(blocks[2:]),
                blocks=blocks[2:],
                start_char=len("\n\n".join(blocks[:2])) + 2,
                end_char=len("\n\n".join(blocks)) + 2,
            ),
        ],
    )

    first_run = chunker.chunk(extracted, request)
    second_run = chunker.chunk(extracted, request)

    assert len(first_run) >= 2
    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert all(chunk.token_len_model <= 24 for chunk in first_run)
    assert all(chunk.citation_full for chunk in first_run)
    assert "alpha2" in first_run[0].content
    assert "alpha2" in first_run[1].content
