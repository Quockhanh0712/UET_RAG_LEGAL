from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.embedding_contract import resolve_embedding_model_name, resolve_normalize_embeddings

DEFAULT_MODEL = resolve_embedding_model_name()


def count_text_tokens(text: str, **kwargs: object) -> int:
    try:
        from src.embeddings import count_text_tokens as embedding_token_counter

        return int(embedding_token_counter(text, **kwargs))
    except Exception:  # pragma: no cover - fallback for lightweight test/runtime environments
        return len(re.findall(r"\S+", str(text or ""), flags=re.UNICODE))

from src.legal_data_processing import build_citation_base, build_citation_full, normalize_doc_id
from src.legal_rag.chunking.metadata_builder import generate_chunk_id
from src.legal_rag.chunking.policy import build_chunking_policy
from src.legal_rag.chunking.provision_chunker import build_unit_windows

from .extractors import normalize_extracted_text
from .models import ExtractedDocument, IngestRequest, IngestedChunk


LEGAL_STRUCTURE_PATTERN = re.compile(
    r"(^|\n)(?:Ch\u01b0\u01a1ng|Chuong)\s+(?:[IVXLCDM]+|\d+)\b|(^|\n)(?:\u0110i\u1ec1u|Dieu)\s+\d+[A-Za-z0-9\-]*\s*[.:]",
    re.IGNORECASE,
)
LEGAL_ARTICLE_PATTERN = re.compile(
    r"(^|\n)\s*(?:\u0110i\u1ec1u|Dieu)\s+\d+[A-Za-z0-9\-]*\s*[.:]",
    re.IGNORECASE,
)
LEGAL_CHAPTER_PATTERN = re.compile(
    r"(^|\n)\s*(?:Ch\u01b0\u01a1ng|Chuong)\s+(?:[IVXLCDM]+|\d+)\b",
    re.IGNORECASE,
)
LEGAL_CLAUSE_PATTERN = re.compile(
    r"(^|\n)\s*(?:(?:Kho\u1ea3n|Khoan)\s+)?[1-9]\d{0,2}\.\s+",
    re.IGNORECASE,
)
LEGAL_FILENAME_HINT_PATTERN = re.compile(
    r"\b(luat|bo_luat|nghi_dinh|thong_tu|quyet_dinh|phap_lenh|nghi_quyet)\b",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r".+?(?:\n+|(?<=[.;:!?])\s+|$)", re.DOTALL)


@dataclass
class _TextUnit:
    text: str
    page_number: int
    start: int
    end: int
    token_len: int


class GenericDocumentChunker:
    def __init__(
        self,
        *,
        encoder_model_name: Optional[str] = None,
        target_chunk_tokens: int = 420,
        hard_cap_tokens: int = 448,
        overlap_tokens: int = 64,
    ):
        self.encoder_model_name = str(encoder_model_name or DEFAULT_MODEL)
        self.policy = build_chunking_policy(
            min_chars=1200,
            max_chars=2800,
            overlap_chars=120,
            target_chunk_tokens=target_chunk_tokens,
            hard_cap_tokens=hard_cap_tokens,
            overlap_tokens=overlap_tokens,
        )
        self.target_chunk_tokens = self.policy.target_chunk_tokens
        self.hard_cap_tokens = self.policy.hard_cap_tokens
        self.overlap_tokens = self.policy.overlap_tokens

    def _encoder_profile(self) -> str:
        normalize_flag = "true" if resolve_normalize_embeddings() else "false"
        slug = re.sub(r"[^a-z0-9]+", "_", self.encoder_model_name.lower()).strip("_")
        return f"{slug}|norm={normalize_flag}"

    def _count_tokens(self, text: str) -> int:
        return int(count_text_tokens(str(text or ""), model_name=self.encoder_model_name, text_type="passage", add_prefix=True))

    @staticmethod
    def _estimate_token_len(text: str) -> int:
        return len(re.findall(r"\S+", str(text or ""), flags=re.UNICODE))

    def _split_large_text(self, text: str, page_number: int, start: int) -> List[_TextUnit]:
        protected = normalize_extracted_text(text)
        pieces: List[_TextUnit] = []
        for match in SENTENCE_PATTERN.finditer(protected):
            piece = normalize_extracted_text(match.group(0))
            if not piece:
                continue
            piece_start = start + match.start()
            piece_end = piece_start + len(piece)
            token_len = self._count_tokens(piece)
            if token_len <= self.hard_cap_tokens:
                pieces.append(_TextUnit(piece, page_number, piece_start, piece_end, token_len))
                continue

            words = re.findall(r"\S+\s*", piece)
            cursor = 0
            local_start = 0
            while cursor < len(words):
                window = ""
                best_end = cursor
                for idx in range(cursor, len(words)):
                    candidate = f"{window}{words[idx]}"
                    if self._count_tokens(candidate.strip()) > self.hard_cap_tokens and window:
                        break
                    window = candidate
                    best_end = idx
                    if self._count_tokens(window.strip()) >= self.target_chunk_tokens:
                        break
                compact = normalize_extracted_text(window)
                local_piece_start = piece_start + local_start
                local_piece_end = local_piece_start + len(compact)
                pieces.append(
                    _TextUnit(
                        compact,
                        page_number,
                        local_piece_start,
                        local_piece_end,
                        self._count_tokens(compact),
                    )
                )
                local_start += len(window)
                cursor = best_end + 1
        return pieces

    def _build_units(self, extracted: ExtractedDocument) -> List[_TextUnit]:
        units: List[_TextUnit] = []
        if extracted.pages:
            for page in extracted.pages:
                cursor = page.start_char
                blocks = page.blocks or [paragraph for paragraph in page.text.split("\n\n") if paragraph]
                for block in blocks:
                    normalized_block = normalize_extracted_text(block)
                    if not normalized_block:
                        continue
                    block_start = cursor
                    block_end = block_start + len(normalized_block)
                    token_len = self._count_tokens(normalized_block)
                    if token_len <= self.hard_cap_tokens:
                        units.append(_TextUnit(normalized_block, page.page_number, block_start, block_end, token_len))
                    else:
                        units.extend(self._split_large_text(normalized_block, page.page_number, block_start))
                    cursor = block_end + 2
        else:
            normalized = normalize_extracted_text(extracted.normalized_text)
            if normalized:
                units.extend(self._split_large_text(normalized, 1, 0))
        return units

    def chunk(self, extracted: ExtractedDocument, request: IngestRequest) -> List[IngestedChunk]:
        units = self._build_units(extracted)
        if not units and extracted.normalized_text:
            units = self._split_large_text(extracted.normalized_text, 1, 0)
        if not units:
            return []

        windows = build_unit_windows(
            units,
            token_getter=lambda unit: int(unit.token_len),
            target_chunk_tokens=self.target_chunk_tokens,
            hard_cap_tokens=self.hard_cap_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        citation_base = build_citation_base("", request.doc_id, request.filename)
        chunks: List[IngestedChunk] = []
        total_chunks = len(windows)
        for index, window in enumerate(windows):
            content = "\n\n".join(unit.text for unit in window).strip()
            if not content:
                continue
            first_page = window[0].page_number
            last_page = window[-1].page_number
            page_label = f"Trang {first_page}" if first_page == last_page else f"Trang {first_page}-{last_page}"
            citation_full = build_citation_full(citation_base, page_label)
            chunk_id = generate_chunk_id(
                doc_id=request.doc_id,
                dieu_so="",
                chunk_index=index,
                text=content,
                overflow_index=1,
            )
            legal_path = page_label
            chunks.append(
                IngestedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    doc_id=request.doc_id,
                    so_ky_hieu=request.doc_id,
                    van_ban_id=request.doc_id,
                    ten_van_ban=request.filename,
                    loai_van_ban=request.file_type.upper(),
                    doc_type="USER_UPLOAD",
                    doc_type_rank=999,
                    law_family=request.filename,
                    status_label="active",
                    path=legal_path,
                    legal_path=legal_path,
                    chunk_type="GenericDocumentChunk",
                    segment_level="page_block_window",
                    citation_base=citation_base,
                    citation_short=page_label,
                    citation_full=citation_full,
                    chunk_index=index,
                    total_chunks=total_chunks,
                    char_start=window[0].start,
                    char_end=window[-1].end,
                    token_len_est=self._estimate_token_len(content),
                    token_len_model=self._count_tokens(content),
                    overflow_index=1,
                    overflow_total=1,
                    window_char_start=window[0].start,
                    window_char_end=window[-1].end,
                    encoder_profile=self._encoder_profile(),
                    hierarchy={"page_number": first_page},
                    user_id=request.user_id,
                    session_id=request.session_id,
                    filename=request.filename,
                    source_type="user_upload",
                    page_number=first_page,
                    upload_time=request.upload_time,
                    ingest_mode=request.content_mode,
                    parser_used=extracted.parser_used,
                    ocr_provider=extracted.ocr_provider,
                )
            )
        return chunks


class ChunkRouter:
    def __init__(self, legal_chunker=None, generic_chunker=None):
        if legal_chunker is None:
            from src.legal_chunker_v2 import LegalChunkerV2

            legal_chunker = LegalChunkerV2(
                target_chunk_tokens=420,
                hard_cap_tokens=448,
                overlap_tokens=64,
            )
        self.legal_chunker = legal_chunker
        self.generic_chunker = generic_chunker or GenericDocumentChunker(
            target_chunk_tokens=420,
            hard_cap_tokens=448,
            overlap_tokens=64,
        )

    @staticmethod
    def _filename_hint(extracted_or_text: ExtractedDocument | str) -> str:
        if isinstance(extracted_or_text, ExtractedDocument):
            file_stem = extracted_or_text.file_path.stem if extracted_or_text.file_path else ""
            metadata_name = str(extracted_or_text.metadata.get("filename") or "")
            return normalize_extracted_text(f"{file_stem} {metadata_name}").lower().replace(" ", "_")
        return ""

    def route(self, extracted_or_text: ExtractedDocument | str) -> str:
        if isinstance(extracted_or_text, ExtractedDocument):
            normalized = str(extracted_or_text.normalized_text or extracted_or_text.text or "")
        else:
            normalized = str(extracted_or_text or "")

        article_matches = len(LEGAL_ARTICLE_PATTERN.findall(normalized))
        chapter_matches = len(LEGAL_CHAPTER_PATTERN.findall(normalized))
        clause_matches = len(LEGAL_CLAUSE_PATTERN.findall(normalized))
        filename_hint = self._filename_hint(extracted_or_text)

        is_legal = (
            (article_matches >= 2)
            or (article_matches >= 1 and chapter_matches >= 1)
            or (article_matches >= 1 and clause_matches >= 1)
            or (
                article_matches >= 1
                and bool(filename_hint)
                and bool(LEGAL_FILENAME_HINT_PATTERN.search(filename_hint))
            )
            or len(LEGAL_STRUCTURE_PATTERN.findall(normalized)) >= 2
        )
        return "legal" if is_legal else "generic"

    def chunk(self, extracted: ExtractedDocument, request: IngestRequest) -> List[IngestedChunk]:
        mode = self.route(extracted)
        if mode == "generic":
            return self.generic_chunker.chunk(extracted, request)

        metadata: Dict[str, Any] = {
            "doc_id": normalize_doc_id(request.doc_id),
            "so_ky_hieu": normalize_doc_id(request.doc_id),
            "van_ban_id": normalize_doc_id(request.doc_id),
            "ten_van_ban": request.filename,
            "loai_van_ban": request.file_type.upper(),
            "doc_type": "USER_UPLOAD",
            "doc_type_rank": 999,
            "citation_base": build_citation_base("", request.doc_id, request.filename),
            "encoder_profile": self.legal_chunker._build_encoder_profile(),
        }
        legal_chunks = self.legal_chunker.chunk_text(extracted.normalized_text, metadata=metadata)
        normalized_chunks: List[IngestedChunk] = []
        total_chunks = len(legal_chunks)
        for index, chunk in enumerate(legal_chunks):
            payload = chunk.to_dict()
            payload.pop("text", None)
            payload.update(
                {
                    "doc_type": "USER_UPLOAD",
                    "doc_type_rank": 999,
                    "ten_van_ban": request.filename,
                    "loai_van_ban": request.file_type.upper(),
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "page_number": extracted.resolve_page_number(
                        int(payload.get("char_start", 0) or 0),
                        int(payload.get("char_end", 0) or 0),
                    ),
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "filename": request.filename,
                    "source_type": "user_upload",
                    "upload_time": request.upload_time,
                    "ingest_mode": request.content_mode,
                    "parser_used": extracted.parser_used,
                    "ocr_provider": extracted.ocr_provider,
                }
            )
            normalized_chunks.append(IngestedChunk.from_payload(payload))
        return normalized_chunks
