from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.embedding_contract import resolve_embedding_model_name, resolve_normalize_embeddings
from src.legal_data_processing import build_citation_base, build_citation_full, normalize_doc_id
from src.legal_rag.chunking.metadata_builder import generate_chunk_id
from src.legal_rag.chunking.policy import build_chunking_policy
from src.legal_rag.chunking.provision_chunker import build_unit_windows

from .extractors import normalize_extracted_text
from .models import ExtractedDocument, ExtractedPage, IngestRequest, IngestedChunk

DEFAULT_MODEL = resolve_embedding_model_name()


def count_text_tokens(text: str, **kwargs: object) -> int:
    try:
        from src.embeddings import count_text_tokens as embedding_token_counter

        return int(embedding_token_counter(text, **kwargs))
    except Exception:  # pragma: no cover - fallback for lightweight test/runtime environments
        return len(re.findall(r"\S+", str(text or ""), flags=re.UNICODE))


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
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}|[IVXLCDM]+|[A-Z])(?:[.)]|)\s+\S",
    re.IGNORECASE,
)
SECTION_HEADING_PATTERN = re.compile(
    r"^(?:PHAN|PH\u1ea6N|CHUONG|CH\u01af\u01a0NG|MUC|M\u1ee4C|TIEU\s*MUC|TI\u1ec2U\s*M\u1ee4C|SECTION|PART|CHAPTER)\b",
    re.IGNORECASE,
)


@dataclass
class _TextUnit:
    text: str
    page_number: int
    start: int
    end: int
    token_len: int


@dataclass
class _SectionPlan:
    section_id: str
    title: str
    path: str
    strategy: str
    confidence: float
    units: List[_TextUnit]
    page_start: int
    page_end: int
    char_start: int
    char_end: int


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
        self.last_chunk_artifacts: Dict[str, Any] = {}

    def _encoder_profile(self) -> str:
        normalize_flag = "true" if resolve_normalize_embeddings() else "false"
        slug = re.sub(r"[^a-z0-9]+", "_", self.encoder_model_name.lower()).strip("_")
        return f"{slug}|norm={normalize_flag}"

    def _count_tokens(self, text: str) -> int:
        return int(
            count_text_tokens(
                str(text or ""),
                model_name=self.encoder_model_name,
                text_type="passage",
                add_prefix=True,
            )
        )

    @staticmethod
    def _estimate_token_len(text: str) -> int:
        return len(re.findall(r"\S+", str(text or ""), flags=re.UNICODE))

    @staticmethod
    def _clip_label(text: str, max_chars: int = 120) -> str:
        compact = normalize_extracted_text(text).replace("\n", " ").strip()
        if len(compact) <= max_chars:
            return compact
        clipped = compact[: max_chars + 1]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped.rstrip(" -_:,./")

    @staticmethod
    def _source_blocks_from_text(text: str) -> List[str]:
        normalized = str(text or "").replace("\r\n", "\n")
        blocks = [normalize_extracted_text(item) for item in re.split(r"\n{2,}", normalized) if normalize_extracted_text(item)]
        if blocks:
            return blocks
        lines = [normalize_extracted_text(line) for line in normalized.splitlines() if normalize_extracted_text(line)]
        if len(lines) >= 2:
            return lines
        compact = normalize_extracted_text(normalized)
        return [compact] if compact else []

    def _build_block_units(self, extracted: ExtractedDocument) -> List[_TextUnit]:
        blocks: List[_TextUnit] = []
        if extracted.pages:
            for page in extracted.pages:
                blocks.extend(self._page_blocks(page))
            return blocks

        cursor = 0
        for block in self._source_blocks_from_text(extracted.normalized_text):
            start = cursor
            end = start + len(block)
            blocks.append(_TextUnit(block, 1, start, end, self._count_tokens(block)))
            cursor = end + 2
        return blocks

    def _page_blocks(self, page: ExtractedPage) -> List[_TextUnit]:
        blocks: List[_TextUnit] = []
        raw_blocks = page.blocks or self._source_blocks_from_text(page.text)
        if not raw_blocks:
            return blocks

        cursor = int(page.start_char or 0)
        for raw_block in raw_blocks:
            block = normalize_extracted_text(raw_block)
            if not block:
                continue
            start = cursor
            end = start + len(block)
            blocks.append(_TextUnit(block, page.page_number, start, end, self._count_tokens(block)))
            cursor = end + 2
        return blocks

    def _split_large_text(self, block: _TextUnit) -> List[_TextUnit]:
        protected = normalize_extracted_text(block.text)
        if not protected:
            return []
        pieces: List[_TextUnit] = []
        for match in SENTENCE_PATTERN.finditer(protected):
            piece = normalize_extracted_text(match.group(0))
            if not piece:
                continue
            piece_start = block.start + match.start()
            piece_end = piece_start + len(piece)
            token_len = self._count_tokens(piece)
            if token_len <= self.hard_cap_tokens:
                pieces.append(_TextUnit(piece, block.page_number, piece_start, piece_end, token_len))
                continue

            words = re.findall(r"\S+\s*", piece)
            cursor = 0
            local_start = 0
            while cursor < len(words):
                window = ""
                best_end = cursor
                for idx in range(cursor, len(words)):
                    candidate = f"{window}{words[idx]}"
                    candidate_text = candidate.strip()
                    if self._count_tokens(candidate_text) > self.hard_cap_tokens and window:
                        break
                    window = candidate
                    best_end = idx
                    if self._count_tokens(candidate_text) >= self.target_chunk_tokens:
                        break
                compact = normalize_extracted_text(window)
                if not compact:
                    break
                local_piece_start = piece_start + local_start
                local_piece_end = local_piece_start + len(compact)
                pieces.append(
                    _TextUnit(
                        compact,
                        block.page_number,
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
        for block in self._build_block_units(extracted):
            if block.token_len <= self.hard_cap_tokens:
                units.append(block)
            else:
                units.extend(self._split_large_text(block))
        return units

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        compact = normalize_extracted_text(text).replace("\n", " ").strip()
        if not compact:
            return False
        words = compact.split()
        if len(words) > 18:
            return False
        if NUMBERED_HEADING_PATTERN.match(compact):
            return True
        if SECTION_HEADING_PATTERN.match(compact):
            return True
        if compact.endswith((".", ";", ":", ",")) and len(words) > 10:
            return False
        uppercase_ratio = 0.0
        letters = [char for char in compact if char.isalpha()]
        if letters:
            uppercase_ratio = sum(1 for char in letters if char == char.upper()) / len(letters)
        return uppercase_ratio >= 0.7 and len(words) <= 12

    def _profile_document(self, blocks: List[_TextUnit], extracted: ExtractedDocument) -> tuple[str, float]:
        if not blocks:
            return "flat_text_or_ocr", 0.35

        heading_count = sum(1 for block in blocks if self._looks_like_heading(block.text))
        numbered_heading_count = sum(
            1 for block in blocks if NUMBERED_HEADING_PATTERN.match(normalize_extracted_text(block.text).replace("\n", " ").strip())
        )
        page_count = len(extracted.pages) if extracted.pages else 1
        short_block_ratio = sum(1 for block in blocks if len(block.text.split()) <= 14) / max(len(blocks), 1)

        if heading_count >= 3 or numbered_heading_count >= 2:
            confidence = min(0.95, 0.58 + (0.08 * heading_count) + (0.05 * numbered_heading_count))
            return "structured_generic", round(confidence, 2)
        if heading_count >= 1 or (len(blocks) >= 4 and short_block_ratio < 0.7):
            confidence = min(0.82, 0.46 + (0.06 * heading_count) + (0.02 * min(page_count, 4)))
            return "semi_structured", round(confidence, 2)
        confidence = 0.35 if short_block_ratio >= 0.7 else 0.42
        return "flat_text_or_ocr", round(confidence, 2)

    def _section_seed_title(self, block: _TextUnit, index: int, strategy: str) -> str:
        compact = self._clip_label(block.text, max_chars=96)
        if compact and self._looks_like_heading(block.text):
            return compact
        if strategy == "flat_text_or_ocr":
            return f"Doan {index + 1}"
        return f"Muc {index + 1}"

    @staticmethod
    def _section_path(title: str, index: int) -> str:
        compact = normalize_extracted_text(title).replace("\n", " ").strip()
        return compact or f"Section {index + 1}"

    def _make_section(
        self,
        *,
        request: IngestRequest,
        index: int,
        title: str,
        strategy: str,
        confidence: float,
        units: List[_TextUnit],
    ) -> _SectionPlan:
        char_start = min(unit.start for unit in units)
        char_end = max(unit.end for unit in units)
        page_start = min(unit.page_number for unit in units)
        page_end = max(unit.page_number for unit in units)
        stable_key = f"{request.doc_id}|{index}|{title}|{char_start}|{char_end}"
        section_id = f"sec_{hashlib.md5(stable_key.encode('utf-8')).hexdigest()[:16]}"
        return _SectionPlan(
            section_id=section_id,
            title=title,
            path=self._section_path(title, index),
            strategy=strategy,
            confidence=confidence,
            units=units,
            page_start=page_start,
            page_end=page_end,
            char_start=char_start,
            char_end=char_end,
        )

    def _build_structured_sections(
        self,
        *,
        request: IngestRequest,
        blocks: List[_TextUnit],
        strategy: str,
        confidence: float,
    ) -> List[_SectionPlan]:
        sections: List[_SectionPlan] = []
        current_units: List[_TextUnit] = []
        current_title: Optional[str] = None

        def finalize_current() -> None:
            if not current_units:
                return
            title = current_title or self._section_seed_title(current_units[0], len(sections), strategy)
            sections.append(
                self._make_section(
                    request=request,
                    index=len(sections),
                    title=title,
                    strategy=strategy,
                    confidence=confidence,
                    units=list(current_units),
                )
            )

        for block in blocks:
            block_units = [block] if block.token_len <= self.hard_cap_tokens else self._split_large_text(block)
            if not block_units:
                continue
            if self._looks_like_heading(block.text):
                finalize_current()
                current_units = list(block_units)
                current_title = self._section_seed_title(block, len(sections), strategy)
                continue
            if not current_units:
                current_title = self._section_seed_title(block, len(sections), strategy)
            current_units.extend(block_units)
        finalize_current()
        return sections

    def _build_flat_sections(
        self,
        *,
        request: IngestRequest,
        units: List[_TextUnit],
        strategy: str,
        confidence: float,
    ) -> List[_SectionPlan]:
        sections: List[_SectionPlan] = []
        current: List[_TextUnit] = []
        current_tokens = 0
        section_target = max(self.target_chunk_tokens * 2, self.target_chunk_tokens + self.overlap_tokens)

        for unit in units:
            if current and current_tokens >= section_target:
                sections.append(
                    self._make_section(
                        request=request,
                        index=len(sections),
                        title=f"Doan {len(sections) + 1}",
                        strategy=strategy,
                        confidence=confidence,
                        units=list(current),
                    )
                )
                current = []
                current_tokens = 0
            current.append(unit)
            current_tokens += int(unit.token_len)

        if current:
            sections.append(
                self._make_section(
                    request=request,
                    index=len(sections),
                    title=f"Doan {len(sections) + 1}",
                    strategy=strategy,
                    confidence=confidence,
                    units=list(current),
                )
            )
        return sections

    def _build_sections(self, extracted: ExtractedDocument, request: IngestRequest) -> List[_SectionPlan]:
        blocks = self._build_block_units(extracted)
        strategy, confidence = self._profile_document(blocks, extracted)
        if strategy in {"structured_generic", "semi_structured"}:
            sections = self._build_structured_sections(
                request=request,
                blocks=blocks,
                strategy=strategy,
                confidence=confidence,
            )
        else:
            sections = self._build_flat_sections(
                request=request,
                units=self._build_units(extracted),
                strategy=strategy,
                confidence=confidence,
            )
        if sections:
            return sections
        fallback_units = self._build_units(extracted)
        if not fallback_units and extracted.normalized_text:
            synthetic = _TextUnit(
                text=normalize_extracted_text(extracted.normalized_text),
                page_number=1,
                start=0,
                end=len(normalize_extracted_text(extracted.normalized_text)),
                token_len=self._count_tokens(extracted.normalized_text),
            )
            fallback_units = self._split_large_text(synthetic) if synthetic.token_len > self.hard_cap_tokens else [synthetic]
        if not fallback_units:
            return []
        return self._build_flat_sections(
            request=request,
            units=fallback_units,
            strategy="flat_text_or_ocr",
            confidence=0.35,
        )

    @staticmethod
    def _page_label(page_start: int, page_end: int) -> str:
        return f"Trang {page_start}" if page_start == page_end else f"Trang {page_start}-{page_end}"

    def _legal_path_for_window(self, section: _SectionPlan, window: List[_TextUnit]) -> str:
        page_start = min(unit.page_number for unit in window)
        page_end = max(unit.page_number for unit in window)
        page_label = self._page_label(page_start, page_end)
        if section.path and section.path != page_label:
            return f"{section.path} | {page_label}"
        return section.path or page_label

    def _build_chunk_artifacts(self, *, strategy: str, confidence: float, sections: List[_SectionPlan], chunk_map: Dict[str, List[str]]) -> Dict[str, Any]:
        return {
            "structure_strategy": strategy,
            "structure_confidence": confidence,
            "section_count": len(sections),
            "sections": [
                {
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "section_path": section.path,
                    "structure_strategy": section.strategy,
                    "structure_confidence": section.confidence,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "char_start": section.char_start,
                    "char_end": section.char_end,
                    "chunk_ids": list(chunk_map.get(section.section_id, [])),
                }
                for section in sections
            ],
        }

    def chunk(self, extracted: ExtractedDocument, request: IngestRequest) -> List[IngestedChunk]:
        sections = self._build_sections(extracted, request)
        if not sections:
            self.last_chunk_artifacts = {}
            return []

        strategy = sections[0].strategy
        confidence = sections[0].confidence
        citation_base = build_citation_base("", request.doc_id, request.filename)
        chunk_specs: List[tuple[_SectionPlan, List[_TextUnit]]] = []
        for section in sections:
            windows = build_unit_windows(
                section.units,
                token_getter=lambda unit: int(unit.token_len),
                target_chunk_tokens=self.target_chunk_tokens,
                hard_cap_tokens=self.hard_cap_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            for window in windows:
                if window:
                    chunk_specs.append((section, window))

        chunks: List[IngestedChunk] = []
        total_chunks = len(chunk_specs)
        chunk_map: Dict[str, List[str]] = {}
        for index, (section, window) in enumerate(chunk_specs):
            content = "\n\n".join(unit.text for unit in window).strip()
            if not content:
                continue
            page_start = min(unit.page_number for unit in window)
            page_end = max(unit.page_number for unit in window)
            page_label = self._page_label(page_start, page_end)
            legal_path = self._legal_path_for_window(section, window)
            citation_short = section.title or page_label
            citation_full = build_citation_full(citation_base, legal_path)
            chunk_id = generate_chunk_id(
                doc_id=request.doc_id,
                dieu_so="",
                chunk_index=index,
                text=content,
                overflow_index=1,
            )
            chunk_map.setdefault(section.section_id, []).append(chunk_id)
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
                    segment_level="semantic_window" if strategy == "flat_text_or_ocr" else "section_window",
                    citation_base=citation_base,
                    citation_short=citation_short,
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
                    hierarchy={
                        "page_number": page_start,
                        "page_start": page_start,
                        "page_end": page_end,
                        "parent_section_id": section.section_id,
                        "section_title": section.title,
                    },
                    parent_section_id=section.section_id,
                    section_title=section.title,
                    section_path=section.path,
                    structure_strategy=section.strategy,
                    structure_confidence=section.confidence,
                    page_start=page_start,
                    page_end=page_end,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    filename=request.filename,
                    source_type="user_upload",
                    page_number=page_start,
                    upload_time=request.upload_time,
                    ingest_mode=request.content_mode,
                    parser_used=extracted.parser_used,
                    ocr_provider=extracted.ocr_provider,
                )
            )
        self.last_chunk_artifacts = self._build_chunk_artifacts(
            strategy=strategy,
            confidence=confidence,
            sections=sections,
            chunk_map=chunk_map,
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
        self.last_chunk_artifacts: Dict[str, Any] = {}

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
            chunks = self.generic_chunker.chunk(extracted, request)
            self.last_chunk_artifacts = dict(getattr(self.generic_chunker, "last_chunk_artifacts", {}) or {})
            return chunks

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
        self.last_chunk_artifacts = {}
        return normalized_chunks
