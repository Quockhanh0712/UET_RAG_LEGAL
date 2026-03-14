"""
Legal document chunker v2.

Supports two input shapes:
1. Full document text (preferred): `toan_van_clean` / `toan_van`
2. Legacy article-level rows: `clean_text` / `noi_dung`

Retrieval chunking strategy is deterministic and legal-aware:
- primary boundary: Dieu
- overflow split only inside the same Dieu
- Khoan/Diem are preserved as metadata only
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.embeddings import DEFAULT_MODEL, count_text_tokens
from src.legal_data_processing import (
    build_citation_base,
    build_citation_full,
    normalize_document_row,
    normalize_doc_id,
    normalize_whitespace,
)

logger = logging.getLogger(__name__)
NUMERIC_DOT_PLACEHOLDER = "\uFFF0"


@dataclass
class LegalChunk:
    chunk_id: str
    content: str
    doc_id: str
    so_ky_hieu: str
    van_ban_id: str
    ten_van_ban: str
    loai_van_ban: str
    doc_type: str = "OTHER"
    doc_type_rank: int = 999
    co_quan_ban_hanh: str = ""
    co_quan: str = ""
    chuc_danh: str = ""
    nguoi_ky: str = ""
    law_family: str = ""
    ngay_ban_hanh: Optional[str] = None
    ngay_co_hieu_luc: Optional[str] = None
    ngay_het_hieu_luc: Optional[str] = None
    is_active: bool = True
    status_label: str = "active"
    chuong: str = ""
    ten_chuong: str = ""
    dieu_so: str = ""
    tieu_de_dieu: str = ""
    khoan_so: str = ""
    diem_so: str = ""
    path: str = ""
    legal_path: str = ""
    chunk_type: str = "FallbackChunk"
    segment_level: str = "document"
    fallback_reason: str = ""
    provision_id: str = ""
    parent_provision_id: str = ""
    citation_base: str = ""
    citation_short: str = ""
    citation_full: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    char_start: int = 0
    char_end: int = 0
    token_len_est: int = 0
    token_len_model: int = 0
    overflow_index: int = 1
    overflow_total: int = 1
    window_char_start: int = 0
    window_char_end: int = 0
    clause_spans: List[str] = field(default_factory=list)
    encoder_profile: str = "default"
    hierarchy: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content

    @text.setter
    def text(self, value: str) -> None:
        self.content = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "text": self.content,  # compatibility alias
            "doc_id": self.doc_id,
            "so_ky_hieu": self.so_ky_hieu,
            "van_ban_id": self.van_ban_id,
            "ten_van_ban": self.ten_van_ban,
            "loai_van_ban": self.loai_van_ban,
            "doc_type": self.doc_type,
            "doc_type_rank": self.doc_type_rank,
            "co_quan_ban_hanh": self.co_quan_ban_hanh,
            "co_quan": self.co_quan,
            "chuc_danh": self.chuc_danh,
            "nguoi_ky": self.nguoi_ky,
            "law_family": self.law_family,
            "ngay_ban_hanh": self.ngay_ban_hanh,
            "ngay_co_hieu_luc": self.ngay_co_hieu_luc,
            "ngay_het_hieu_luc": self.ngay_het_hieu_luc,
            "is_active": self.is_active,
            "status_label": self.status_label,
            "chuong": self.chuong,
            "ten_chuong": self.ten_chuong,
            "dieu_so": self.dieu_so,
            "tieu_de_dieu": self.tieu_de_dieu,
            "khoan_so": self.khoan_so,
            "diem_so": self.diem_so,
            "path": self.path,
            "legal_path": self.legal_path,
            "chunk_type": self.chunk_type,
            "segment_level": self.segment_level,
            "fallback_reason": self.fallback_reason,
            "provision_id": self.provision_id,
            "parent_provision_id": self.parent_provision_id,
            "citation_base": self.citation_base,
            "citation_short": self.citation_short,
            "citation_full": self.citation_full,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_len_est": self.token_len_est,
            "token_len_model": self.token_len_model,
            "overflow_index": self.overflow_index,
            "overflow_total": self.overflow_total,
            "window_char_start": self.window_char_start,
            "window_char_end": self.window_char_end,
            "clause_spans": list(self.clause_spans),
            "encoder_profile": self.encoder_profile,
            "hierarchy": self.hierarchy,
        }


@dataclass
class _TextUnit:
    text: str
    start: int
    end: int
    token_len: int


@dataclass
class _ArticleWindow:
    text: str
    start: int
    end: int
    token_len: int
    overflow_index: int
    overflow_total: int
    clause_spans: List[str]
    fallback_reason: str = ""


class LegalChunkerV2:
    CHUONG_PATTERN = re.compile(
        r"^\s*(?:Ch\u01b0\u01a1ng|Chuong)\s+([IVXLCDM]+|\d+)\s*[.:]?\s*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    DIEU_PATTERN = re.compile(
        r"^\s*(?:\u0110i\u1ec1u|Dieu)\s+(?P<dieu_so>\d+[A-Za-z0-9\-]*)\s*[.:]\s*(?P<tail>.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    KHOAN_PATTERN = re.compile(
        r"^\s*(?:(?:Kho\u1ea3n|Khoan)\s+)?(?P<khoan_so>[1-9]\d{0,2})\.\s+",
        re.IGNORECASE | re.MULTILINE,
    )
    KHOAN_INLINE_PATTERN = re.compile(
        r"(?:^|[\n;:]\s+|[.!?]\s+|\s+)(?:(?:Kho\u1ea3n|Khoan)\s+)?(?P<khoan_so>[1-9]\d{0,2})\.\s+",
        re.IGNORECASE,
    )
    DIEM_PATTERN = re.compile(
        r"^\s*(?:(?:\u0110i\u1ec3m|Diem)\s+)?(?P<diem_so>[a-z\u0111])\)\s+",
        re.IGNORECASE | re.MULTILINE,
    )
    ARTICLE_MARKER_PATTERN = re.compile(
        r"(?<!\n)\s+(?=(?:\u0110i\u1ec1u|Dieu)\s+\d+[A-Za-z0-9\-]*\s*[.:])",
        re.IGNORECASE,
    )
    CHAPTER_MARKER_PATTERN = re.compile(
        r"(?<!\n)\s+(?=(?:Ch\u01b0\u01a1ng|Chuong)\s+(?:[IVXLCDM]+|\d+)\b)",
        re.IGNORECASE,
    )
    CLAUSE_MARKER_PATTERN = re.compile(
        r"(?<=[.;:])\s+(?=(?:(?:Kho\u1ea3n|Khoan)\s+)?[1-9]\d{0,2}\.\s+)",
        re.IGNORECASE,
    )
    SENTENCE_CHUNK_PATTERN = re.compile(r".+?(?:\n+|(?<=[.;:!?])\s+|$)", re.DOTALL)

    def __init__(
        self,
        min_chars: int = 1200,
        max_chars: int = 2800,
        overlap_chars: int = 120,
        encoder_model_name: Optional[str] = None,
        encoder_max_len: int = 512,
        target_chunk_tokens: int = 420,
        hard_cap_tokens: int = 448,
        overlap_tokens: int = 64,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.encoder_model_name = str(encoder_model_name or DEFAULT_MODEL)
        self.encoder_max_len = int(encoder_max_len)
        self.target_chunk_tokens = int(target_chunk_tokens)
        self.hard_cap_tokens = int(hard_cap_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self._encoder_profile = self._build_encoder_profile()

    def _build_encoder_profile(self) -> str:
        normalize_flag = os.getenv("EMBEDDING_NORMALIZE", "true").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", self.encoder_model_name.lower()).strip("_")
        return f"{slug}|norm={normalize_flag}"

    @staticmethod
    def _normalize_legal_text(text: str) -> str:
        text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    @staticmethod
    def _protect_numeric_dots(text: str) -> str:
        return re.sub(r"(?<=\d)\.(\s*)(?=\d)", lambda m: NUMERIC_DOT_PLACEHOLDER + (m.group(1) or ""), str(text or ""))

    @staticmethod
    def _restore_numeric_dots(text: str) -> str:
        return str(text or "").replace(NUMERIC_DOT_PLACEHOLDER, ".")

    def _stabilize_inline_markers(self, text: str) -> str:
        stabilized = self._normalize_legal_text(text)
        stabilized = self._protect_numeric_dots(stabilized)
        stabilized = self.CHAPTER_MARKER_PATTERN.sub("\n", stabilized)
        stabilized = self.ARTICLE_MARKER_PATTERN.sub("\n", stabilized)
        stabilized = re.sub(r"\n{2,}", "\n", stabilized).strip()
        return self._restore_numeric_dots(stabilized)

    def _stabilize_clause_markers(self, text: str) -> str:
        stabilized = self._normalize_legal_text(text)
        stabilized = self._protect_numeric_dots(stabilized)
        stabilized = self.CLAUSE_MARKER_PATTERN.sub("\n", stabilized)
        stabilized = re.sub(r"\n{2,}", "\n", stabilized).strip()
        return self._restore_numeric_dots(stabilized)

    def _build_provision_id(self, doc_id: str, dieu_so: str = "", suffix: str = "") -> str:
        parts = [doc_id or "UNKNOWN"]
        if dieu_so:
            parts.append(f"D{dieu_so}")
        if suffix:
            parts.append(suffix)
        return "_".join(parts)

    def _generate_chunk_id(
        self,
        doc_id: str,
        dieu_so: str,
        chunk_index: int,
        text: str,
        overflow_index: int = 1,
    ) -> str:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        base = self._build_provision_id(doc_id, dieu_so)
        if overflow_index > 1:
            base = f"{base}_W{overflow_index}"
        return f"{base}_C{chunk_index}_{digest}"

    @staticmethod
    def _chapter_at(
        chuong_matches: Sequence[re.Match],
        position: int,
    ) -> Tuple[str, str]:
        selected: Optional[re.Match] = None
        for match in chuong_matches:
            if match.start() <= position:
                selected = match
            else:
                break
        if selected is None:
            return "", ""
        chuong = (selected.group(1) or "").strip()
        ten_chuong = normalize_whitespace(selected.group(2) or "")
        return chuong, ten_chuong

    @staticmethod
    def _estimate_token_len(text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", str(text)))

    def _count_tokens(self, text: str, text_type: str = "passage") -> int:
        if not text:
            return 0
        return int(count_text_tokens(text, model_name=self.encoder_model_name, text_type=text_type, add_prefix=True))

    @staticmethod
    def _first_nonempty_line(text: str) -> str:
        for line in str(text or "").splitlines():
            line = normalize_whitespace(line)
            if line:
                return line
        return ""

    @staticmethod
    def _clean_article_title(text: str) -> str:
        candidate = normalize_whitespace(text)
        if not candidate:
            return ""
        candidate = re.sub(r"^[\-\.:;,\)\]]+\s*", "", candidate)
        candidate = re.sub(r"\s*[\-_:;,/]+$", "", candidate).strip()
        if len(candidate) > 160:
            candidate = candidate[:160].rsplit(" ", 1)[0].strip()
        if re.search(r"\b(?:Can cu|Căn cứ|Chuong|Dieu)\b", candidate, flags=re.IGNORECASE):
            return ""
        return candidate

    def _extract_article_structure(self, article_text: str) -> Tuple[str, str, str]:
        normalized = self._normalize_legal_text(article_text)
        match = self.DIEU_PATTERN.match(normalized)
        if not match:
            return "", "", normalized

        dieu_so = normalize_whitespace(match.group("dieu_so") or "")
        tail = normalize_whitespace(match.group("tail") or "")
        if not tail:
            return dieu_so, "", ""

        protected_tail = self._protect_numeric_dots(tail)
        clause_match = self.KHOAN_INLINE_PATTERN.search(protected_tail)
        if clause_match:
            prefix = normalize_whitespace(self._restore_numeric_dots(protected_tail[: clause_match.start()]))
            title_candidate = self._clean_article_title(prefix)
            body = normalize_whitespace(self._restore_numeric_dots(protected_tail[clause_match.start() :]))
            if 3 <= len(title_candidate) <= 140:
                return dieu_so, title_candidate, body

        first_line = self._first_nonempty_line(tail)
        if first_line and "\n" in tail:
            title_candidate = self._clean_article_title(first_line)
            remainder = normalize_whitespace(tail[len(first_line) :])
            if 3 <= len(title_candidate) <= 140 and remainder:
                return dieu_so, title_candidate, remainder

        return dieu_so, "", tail

    def _segment_body_units(self, body_text: str) -> List[_TextUnit]:
        normalized = self._normalize_legal_text(body_text)
        if not normalized:
            return []

        protected = self._protect_numeric_dots(normalized)
        units: List[_TextUnit] = []
        for match in self.SENTENCE_CHUNK_PATTERN.finditer(protected):
            raw_piece = match.group(0)
            if not raw_piece:
                continue
            leading_ws = len(raw_piece) - len(raw_piece.lstrip())
            trimmed_piece = raw_piece.strip()
            if not trimmed_piece:
                continue
            start = match.start() + leading_ws
            end = start + len(trimmed_piece)
            piece_text = normalize_whitespace(self._restore_numeric_dots(normalized[start:end]))
            if not piece_text:
                continue
            units.append(_TextUnit(text=piece_text, start=start, end=end, token_len=self._count_tokens(piece_text)))

        if units:
            return units

        cleaned = normalize_whitespace(normalized)
        return [_TextUnit(text=cleaned, start=0, end=len(cleaned), token_len=self._count_tokens(cleaned))]

    def _fallback_split_large_piece(self, piece_text: str, piece_start: int, header_tokens: int) -> List[_TextUnit]:
        words = list(re.finditer(r"\S+\s*", piece_text))
        if not words:
            return []

        target_budget = max(32, self.target_chunk_tokens - header_tokens)
        hard_budget = max(40, self.hard_cap_tokens - header_tokens)
        units: List[_TextUnit] = []
        cursor = 0
        while cursor < len(words):
            start = words[cursor].start()
            best_end = words[cursor].end()
            best_text = normalize_whitespace(piece_text[start:best_end])
            best_tokens = self._count_tokens(best_text)
            idx = cursor
            while idx + 1 < len(words):
                candidate_end = words[idx + 1].end()
                candidate_text = normalize_whitespace(piece_text[start:candidate_end])
                candidate_tokens = self._count_tokens(candidate_text)
                if candidate_tokens <= target_budget:
                    best_end = candidate_end
                    best_text = candidate_text
                    best_tokens = candidate_tokens
                    idx += 1
                    continue
                if candidate_tokens <= hard_budget and best_tokens < target_budget:
                    best_end = candidate_end
                    best_text = candidate_text
                    best_tokens = candidate_tokens
                    idx += 1
                break
            units.append(
                _TextUnit(
                    text=best_text,
                    start=piece_start + start,
                    end=piece_start + best_end,
                    token_len=best_tokens,
                )
            )
            cursor = idx + 1 if idx > cursor else cursor + 1
        return units

    def _compute_clause_ranges(self, body_text: str) -> List[Tuple[str, int, int]]:
        stabilized = self._normalize_legal_text(body_text)
        protected = self._protect_numeric_dots(stabilized)
        matches = list(self.KHOAN_INLINE_PATTERN.finditer(protected))
        if not matches:
            return []

        ranges: List[Tuple[str, int, int]] = []
        for idx, match in enumerate(matches):
            khoan_so = normalize_whitespace(match.group("khoan_so") or "")
            if not khoan_so:
                continue
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(protected)
            label = f"Khoan {khoan_so}"
            khoan_text = self._restore_numeric_dots(protected[start:end])
            point_matches = list(self.DIEM_PATTERN.finditer(self._stabilize_clause_markers(khoan_text)))
            if point_matches:
                first_point = normalize_whitespace(point_matches[0].group("diem_so") or "")
                last_point = normalize_whitespace(point_matches[-1].group("diem_so") or "")
                if first_point and last_point:
                    label = f"{label} > Diem {first_point}" if first_point == last_point else f"{label} > Diem {first_point}-{last_point}"
            ranges.append((label, start, end))
        return ranges

    @staticmethod
    def _compress_clause_labels(labels: Iterable[str]) -> List[str]:
        ordered = list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        compressed: List[str] = []
        numeric_run: List[int] = []

        def flush_run() -> None:
            nonlocal numeric_run
            if not numeric_run:
                return
            compressed.append(f"Khoan {numeric_run[0]}" if len(numeric_run) == 1 else f"Khoan {numeric_run[0]}-{numeric_run[-1]}")
            numeric_run = []

        for label in ordered:
            match = re.fullmatch(r"Khoan (\d+)", label)
            if not match:
                flush_run()
                compressed.append(label)
                continue
            value = int(match.group(1))
            if numeric_run and value == numeric_run[-1] + 1:
                numeric_run.append(value)
            else:
                flush_run()
                numeric_run = [value]

        flush_run()
        return compressed

    def _clause_spans_for_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self._compress_clause_labels(label for label, _, _ in self._compute_clause_ranges(text))

    def _split_article_windows(self, dieu_so: str, title: str, body_text: str) -> List[_ArticleWindow]:
        header = f"Dieu {dieu_so}"
        if title:
            header = f"{header}. {title}"
        normalized_body = self._normalize_legal_text(body_text)
        full_text = header if not normalized_body else f"{header}\n{normalized_body}"
        full_tokens = self._count_tokens(full_text)
        clause_spans = self._clause_spans_for_text(normalized_body)
        if full_tokens <= self.encoder_max_len:
            return [
                _ArticleWindow(
                    text=full_text,
                    start=0,
                    end=len(normalized_body),
                    token_len=full_tokens,
                    overflow_index=1,
                    overflow_total=1,
                    clause_spans=clause_spans,
                )
            ]

        header_tokens = self._count_tokens(header)
        units = self._segment_body_units(normalized_body)
        expanded_units: List[_TextUnit] = []
        for unit in units:
            if header_tokens + unit.token_len <= self.hard_cap_tokens:
                expanded_units.append(unit)
            else:
                expanded_units.extend(self._fallback_split_large_piece(unit.text, unit.start, header_tokens))

        if not expanded_units:
            return [
                _ArticleWindow(
                    text=full_text,
                    start=0,
                    end=len(normalized_body),
                    token_len=full_tokens,
                    overflow_index=1,
                    overflow_total=1,
                    clause_spans=clause_spans,
                    fallback_reason="empty_body_units",
                )
            ]

        windows: List[_ArticleWindow] = []
        cursor = 0
        while cursor < len(expanded_units):
            current_units: List[_TextUnit] = []
            token_total = header_tokens
            idx = cursor
            while idx < len(expanded_units):
                unit = expanded_units[idx]
                if current_units and token_total + unit.token_len > self.target_chunk_tokens:
                    break
                if current_units and token_total + unit.token_len > self.hard_cap_tokens:
                    break
                current_units.append(unit)
                token_total += unit.token_len
                idx += 1
                if token_total >= self.target_chunk_tokens:
                    break

            if not current_units:
                unit = expanded_units[cursor]
                current_units = [unit]
                token_total = header_tokens + unit.token_len
                idx = cursor + 1

            body_start = current_units[0].start
            body_end = current_units[-1].end
            window_body = normalize_whitespace(normalized_body[body_start:body_end])
            window_text = header if not window_body else f"{header}\n{window_body}"
            windows.append(
                _ArticleWindow(
                    text=window_text,
                    start=body_start,
                    end=body_end,
                    token_len=self._count_tokens(window_text),
                    overflow_index=len(windows) + 1,
                    overflow_total=0,
                    clause_spans=self._clause_spans_for_text(window_body),
                    fallback_reason="article_window",
                )
            )

            if idx >= len(expanded_units):
                break

            next_cursor = idx
            overlap_sum = 0
            back = idx - 1
            while back > cursor and overlap_sum < self.overlap_tokens:
                overlap_sum += expanded_units[back].token_len
                next_cursor = back
                back -= 1
            cursor = next_cursor if next_cursor > cursor else min(idx, cursor + 1)

        total = len(windows)
        for item in windows:
            item.overflow_total = total
        return windows

    @staticmethod
    def _prepare_record(record: Dict[str, Any]) -> Dict[str, Any]:
        raw_record = dict(record or {})
        normalized = normalize_document_row(raw_record)
        merged = dict(raw_record)
        merged.update(normalized)
        return merged

    def _build_chunk(
        self,
        record: Dict[str, Any],
        content: str,
        chunk_index: int,
        dieu_so: str = "",
        tieu_de_dieu: str = "",
        chuong: str = "",
        ten_chuong: str = "",
        char_start: int = 0,
        char_end: int = 0,
        overflow_index: int = 1,
        overflow_total: int = 1,
        clause_spans: Optional[List[str]] = None,
        fallback_reason: str = "",
    ) -> LegalChunk:
        content = self._normalize_legal_text(content or "")

        doc_id = normalize_doc_id(record.get("doc_id") or record.get("so_ky_hieu") or record.get("van_ban_id"))
        so_ky_hieu = record.get("so_ky_hieu") or record.get("van_ban_id") or doc_id
        citation_base = record.get("citation_base") or build_citation_base(
            str(record.get("loai_van_ban", "")),
            str(so_ky_hieu),
            str(record.get("ten_van_ban", "")),
        )
        legal_path = f"Dieu {dieu_so}" if dieu_so else "Document"
        citation_short = legal_path if dieu_so else "Nguon phap ly"
        citation_full = build_citation_full(citation_base, legal_path if dieu_so else "")
        chunk_type = "ProvisionChunk" if dieu_so else "FallbackChunk"
        segment_level = "article_window" if dieu_so and overflow_total > 1 else ("article" if dieu_so else "document")
        provision_id = self._build_provision_id(doc_id, dieu_so=dieu_so)
        effective_clause_spans = list(clause_spans or [])

        chunk_id = self._generate_chunk_id(
            doc_id=doc_id,
            dieu_so=dieu_so,
            chunk_index=chunk_index,
            text=content,
            overflow_index=overflow_index,
        )

        hierarchy = {
            "chuong": chuong,
            "ten_chuong": ten_chuong,
            "dieu": dieu_so,
            "khoan": "",
            "diem": "",
            "clause_spans": effective_clause_spans,
        }

        co_quan = str(record.get("co_quan_ban_hanh") or record.get("co_quan") or "")
        token_len_est = self._estimate_token_len(content)
        token_len_model = self._count_tokens(content)
        encoder_profile = str(record.get("encoder_profile") or self._encoder_profile)
        chunk = LegalChunk(
            chunk_id=chunk_id,
            content=content,
            doc_id=doc_id,
            so_ky_hieu=str(so_ky_hieu),
            van_ban_id=doc_id,
            ten_van_ban=str(record.get("ten_van_ban", "")),
            loai_van_ban=str(record.get("loai_van_ban", "")),
            doc_type=str(record.get("doc_type", "OTHER")),
            doc_type_rank=int(record.get("doc_type_rank", 999)),
            co_quan_ban_hanh=co_quan,
            co_quan=co_quan,
            chuc_danh=str(record.get("chuc_danh", "")),
            nguoi_ky=str(record.get("nguoi_ky", "")),
            law_family=str(record.get("law_family", "")),
            ngay_ban_hanh=record.get("ngay_ban_hanh"),
            ngay_co_hieu_luc=record.get("ngay_co_hieu_luc"),
            ngay_het_hieu_luc=record.get("ngay_het_hieu_luc"),
            is_active=bool(record.get("is_active", True)),
            status_label=str(record.get("status_label", "active")),
            chuong=chuong,
            ten_chuong=ten_chuong,
            dieu_so=dieu_so,
            tieu_de_dieu=tieu_de_dieu,
            khoan_so="",
            diem_so="",
            path=legal_path,
            legal_path=legal_path,
            chunk_type=chunk_type,
            segment_level=segment_level,
            fallback_reason=fallback_reason,
            provision_id=provision_id,
            parent_provision_id="" if dieu_so else doc_id,
            citation_base=citation_base,
            citation_short=citation_short,
            citation_full=citation_full,
            chunk_index=chunk_index,
            total_chunks=1,
            char_start=char_start,
            char_end=char_end,
            token_len_est=token_len_est,
            token_len_model=token_len_model,
            overflow_index=overflow_index,
            overflow_total=overflow_total,
            window_char_start=char_start,
            window_char_end=char_end,
            clause_spans=effective_clause_spans,
            encoder_profile=encoder_profile,
            hierarchy=hierarchy,
        )
        return chunk

    def _fallback_document_chunks(self, record: Dict[str, Any], full_text: str) -> List[LegalChunk]:
        normalized = self._normalize_legal_text(full_text)
        piece = normalized.strip()
        if not piece:
            return []
        if self._estimate_token_len(piece) <= self.encoder_max_len and self._count_tokens(piece) <= self.encoder_max_len:
            chunk = self._build_chunk(
                record=record,
                content=piece,
                chunk_index=0,
                char_start=0,
                char_end=len(piece),
                overflow_index=1,
                overflow_total=1,
                clause_spans=self._clause_spans_for_text(piece),
                fallback_reason="no_article_detected",
            )
            chunk.total_chunks = 1
            return [chunk]

        units = self._segment_body_units(piece)
        expanded_units: List[_TextUnit] = []
        for unit in units:
            if unit.token_len <= self.hard_cap_tokens:
                expanded_units.append(unit)
            else:
                expanded_units.extend(self._fallback_split_large_piece(unit.text, unit.start, header_tokens=0))

        if not expanded_units:
            chunk = self._build_chunk(
                record=record,
                content=piece,
                chunk_index=0,
                char_start=0,
                char_end=len(piece),
                overflow_index=1,
                overflow_total=1,
                clause_spans=self._clause_spans_for_text(piece),
                fallback_reason="no_article_detected",
            )
            chunk.total_chunks = 1
            return [chunk]

        chunks: List[LegalChunk] = []
        cursor = 0
        while cursor < len(expanded_units):
            current_units: List[_TextUnit] = []
            token_total = 0
            idx = cursor
            while idx < len(expanded_units):
                unit = expanded_units[idx]
                if current_units and token_total + unit.token_len > self.target_chunk_tokens:
                    break
                if current_units and token_total + unit.token_len > self.hard_cap_tokens:
                    break
                current_units.append(unit)
                token_total += unit.token_len
                idx += 1
                if token_total >= self.target_chunk_tokens:
                    break

            if not current_units:
                current_units = [expanded_units[cursor]]
                idx = cursor + 1

            body_start = current_units[0].start
            body_end = current_units[-1].end
            window_body = piece[body_start:body_end].strip()
            chunks.append(
                self._build_chunk(
                    record=record,
                    content=window_body,
                    chunk_index=len(chunks),
                    char_start=body_start,
                    char_end=body_end,
                    overflow_index=len(chunks) + 1,
                    overflow_total=0,
                    clause_spans=self._clause_spans_for_text(window_body),
                    fallback_reason="no_article_detected_window",
                )
            )

            if idx >= len(expanded_units):
                break

            next_cursor = idx
            overlap_sum = 0
            back = idx - 1
            while back > cursor and overlap_sum < self.overlap_tokens:
                overlap_sum += expanded_units[back].token_len
                next_cursor = back
                back -= 1
            cursor = next_cursor if next_cursor > cursor else min(idx, cursor + 1)

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
            chunk.overflow_total = total
        return chunks

    def chunk_document(self, record: Dict[str, Any], doc_index: int = 0) -> List[LegalChunk]:
        record = self._prepare_record(record)
        full_text = (
            record.get("toan_van_clean")
            or record.get("toan_van")
            or record.get("clean_text")
            or record.get("noi_dung")
            or ""
        )
        full_text = str(full_text or "").strip()
        if not full_text:
            return []

        full_text = self._stabilize_inline_markers(full_text)
        dieu_matches = list(self.DIEU_PATTERN.finditer(full_text))
        chuong_matches = list(self.CHUONG_PATTERN.finditer(full_text))

        if not dieu_matches:
            return self._fallback_document_chunks(record, full_text)

        all_chunks: List[LegalChunk] = []
        chunk_index = 0

        for idx, dieu_match in enumerate(dieu_matches):
            article_start = dieu_match.start()
            article_end = dieu_matches[idx + 1].start() if idx + 1 < len(dieu_matches) else len(full_text)
            article_text = full_text[article_start:article_end].strip()
            if not article_text:
                continue

            dieu_so, tieu_de_dieu, article_body = self._extract_article_structure(article_text)
            if not dieu_so:
                continue
            chuong, ten_chuong = self._chapter_at(chuong_matches, article_start)
            windows = self._split_article_windows(dieu_so, tieu_de_dieu, article_body)
            for window in windows:
                all_chunks.append(
                    self._build_chunk(
                        record=record,
                        content=window.text,
                        chunk_index=chunk_index,
                        dieu_so=dieu_so,
                        tieu_de_dieu=tieu_de_dieu,
                        chuong=chuong,
                        ten_chuong=ten_chuong,
                        char_start=article_start + window.start,
                        char_end=article_start + window.end,
                        overflow_index=window.overflow_index,
                        overflow_total=window.overflow_total,
                        clause_spans=window.clause_spans,
                        fallback_reason=window.fallback_reason,
                    )
                )
                chunk_index += 1

        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total
        return all_chunks

    def chunk_article(self, row: Dict[str, Any], article_idx: int) -> List[LegalChunk]:
        """
        Backward-compatible method for legacy article-level rows.
        """
        row = self._prepare_record(row)
        text = row.get("clean_text") or row.get("noi_dung") or row.get("toan_van_clean") or row.get("toan_van") or ""
        text = self._normalize_legal_text(str(text or ""))
        if not text:
            return []

        record = dict(row)
        if "doc_id" not in record:
            record["doc_id"] = normalize_doc_id(record.get("van_ban_id") or record.get("so_ky_hieu"))
        if "so_ky_hieu" not in record:
            record["so_ky_hieu"] = record.get("van_ban_id", record.get("doc_id"))
        if "co_quan_ban_hanh" not in record:
            record["co_quan_ban_hanh"] = record.get("co_quan", "")
        if "citation_base" not in record:
            record["citation_base"] = build_citation_base(
                str(record.get("loai_van_ban", "")),
                str(record.get("so_ky_hieu", "")),
                str(record.get("ten_van_ban", "")),
            )

        # If article header already present, let chunk_document parse structure
        if self.DIEU_PATTERN.search(self._stabilize_inline_markers(text)):
            record["toan_van_clean"] = text
            return self.chunk_document(record, doc_index=article_idx)

        dieu_so = str(row.get("dieu_so") or "").strip()
        tieu_de_dieu = self._clean_article_title(str(row.get("tieu_de_dieu") or ""))
        chuong = str(row.get("chuong") or "").strip()
        ten_chuong = str(row.get("ten_chuong") or "").strip()
        if dieu_so:
            synthesized_header = f"Dieu {dieu_so}"
            if tieu_de_dieu:
                synthesized_header = f"{synthesized_header}. {tieu_de_dieu}"
            record["toan_van_clean"] = f"{synthesized_header}\n{text}"
            chunks = self.chunk_document(record, doc_index=article_idx)
            for chunk in chunks:
                if not chunk.chuong:
                    chunk.chuong = chuong
                if not chunk.ten_chuong:
                    chunk.ten_chuong = ten_chuong
            return chunks

        return self._fallback_document_chunks(record, text)

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[LegalChunk]:
        metadata = metadata or {}
        row = dict(metadata)
        row["toan_van_clean"] = text
        row.setdefault("doc_id", normalize_doc_id(row.get("so_ky_hieu") or row.get("van_ban_id") or "UNKNOWN"))
        row.setdefault("so_ky_hieu", row.get("doc_id"))
        row.setdefault("van_ban_id", row.get("doc_id"))
        row.setdefault("citation_base", build_citation_base(row.get("loai_van_ban", ""), row.get("so_ky_hieu", ""), row.get("ten_van_ban", "")))
        return self.chunk_document(row, doc_index=0)

    def process_dataframe(self, df) -> List[LegalChunk]:
        chunks: List[LegalChunk] = []
        for idx, row in df.iterrows():
            record = row.to_dict()
            if record.get("toan_van_clean") or record.get("toan_van"):
                doc_chunks = self.chunk_document(record, doc_index=idx)
            else:
                doc_chunks = self.chunk_article(record, article_idx=idx)
            chunks.extend(doc_chunks)

            if (idx + 1) % 500 == 0:
                logger.info("[CHUNKER] Processed %s rows, generated %s chunks", idx + 1, len(chunks))

        logger.info("[CHUNKER] Done: %s chunks from %s rows", len(chunks), len(df))
        return chunks


def get_legal_chunker(
    min_chars: int = 1200,
    max_chars: int = 2800,
    overlap_chars: int = 120,
    encoder_model_name: Optional[str] = None,
) -> LegalChunkerV2:
    return LegalChunkerV2(
        min_chars=min_chars,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        encoder_model_name=encoder_model_name,
    )
