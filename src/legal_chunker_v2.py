"""
Legal document chunker v2.

Supports two input shapes:
1. Full document text (preferred): `toan_van_clean` / `toan_van`
2. Legacy article-level rows: `clean_text` / `noi_dung`

Chunking strategy is deterministic and legal-aware:
Chuong -> Dieu -> Khoan -> Diem
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.legal_data_processing import build_citation_base, normalize_doc_id, normalize_whitespace

logger = logging.getLogger(__name__)


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
            "encoder_profile": self.encoder_profile,
            "hierarchy": self.hierarchy,
        }


class LegalChunkerV2:
    CHUONG_PATTERN = re.compile(
        r"^\s*(?:Ch\u01b0\u01a1ng|Chuong)\s+([IVXLCDM]+|\d+)\s*[.:]?\s*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    DIEU_PATTERN = re.compile(
        r"^\s*(?:\u0110i\u1ec1u|Dieu)\s+(?P<dieu_so>\d+[A-Za-z0-9\-]*)\s*[.:]?\s*(?P<title>.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    KHOAN_PATTERN = re.compile(
        r"^\s*(?:(?:Kho\u1ea3n|Khoan)\s+)?(?P<khoan_so>\d+)\.\s+",
        re.IGNORECASE | re.MULTILINE,
    )
    DIEM_PATTERN = re.compile(
        r"^\s*(?:(?:\u0110i\u1ec3m|Diem)\s+)?(?P<diem_so>[a-z\u0111])\)\s+",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self, min_chars: int = 1200, max_chars: int = 2800, overlap_chars: int = 120):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def _normalize_legal_text(text: str) -> str:
        text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    @staticmethod
    def _stabilize_inline_markers(text: str) -> str:
        """
        Stabilize legal structure markers that appear inline so downstream
        line-anchored regex can parse Chuong/Dieu/Khoan/Diem deterministically.
        """
        stabilized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        stabilized = re.sub(r"[ \t]+", " ", stabilized)
        stabilized = re.sub(r"\n{2,}", "\n", stabilized).strip()

        replacements = [
            (
                r"(?<!\n)\s+(?=(?:Ch\u01b0\u01a1ng|Chuong)\s+(?:[IVXLCDM]+|\d+)\b)",
                "\n",
            ),
            (
                r"(?<!\n)\s+(?=(?:\u0110i\u1ec1u|Dieu)\s+\d+[A-Za-z0-9\-]*\b)",
                "\n",
            ),
            (
                r"(?<=[\.;:])\s+(?=(?:(?:Kho\u1ea3n|Khoan)\s+)?\d+\.\s+)",
                "\n",
            ),
            (
                r"(?<=[\.;:])\s+(?=(?:(?:\u0110i\u1ec3m|Diem)\s+)?[a-z\u0111]\)\s+)",
                "\n",
            ),
        ]
        for pattern, replacement in replacements:
            stabilized = re.sub(pattern, replacement, stabilized, flags=re.IGNORECASE)

        lines = [line.strip() for line in stabilized.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def _build_provision_id(
        self,
        doc_id: str,
        dieu_so: str = "",
        khoan_so: str = "",
        diem_so: str = "",
        suffix: str = "",
    ) -> str:
        parts = [doc_id or "UNKNOWN"]
        if dieu_so:
            parts.append(f"D{dieu_so}")
        if khoan_so:
            parts.append(f"K{khoan_so}")
        if diem_so:
            parts.append(f"P{diem_so}")
        if suffix:
            parts.append(suffix)
        return "_".join(parts)

    def _generate_chunk_id(
        self,
        doc_id: str,
        dieu_so: str,
        khoan_so: str,
        diem_so: str,
        chunk_index: int,
        text: str,
    ) -> str:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        base = self._build_provision_id(doc_id, dieu_so, khoan_so, diem_so)
        return f"{base}_C{chunk_index}_{digest}"

    def _extract_segments(
        self,
        text: str,
        pattern: re.Pattern,
        key: str,
        base_offset: int = 0,
    ) -> List[Tuple[str, str, int, int]]:
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        segments: List[Tuple[str, str, int, int]] = []
        if matches[0].start() > 0:
            pre = text[: matches[0].start()].strip()
            if pre:
                segments.append(("", pre, base_offset, base_offset + matches[0].start()))

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            seg_text = text[start:end].strip()
            seg_id = (match.group(key) or "").strip()
            segments.append((seg_id, seg_text, base_offset + start, base_offset + end))

        return segments

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        search = text[start:end]
        if not search:
            return end

        for idx in range(len(search) - 1, -1, -1):
            if search[idx] in ".;:\n":
                return start + idx + 1

        for idx in range(len(search) - 1, -1, -1):
            if search[idx] == " ":
                return start + idx + 1

        return end

    def _hard_split(self, text: str, base_offset: int) -> List[Tuple[str, int, int]]:
        text = normalize_whitespace(text)
        if len(text) <= self.max_chars:
            return [(text, base_offset, base_offset + len(text))]

        chunks: List[Tuple[str, int, int]] = []
        cursor = 0
        text_len = len(text)

        while cursor < text_len:
            chunk_start = cursor
            ideal_end = min(chunk_start + self.max_chars, text_len)
            if ideal_end >= text_len:
                chunk_end = text_len
            else:
                min_end = min(chunk_start + self.min_chars, text_len)
                chunk_end = self._find_split_point(text, min_end, ideal_end)
                if chunk_end <= chunk_start:
                    chunk_end = ideal_end

            piece = text[chunk_start:chunk_end].strip()
            if piece:
                chunks.append((piece, base_offset + chunk_start, base_offset + chunk_end))

            if chunk_end >= text_len:
                break

            cursor = max(chunk_end - self.overlap_chars, chunk_start + 1)

        return chunks

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
        ten_chuong = (selected.group(2) or "").strip()
        return chuong, ten_chuong

    def _compose_path(self, dieu_so: str, khoan_so: str, diem_so: str) -> str:
        parts: List[str] = []
        if dieu_so:
            parts.append(f"Dieu {dieu_so}")
        if khoan_so:
            parts.append(f"Khoan {khoan_so}")
        if diem_so:
            parts.append(f"Diem {diem_so}")
        return " > ".join(parts)

    @staticmethod
    def _classify_chunk_type(
        dieu_so: str,
        khoan_so: str,
        diem_so: str,
        fallback_reason: str,
    ) -> Tuple[str, str]:
        if fallback_reason or not dieu_so:
            return "FallbackChunk", "document"
        if diem_so:
            return "ClauseChunk", "point"
        if khoan_so:
            return "ClauseChunk", "clause"
        return "ProvisionChunk", "article"

    def _compose_citation(self, base: str, path: str) -> Tuple[str, str]:
        short = path or "Nguon phap ly"
        full = f"{base} | {path}" if path else base
        return short, full

    @staticmethod
    def _estimate_token_len(text: str) -> int:
        # Deterministic, model-agnostic estimate for payload diagnostics.
        if not text:
            return 0
        return len(re.findall(r"\S+", str(text)))

    def _build_chunk(
        self,
        record: Dict[str, Any],
        content: str,
        chunk_index: int,
        dieu_so: str = "",
        tieu_de_dieu: str = "",
        khoan_so: str = "",
        diem_so: str = "",
        chuong: str = "",
        ten_chuong: str = "",
        char_start: int = 0,
        char_end: int = 0,
        parent_provision_id: str = "",
        suffix: str = "",
        fallback_reason: str = "",
    ) -> LegalChunk:
        content = (content or "").strip()
        if dieu_so:
            article_header = f"Dieu {dieu_so}"
            if tieu_de_dieu:
                article_header = f"{article_header}. {tieu_de_dieu}"
            heading_pattern = re.compile(
                rf"^\s*(?:\u0110i\u1ec1u|Dieu)\s+{re.escape(dieu_so)}\b",
                re.IGNORECASE,
            )
            if not heading_pattern.search(content):
                content = f"{article_header}\n{content}" if content else article_header
        elif tieu_de_dieu and not content.lower().startswith(tieu_de_dieu.lower()):
            content = f"{tieu_de_dieu}\n{content}" if content else tieu_de_dieu

        doc_id = normalize_doc_id(record.get("doc_id") or record.get("so_ky_hieu") or record.get("van_ban_id"))
        so_ky_hieu = record.get("so_ky_hieu") or record.get("van_ban_id") or doc_id
        citation_base = record.get("citation_base") or build_citation_base(
            str(record.get("loai_van_ban", "")),
            str(so_ky_hieu),
            str(record.get("ten_van_ban", "")),
        )
        path = self._compose_path(dieu_so, khoan_so, diem_so)
        legal_path = path or "Document"
        citation_short, citation_full = self._compose_citation(citation_base, path)
        chunk_type, segment_level = self._classify_chunk_type(
            dieu_so=dieu_so,
            khoan_so=khoan_so,
            diem_so=diem_so,
            fallback_reason=fallback_reason,
        )

        provision_id = self._build_provision_id(
            doc_id,
            dieu_so=dieu_so,
            khoan_so=khoan_so,
            diem_so=diem_so,
            suffix=suffix,
        )

        chunk_id = self._generate_chunk_id(
            doc_id=doc_id,
            dieu_so=dieu_so,
            khoan_so=khoan_so,
            diem_so=diem_so,
            chunk_index=chunk_index,
            text=content,
        )

        hierarchy = {
            "chuong": chuong,
            "ten_chuong": ten_chuong,
            "dieu": dieu_so,
            "khoan": khoan_so,
            "diem": diem_so,
        }

        co_quan = str(record.get("co_quan_ban_hanh") or record.get("co_quan") or "")
        token_len_est = self._estimate_token_len(content)
        encoder_profile = str(record.get("encoder_profile") or "default")
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
            khoan_so=khoan_so,
            diem_so=diem_so,
            path=path,
            legal_path=legal_path,
            chunk_type=chunk_type,
            segment_level=segment_level,
            fallback_reason=fallback_reason,
            provision_id=provision_id,
            parent_provision_id=parent_provision_id,
            citation_base=citation_base,
            citation_short=citation_short,
            citation_full=citation_full,
            chunk_index=chunk_index,
            total_chunks=1,
            char_start=char_start,
            char_end=char_end,
            token_len_est=token_len_est,
            encoder_profile=encoder_profile,
            hierarchy=hierarchy,
        )
        return chunk

    def _split_article_units(
        self,
        article_text: str,
        article_start_offset: int,
    ) -> List[Tuple[str, str, str, int, int]]:
        """
        Returns tuples: (khoan_so, diem_so, text, char_start, char_end).
        """
        khoan_segments = self._extract_segments(
            article_text,
            self.KHOAN_PATTERN,
            key="khoan_so",
            base_offset=article_start_offset,
        )

        if not khoan_segments:
            return [("", "", article_text, article_start_offset, article_start_offset + len(article_text))]

        units: List[Tuple[str, str, str, int, int]] = []
        for khoan_so, khoan_text, khoan_start, khoan_end in khoan_segments:
            diem_segments = self._extract_segments(
                khoan_text,
                self.DIEM_PATTERN,
                key="diem_so",
                base_offset=khoan_start,
            )
            if len(diem_segments) <= 1:
                units.append((khoan_so, "", khoan_text, khoan_start, khoan_end))
                continue

            for diem_so, diem_text, diem_start, diem_end in diem_segments:
                units.append((khoan_so, diem_so, diem_text, diem_start, diem_end))

        return units

    def _chunk_unit_text(
        self,
        text: str,
        char_start: int,
    ) -> List[Tuple[str, int, int, str]]:
        """
        Split long unit text and return list of (content, start, end, suffix).
        """
        pieces = self._hard_split(text, char_start)
        if len(pieces) == 1:
            content, start, end = pieces[0]
            return [(content, start, end, "")]

        result: List[Tuple[str, int, int, str]] = []
        for idx, (content, start, end) in enumerate(pieces):
            result.append((content, start, end, f"S{idx + 1}"))
        return result

    def chunk_document(self, record: Dict[str, Any], doc_index: int = 0) -> List[LegalChunk]:
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
            # Fallback for unstructured document
            chunks: List[LegalChunk] = []
            for idx, (piece, start, end) in enumerate(self._hard_split(full_text, 0)):
                chunks.append(
                    self._build_chunk(
                        record=record,
                        content=piece,
                        chunk_index=idx,
                        char_start=start,
                        char_end=end,
                        suffix=f"F{idx + 1}" if idx > 0 else "",
                        fallback_reason="no_article_detected" if idx == 0 else "no_article_detected_hard_split",
                    )
                )
            total = len(chunks)
            for chunk in chunks:
                chunk.total_chunks = total
            return chunks

        all_chunks: List[LegalChunk] = []
        chunk_index = 0

        for idx, dieu_match in enumerate(dieu_matches):
            article_start = dieu_match.start()
            article_end = dieu_matches[idx + 1].start() if idx + 1 < len(dieu_matches) else len(full_text)
            article_text = full_text[article_start:article_end].strip()
            if not article_text:
                continue

            dieu_so = (dieu_match.group("dieu_so") or "").strip()
            tieu_de_dieu = (dieu_match.group("title") or "").strip()
            chuong, ten_chuong = self._chapter_at(chuong_matches, article_start)

            first_newline = article_text.find("\n")
            if first_newline >= 0:
                article_body = article_text[first_newline + 1 :].strip()
                body_offset = article_start + first_newline + 1
            else:
                article_body = article_text
                body_offset = article_start

            units = self._split_article_units(article_body, body_offset)
            for khoan_so, diem_so, unit_text, unit_start, unit_end in units:
                parent_provision_id = ""
                if diem_so and khoan_so:
                    parent_provision_id = self._build_provision_id(
                        normalize_doc_id(record.get("doc_id") or record.get("so_ky_hieu") or record.get("van_ban_id")),
                        dieu_so=dieu_so,
                        khoan_so=khoan_so,
                    )
                elif khoan_so:
                    parent_provision_id = self._build_provision_id(
                        normalize_doc_id(record.get("doc_id") or record.get("so_ky_hieu") or record.get("van_ban_id")),
                        dieu_so=dieu_so,
                    )

                for piece, start, end, suffix in self._chunk_unit_text(unit_text, unit_start):
                    all_chunks.append(
                        self._build_chunk(
                            record=record,
                            content=piece,
                            chunk_index=chunk_index,
                            dieu_so=dieu_so,
                            tieu_de_dieu=tieu_de_dieu,
                            khoan_so=khoan_so,
                            diem_so=diem_so,
                            chuong=chuong,
                            ten_chuong=ten_chuong,
                            char_start=start,
                            char_end=end,
                            parent_provision_id=parent_provision_id,
                            suffix=suffix,
                            fallback_reason="hard_split_within_provision" if suffix else "",
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
        text = row.get("clean_text") or row.get("noi_dung") or row.get("toan_van_clean") or row.get("toan_van") or ""
        text = self._stabilize_inline_markers(str(text or ""))
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
        if self.DIEU_PATTERN.search(text):
            record["toan_van_clean"] = text
            return self.chunk_document(record, doc_index=article_idx)

        # Otherwise treat as single article with provided metadata
        dieu_so = str(row.get("dieu_so") or "").strip()
        tieu_de_dieu = str(row.get("tieu_de_dieu") or "").strip()
        chuong = str(row.get("chuong") or "").strip()
        ten_chuong = str(row.get("ten_chuong") or "").strip()

        chunks: List[LegalChunk] = []
        for idx, (piece, start, end) in enumerate(self._hard_split(normalize_whitespace(text), 0)):
            chunks.append(
                self._build_chunk(
                    record=record,
                    content=piece,
                    chunk_index=idx,
                    dieu_so=dieu_so,
                    tieu_de_dieu=tieu_de_dieu,
                    chuong=chuong,
                    ten_chuong=ten_chuong,
                    char_start=start,
                    char_end=end,
                    suffix=f"L{idx + 1}" if idx > 0 else "",
                    fallback_reason="legacy_article_hard_split" if idx > 0 else "",
                )
            )

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

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


def get_legal_chunker(min_chars: int = 1200, max_chars: int = 2800, overlap_chars: int = 120) -> LegalChunkerV2:
    return LegalChunkerV2(min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars)
