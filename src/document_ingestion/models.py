from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_NEEDS_OCR = "needs_ocr"
STATUS_DELETED = "deleted"


@dataclass
class IngestRequest:
    doc_id: str
    user_id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    saved_file_path: Path
    artifact_dir: Path
    upload_time: str
    content_mode: str = "text"
    ocr_provider: str = "default"


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    blocks: List[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedDocument:
    file_path: Path
    file_type: str
    parser_used: str
    text: str
    normalized_text: str
    pages: List[ExtractedPage] = field(default_factory=list)
    ocr_provider: Optional[str] = None
    needs_ocr: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else (1 if self.normalized_text else 0)

    @property
    def text_chars(self) -> int:
        return len(self.normalized_text or "")

    def resolve_page_number(self, start_char: int, end_char: int) -> Optional[int]:
        for page in self.pages:
            if page.start_char <= start_char < max(page.end_char, page.start_char + 1):
                return page.page_number
            if start_char <= page.start_char <= end_char:
                return page.page_number
        return self.pages[0].page_number if self.pages else None


@dataclass
class IngestedChunk:
    chunk_id: str
    content: str
    doc_id: str
    so_ky_hieu: str = ""
    van_ban_id: str = ""
    ten_van_ban: str = ""
    loai_van_ban: str = ""
    doc_type: str = "USER_UPLOAD"
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
    chunk_type: str = "GenericDocumentChunk"
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
    user_id: str = ""
    session_id: str = ""
    filename: str = ""
    source_type: str = "user_upload"
    page_number: Optional[int] = None
    upload_time: str = ""
    ingest_mode: str = "text"
    parser_used: str = ""
    ocr_provider: Optional[str] = None

    @property
    def text(self) -> str:
        return self.content

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "IngestedChunk":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestOutcome:
    status: str
    chunks: List[IngestedChunk] = field(default_factory=list)
    parser_used: str = ""
    ocr_provider: Optional[str] = None
    page_count: int = 0
    text_chars: int = 0
    error_message: Optional[str] = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
