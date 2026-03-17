"""Objective legal query extraction and deterministic question-unit synthesis."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.legal_data_processing import strip_accents
from src.services.remote_llm_client import RemoteLLMClient

ALLOWED_ISSUE_TAGS = {"hinh_su", "dan_su", "hanh_chinh", "boi_thuong"}
DEFAULT_QUESTION_FOCUS = "chu_the_chiu_trach_nhiem"
ALLOWED_ACTOR_TYPES = {
    "ca_nhan",
    "to_chuc",
    "quan_ly",
    "nhan_vien",
    "nguoi_thuc_hien_duoc_neu",
    "nguoi_khac",
    "khong_ro",
}
ALLOWED_CONDITION_KINDS = {
    "tuoi",
    "thoi_diem",
    "so_lan",
    "so_tien",
    "hau_qua",
    "quan_he_lao_dong",
    "khac",
}
ALLOWED_CLAUSE_INTENTS = {
    "xac_dinh_chu_the",
    "dieu_kien_ap_dung",
    "trach_nhiem_hinh_su",
    "trach_nhiem_dan_su",
    "trach_nhiem_hanh_chinh",
    "boi_thuong",
}
ISSUE_LABELS = {
    "hinh_su": "trách nhiệm hình sự",
    "dan_su": "trách nhiệm dân sự",
    "hanh_chinh": "trách nhiệm hành chính",
    "boi_thuong": "bồi thường thiệt hại",
}
ACTOR_HINTS: Tuple[Tuple[str, str], ...] = (
    ("giám đốc công ty", "quan_ly"),
    ("chủ tịch", "quan_ly"),
    ("phó giám đốc", "quan_ly"),
    ("giám đốc", "quan_ly"),
    ("cấp trên", "quan_ly"),
    ("kế toán trưởng", "quan_ly"),
    ("kế toán", "nhan_vien"),
    ("thủ quỹ", "nhan_vien"),
    ("nhân viên kỹ thuật", "nhan_vien"),
    ("nhân viên", "nhan_vien"),
    ("người lao động", "nhan_vien"),
    ("người sử dụng lao động", "to_chuc"),
    ("tài xế", "nhan_vien"),
    ("bác sĩ", "ca_nhan"),
    ("y sĩ", "ca_nhan"),
    ("chủ quán", "ca_nhan"),
    ("chủ hộ", "ca_nhan"),
    ("chủ đất", "ca_nhan"),
    ("bị can", "nguoi_khac"),
    ("bị cáo", "nguoi_khac"),
    ("người bị hại", "nguoi_khac"),
    ("nạn nhân", "nguoi_khac"),
    ("nguyên đơn", "nguoi_khac"),
    ("bị đơn", "nguoi_khac"),
    ("bên mua", "nguoi_khac"),
    ("bên bán", "nguoi_khac"),
    ("bên thuê", "nguoi_khac"),
    ("bên cho thuê", "nguoi_khac"),
    ("công ty", "to_chuc"),
    ("doanh nghiệp", "to_chuc"),
    ("hộ kinh doanh", "to_chuc"),
    ("ủy ban nhân dân", "to_chuc"),
    ("ubnd", "to_chuc"),
    ("bệnh viện", "to_chuc"),
    ("trường học", "to_chuc"),
    ("ngân hàng", "to_chuc"),
    ("pháp nhân", "to_chuc"),
    ("cá nhân", "ca_nhan"),
)
ACTION_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("xả thẳng nước thải chưa xử lý ra sông", ("xả thẳng nước thải chưa xử lý ra sông", "xa thang nuoc thai chua xu ly ra song")),
    ("xả nước thải chưa xử lý ra sông", ("xả nước thải chưa xử lý ra sông", "xa nuoc thai chua xu ly ra song")),
    ("xả thải trái phép", ("xả thải trái phép", "xa thai trai phep")),
    ("xả nước thải không phép", ("xả nước thải không phép", "xa nuoc thai khong phep")),
    ("xả thải", ("xả thải", "xa thai", "xả nước thải", "xa nuoc thai")),
    ("hiếp dâm", ("hiếp dâm", "hiep dam")),
    ("lái xe gây tai nạn chết người", ("lái xe gây tai nạn chết người", "lai xe gay tai nan chet nguoi")),
    ("gây tai nạn chết người", ("gây tai nạn chết người", "gay tai nan chet nguoi")),
    ("gây tai nạn", ("gây tai nạn", "gay tai nan")),
    ("tông hỏng nhà dân", ("tông hỏng nhà dân", "tong hong nha dan")),
    ("cố ý gây thương tích", ("cố ý gây thương tích", "co y gay thuong tich")),
    ("gây thương tích", ("gây thương tích", "gay thuong tich", "đánh người", "danh nguoi")),
    ("lừa đảo chiếm đoạt tài sản", ("lừa đảo chiếm đoạt tài sản", "lua dao chiem doat tai san")),
    ("chiếm đoạt tài sản", ("chiếm đoạt tài sản", "chiem doat tai san")),
    ("trộm cắp", ("trộm cắp", "trom cap")),
    ("lừa đảo", ("lừa đảo", "lua dao")),
    ("đánh bạc", ("đánh bạc", "danh bac")),
    ("tổ chức đánh bạc", ("tổ chức đánh bạc", "to chuc danh bac")),
    ("tàng trữ trái phép chất ma túy", ("tàng trữ trái phép chất ma túy", "tang tru trai phep chat ma tuy")),
    ("mua bán trái phép chất ma túy", ("mua bán trái phép chất ma túy", "mua ban trai phep chat ma tuy")),
    ("vận chuyển trái phép chất ma túy", ("vận chuyển trái phép chất ma túy", "van chuyen trai phep chat ma tuy")),
    ("tham ô tài sản", ("tham ô tài sản", "tham o tai san")),
    ("nhận hối lộ", ("nhận hối lộ", "nhan hoi lo")),
    ("đưa hối lộ", ("đưa hối lộ", "dua hoi lo")),
    ("xây dựng không phép", ("xây dựng không phép", "xay dung khong phep")),
    ("lấn chiếm đất đai", ("lấn chiếm đất đai", "lan chiem dat dai")),
    ("khai thác cát trái phép", ("khai thác cát trái phép", "khai thac cat trai phep")),
    ("đơn phương chấm dứt hợp đồng lao động trái pháp luật", ("đơn phương chấm dứt hợp đồng lao động trái pháp luật", "don phuong cham dut hop dong lao dong trai phap luat")),
    ("sa thải trái pháp luật", ("sa thải trái pháp luật", "sa thai trai phap luat")),
    ("trốn đóng bảo hiểm xã hội", ("trốn đóng bảo hiểm xã hội", "tron dong bao hiem xa hoi")),
    ("nợ lương", ("nợ lương", "no luong", "cham tra luong")),
    ("trốn thuế", ("trốn thuế", "tron thue")),
    ("phát tán dữ liệu cá nhân", ("phát tán dữ liệu cá nhân", "phat tan du lieu ca nhan")),
    ("xâm phạm dữ liệu cá nhân", ("xâm phạm dữ liệu cá nhân", "xam pham du lieu ca nhan")),
    ("điều khiển xe có nồng độ cồn", ("điều khiển xe có nồng độ cồn", "dieu khien xe co nong do con", "uống rượu bia lái xe", "uong ruou bia lai xe")),
    ("không đội mũ bảo hiểm", ("không đội mũ bảo hiểm", "khong doi mu bao hiem")),
)
QUESTION_FOCUS_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("xac_dinh_ai_vi_pham_ai", ("ai là người vi phạm ai", "ai vi pham ai")),
    ("chu_the_chiu_trach_nhiem", ("thuộc về ai", "hay cả hai", "ai chịu", "chịu trách nhiệm")),
    ("muc_che_tai", ("mức phạt", "tiền phạt", "trừ vào tiền phạt")),
    ("dieu_kien_ap_dung", ("điều kiện", "độ tuổi", "bao nhiêu tuổi", "miễn trách nhiệm")),
    ("khai_niem_dinh_nghia", ("là gì", "la gi", "được hiểu là gì", "duoc hieu la gi", "khái niệm", "khai niem", "định nghĩa", "dinh nghia")),
)
GENERIC_MAIN_ACT_PATTERNS = (
    "hành vi được nêu", "hanh vi duoc neu", "hành vi chính", "hanh vi chinh", "tình huống được nêu", "tinh huong duoc neu",
)
DRIFT_TOKENS = ("mang thai hộ", "hỗ trợ sinh sản", "tinh giản biên chế", "thai sản")
LAW_HINT_PATTERNS: Tuple[str, ...] = (
    r"\b(Luật\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Luat\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Bộ luật\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Bo luat\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Nghị định\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Nghi dinh\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Thông tư\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Thong tu\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Nghị quyết\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Nghi quyet\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Pháp lệnh\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Phap lenh\s+[A-ZA-Za-z0-9/\- ]{2,80})",
    r"\b(Quyết định\s+[A-ZÀ-ỴA-Za-z0-9/\- ]{2,80})",
    r"\b(Quyet dinh\s+[A-ZA-Za-z0-9/\- ]{2,80})",
)


@dataclass
class SurfaceActor:
    name: str
    actor_type: str = "khong_ro"
    source_span: str = ""

    def to_public_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class KeyCondition:
    kind: str
    value: str
    source_span: str = ""

    def to_public_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ClauseUnit:
    clause_label: str
    actor_refs: List[str] = field(default_factory=list)
    act_span: str = ""
    condition_refs: List[str] = field(default_factory=list)
    intent: str = "xac_dinh_chu_the"

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QuestionUnit:
    question_label: str
    actor_refs: List[str] = field(default_factory=list)
    act_span: str = ""
    condition_refs: List[str] = field(default_factory=list)
    legal_axis: str = ""
    user_asked_text: str = ""
    intent: str = "xac_dinh_chu_the"

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StructuredRewriteExtraction:
    needs_decomposition: bool = False
    main_act: str = ""
    surface_actors: List[SurfaceActor] = field(default_factory=list)
    conditions: List[KeyCondition] = field(default_factory=list)
    explicit_legal_axes: List[str] = field(default_factory=list)
    question_focus: str = DEFAULT_QUESTION_FOCUS
    clause_units: List[ClauseUnit] = field(default_factory=list)
    named_law_hints: List[str] = field(default_factory=list)
    question_units: List[QuestionUnit] = field(default_factory=list)
    main_act_candidate_reasons: List[str] = field(default_factory=list)


@dataclass
class QueryRewriteResult:
    original_query: str
    queries: List[str]
    status: str
    raw_response: str = ""
    missing_numbers: List[str] = field(default_factory=list)
    fallback_reason: str = ""
    latency_ms: float = 0.0
    rewrite_strategy: str = "structured_synthesis"
    quality_passed: bool = False
    retrieval_query_set: List[str] = field(default_factory=list)
    actor_mentions: List[str] = field(default_factory=list)
    issue_tags: List[str] = field(default_factory=list)
    question_focus: str = DEFAULT_QUESTION_FOCUS
    needs_decomposition: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    main_act: str = ""
    surface_actors: List[Dict[str, str]] = field(default_factory=list)
    conditions: List[Dict[str, str]] = field(default_factory=list)
    clause_units: List[Dict[str, object]] = field(default_factory=list)
    named_law_hints: List[str] = field(default_factory=list)
    question_units: List[Dict[str, object]] = field(default_factory=list)
    retrieval_query_plan: List[Dict[str, object]] = field(default_factory=list)
    main_act_candidate_reasons: List[str] = field(default_factory=list)
    decomposition_policy: str = "objective_question_unit"

    @property
    def rewrite_used(self) -> bool:
        effective_queries = self.retrieval_query_set or self.queries
        return self.status == "ok" and any(q != self.original_query for q in effective_queries)

    @property
    def query_facets(self) -> Dict[str, object]:
        return {
            "actors": list(self.actor_mentions),
            "issues": list(self.issue_tags),
            "focus": self.question_focus,
            "main_act": self.main_act,
            "conditions": [item.get("value", "") for item in self.conditions if str(item.get("value", "")).strip()],
        }

    @property
    def debug_payload(self) -> Dict[str, object]:
        return {
            "needs_decomposition": bool(self.needs_decomposition),
            "main_act": self.main_act,
            "main_act_candidate_reasons": list(self.main_act_candidate_reasons),
            "actor_mentions": list(self.actor_mentions),
            "surface_actors": list(self.surface_actors),
            "conditions": list(self.conditions),
            "issue_tags": list(self.issue_tags),
            "explicit_legal_axes": list(self.issue_tags),
            "question_focus": self.question_focus,
            "clause_units": list(self.clause_units),
            "named_law_hints": list(self.named_law_hints),
            "question_units": list(self.question_units),
            "retrieval_query_plan": list(self.retrieval_query_plan),
            "decomposition_policy": self.decomposition_policy,
            "rejection_reasons": list(self.rejection_reasons),
            "quality_fail_reasons": list(self.rejection_reasons),
        }


class LegalQueryRewriter:
    EXTRACTION_SCHEMA: Dict[str, object] = {
        "type": "object",
        "properties": {
            "needs_decomposition": {"type": "boolean"},
            "main_act": {"type": "string"},
            "surface_actors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "actor_type": {"type": "string", "enum": sorted(ALLOWED_ACTOR_TYPES)},
                        "source_span": {"type": "string"},
                    },
                    "required": ["name", "actor_type", "source_span"],
                },
                "maxItems": 6,
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": sorted(ALLOWED_CONDITION_KINDS)},
                        "value": {"type": "string"},
                        "source_span": {"type": "string"},
                    },
                    "required": ["kind", "value", "source_span"],
                },
                "maxItems": 8,
            },
            "explicit_legal_axes": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "question_focus": {"type": "string"},
            "clause_units": {"type": "array", "items": {"type": "object"}, "maxItems": 6},
            "named_law_hints": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "question_units": {"type": "array", "items": {"type": "object"}, "maxItems": 4},
        },
        "required": ["needs_decomposition", "main_act", "surface_actors", "conditions", "explicit_legal_axes", "question_focus", "clause_units", "named_law_hints", "question_units"],
    }

    def __init__(self, tgi_url: str | None = None, timeout_s: float | None = None, max_subqueries: int | None = None, provider_type: str | None = None, base_url: str | None = None, model_name: str | None = None):
        self.tgi_url = tgi_url or "http://222.255.250.24:11434"
        self.timeout_s = float(timeout_s or 30.0)
        self.max_subqueries = min(3, int(max_subqueries or 3))
        self.provider_type = str(provider_type or "ollama_generate").strip()
        self.base_url = str(base_url or self.tgi_url).strip()
        self.model_name = str(model_name or "qwen3.5:9b").strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().strip('"“”')

    @staticmethod
    def _normalize_key(text: str) -> str:
        return strip_accents(LegalQueryRewriter._normalize_text(text)).lower()

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return sorted(set(re.findall(r"\d+", str(text or ""))))

    def _should_rewrite(self, query: str) -> bool:
        normalized = self._normalize_key(query)
        if len(normalized) >= 100 or normalized.count("?") >= 1:
            return True
        if len(self._extract_question_segments(query)) >= 2:
            return True
        if any(marker in normalized for marker in (" hay ca hai", " ai vi pham ai", " thuoc ve ai", " ai chiu")):
            return True
        if len(self._heuristic_named_law_hints(query)) >= 1 and len(self._heuristic_surface_actors(query)) >= 2:
            return True
        return len(self._heuristic_surface_actors(query)) >= 2 or len(self._heuristic_conditions(query)) >= 2 or len(self._heuristic_explicit_axes(query)) >= 2 or normalized.count(" nhung ") >= 1

    def _build_prompt(self, user_query: str) -> str:
        template = """Bạn là một Chuyên gia Trích xuất Dữ liệu Pháp lý khách quan và trung lập. Nhiệm vụ: bóc tách cấu trúc bề mặt của câu hỏi thành JSON hợp lệ cho retrieval.
Nguyên tắc: không phán xét; không kết luận ai có tội/vô tội; main_act phải là hành vi gây hậu quả pháp lý chính, không phải tình tiết phụ như từ thiện/nộp thuế; giữ nguyên source_span cho tên riêng, số tuổi, số tiền, số lần, mệnh đề quan hệ lao động; named_law_hints chỉ là gợi ý yếu; question_units phải tách theo từng ý hỏi độc lập; chỉ xuất JSON, không giải thích.

Few-shot 1:
Input: Công ty A nộp thuế đủ và làm từ thiện hàng tỷ đồng. Giám đốc ép nhân viên xả thẳng nước thải chưa xử lý ra sông. Xin hỏi thành tích từ thiện có được trừ vào tiền phạt không? Căn cứ theo Luật Lao động, nhân viên làm theo chỉ đạo cấp trên thì có được miễn trách nhiệm hình sự không?
Output: {"needs_decomposition":true,"main_act":"xả thẳng nước thải chưa xử lý ra sông","surface_actors":[{"name":"Công ty A","actor_type":"to_chuc","source_span":"Công ty A"},{"name":"Giám đốc","actor_type":"quan_ly","source_span":"Giám đốc"},{"name":"Nhân viên","actor_type":"nhan_vien","source_span":"Nhân viên"}],"conditions":[{"kind":"khac","value":"nộp thuế đủ","source_span":"nộp thuế đủ"},{"kind":"khac","value":"làm từ thiện hàng tỷ đồng","source_span":"làm từ thiện hàng tỷ đồng"},{"kind":"quan_he_lao_dong","value":"làm theo chỉ đạo cấp trên","source_span":"làm theo chỉ đạo cấp trên"}],"explicit_legal_axes":["hanh_chinh","hinh_su"],"question_focus":"nhieu_y_hoi","clause_units":[],"named_law_hints":["Luật Lao động"],"question_units":[{"question_label":"Xem xét giảm hoặc trừ tiền phạt","actor_refs":["Công ty A"],"act_span":"xả thẳng nước thải chưa xử lý ra sông","condition_refs":["làm từ thiện hàng tỷ đồng","nộp thuế đủ"],"legal_axis":"hanh_chinh","user_asked_text":"thành tích từ thiện có được trừ vào tiền phạt không","intent":"trach_nhiem_hanh_chinh"},{"question_label":"Điều kiện miễn trách nhiệm hình sự của nhân viên","actor_refs":["Nhân viên","Giám đốc"],"act_span":"xả thẳng nước thải chưa xử lý ra sông","condition_refs":["làm theo chỉ đạo cấp trên"],"legal_axis":"hinh_su","user_asked_text":"nhân viên làm theo chỉ đạo cấp trên thì có được miễn trách nhiệm hình sự không","intent":"trach_nhiem_hinh_su"}]}
Few-shot 2:
Input: Anh A hiếp dâm chị N nhiều lần nhưng anh A mới 15 tuổi, chị N 20 tuổi. Hỏi ai vi phạm ai và điều kiện tuổi ảnh hưởng thế nào?
Output: {"needs_decomposition":true,"main_act":"hiếp dâm","surface_actors":[{"name":"Anh A","actor_type":"nguoi_thuc_hien_duoc_neu","source_span":"Anh A"},{"name":"chị N","actor_type":"nguoi_khac","source_span":"chị N"}],"conditions":[{"kind":"tuoi","value":"15 tuổi","source_span":"15 tuổi"},{"kind":"tuoi","value":"20 tuổi","source_span":"20 tuổi"},{"kind":"so_lan","value":"nhiều lần","source_span":"nhiều lần"}],"explicit_legal_axes":["hinh_su"],"question_focus":"xac_dinh_ai_vi_pham_ai","clause_units":[],"named_law_hints":[],"question_units":[{"question_label":"Xác định chủ thể thực hiện hành vi","actor_refs":["Anh A","chị N"],"act_span":"hiếp dâm","condition_refs":[],"legal_axis":"hinh_su","user_asked_text":"ai vi phạm ai","intent":"xac_dinh_chu_the"},{"question_label":"Điều kiện về độ tuổi","actor_refs":["Anh A"],"act_span":"hiếp dâm","condition_refs":["15 tuổi","20 tuổi","nhiều lần"],"legal_axis":"hinh_su","user_asked_text":"điều kiện tuổi ảnh hưởng thế nào","intent":"dieu_kien_ap_dung"}]}
Few-shot 3:
Input: Anh tài xế B lái xe tải của công ty TNHH C giao hàng, do mất phanh tông hỏng nhà dân. Xin hỏi công ty C hay anh B phải bồi thường?
Output: {"needs_decomposition":true,"main_act":"tông hỏng nhà dân","surface_actors":[{"name":"Anh tài xế B","actor_type":"nhan_vien","source_span":"Anh tài xế B"},{"name":"công ty TNHH C","actor_type":"to_chuc","source_span":"công ty TNHH C"}],"conditions":[{"kind":"quan_he_lao_dong","value":"lái xe tải của công ty TNHH C giao hàng","source_span":"lái xe tải của công ty TNHH C giao hàng"},{"kind":"hau_qua","value":"mất phanh","source_span":"mất phanh"}],"explicit_legal_axes":["boi_thuong"],"question_focus":"chu_the_chiu_trach_nhiem","clause_units":[],"named_law_hints":[],"question_units":[{"question_label":"Xác định chủ thể bồi thường","actor_refs":["Anh tài xế B","công ty TNHH C"],"act_span":"tông hỏng nhà dân","condition_refs":["lái xe tải của công ty TNHH C giao hàng","mất phanh"],"legal_axis":"boi_thuong","user_asked_text":"công ty C hay anh B phải bồi thường","intent":"boi_thuong"}]}
Few-shot 4:
Input: Mở quán karaoke xả rác bừa bãi ra hè phố bị phạt hành chính bao nhiêu tiền?
Output: {"needs_decomposition":false,"main_act":"xả rác bừa bãi ra hè phố","surface_actors":[{"name":"Chủ quán karaoke","actor_type":"khong_ro","source_span":"Mở quán karaoke"}],"conditions":[{"kind":"khac","value":"quán karaoke","source_span":"Mở quán karaoke"}],"explicit_legal_axes":["hanh_chinh"],"question_focus":"muc_che_tai","clause_units":[],"named_law_hints":[],"question_units":[{"question_label":"Mức phạt hành chính","actor_refs":["Chủ quán karaoke"],"act_span":"xả rác bừa bãi ra hè phố","condition_refs":["quán karaoke"],"legal_axis":"hanh_chinh","user_asked_text":"bị phạt hành chính bao nhiêu tiền","intent":"trach_nhiem_hanh_chinh"}]}

Câu hỏi cần xử lý:
__USER_QUERY__"""
        return template.replace("__USER_QUERY__", user_query)

    def _call_llm(self, prompt: str) -> str:
        client = RemoteLLMClient(provider_type=self.provider_type, base_url=self.base_url, model=self.model_name, timeout_s=self.timeout_s)
        return client.generate_prompt(prompt=prompt, task="rewrite", format_schema=self.EXTRACTION_SCHEMA)

    def _default_actor_type(self, name: str) -> str:
        normalized = self._normalize_key(name)
        if any(token in normalized for token in ("cong ty", "doanh nghiep", "to chuc", "phap nhan", "ho kinh doanh")):
            return "to_chuc"
        if any(token in normalized for token in ("giam doc", "quan ly", "cap tren")):
            return "quan_ly"
        if any(token in normalized for token in ("nhan vien", "nguoi lao dong", "tai xe")):
            return "nhan_vien"
        if re.search(r"\b(anh|chi|ong|ba|co|em)\s+[a-z]\b", normalized):
            return "ca_nhan"
        return "khong_ro"

    def _clean_actor_span(self, value: str) -> str:
        text = self._normalize_text(value)
        lowered = self._normalize_key(text)
        cut = len(text)
        for marker in (" nam nao", " năm nào", " do ", " vi ", " vì ", " xin hoi", " xin hỏi", " nhung ", " nhưng ", " thi ", " thì "):
            pos = lowered.find(self._normalize_key(marker))
            if pos > 0:
                cut = min(cut, pos)
        return text[:cut].strip(" ,.;:-")

    def _heuristic_surface_actors(self, query: str) -> List[SurfaceActor]:
        actors: List[SurfaceActor] = []
        seen = set()
        patterns = (
            r"\b(?:Anh|anh|Chị|chị|Chi|chi|Ông|ông|Bà|bà|Cô|cô|Em|em)\s+[A-Za-zĐđ]\b",
            r"\b(?:Công ty|Cong ty|Doanh nghiệp|Doanh nghiep|Hộ kinh doanh|Ho kinh doanh|Công ty TNHH|Cong ty TNHH|Công ty cổ phần|Cong ty co phan)\s+[A-Za-z0-9Đđ][^,.;:?]{0,40}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                value = self._clean_actor_span(match.group(0))
                key = self._normalize_key(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                actors.append(SurfaceActor(name=value, actor_type=self._default_actor_type(value), source_span=value))
        normalized = self._normalize_key(query)
        for label, actor_type in ACTOR_HINTS:
            if self._normalize_key(label) in normalized and self._normalize_key(label) not in seen:
                seen.add(self._normalize_key(label))
                actors.append(SurfaceActor(name=label, actor_type=actor_type, source_span=label))
        return actors[:6]

    def _heuristic_conditions(self, query: str) -> List[KeyCondition]:
        conditions: List[KeyCondition] = []
        seen = set()
        patterns = [
            ("tuoi", "\\b\\d+\\s*(?:tu\\u1ed5i|tuoi)\\b"),
            ("so_tien", "\\b\\d[\\d\\.,]*\\s*(?:\\u0111\\u1ed3ng|dong|vnd|tri\\u1ec7u|trieu|t\\u1ef7|ty|ngh\\u00ecn|nghin)\\b"),
            ("so_tien", "\\b(?:h\\u00e0ng t\\u1ef7 \\u0111\\u1ed3ng|hang ty dong)\\b"),
            ("thoi_diem", "\\b(?:ng\\u00e0y|ngay)\\s+\\d{1,2}/\\d{1,2}/\\d{4}\\b"),
            ("thoi_diem", "\\b(?:n\\u0103m|nam)\\s+\\d{4}\\b"),
            ("thoi_diem", "\\b(?:th\\u00e1ng|thang)\\s+\\d{1,2}/\\d{4}\\b"),
            ("thoi_diem", "\\b(?:trong|sau)\\s+\\d+\\s*(?:ng\\u00e0y|ngay|th\\u00e1ng|thang|n\\u0103m|nam)\\b"),
            ("so_lan", "\\b(?:nhi\\u1ec1u l\\u1ea7n|nhieu lan)\\b"),
            ("so_lan", "\\b\\d+\\s*l\\u1ea7n\\b"),
            ("hau_qua", "\\b(?:ch\\u1ebft ng\\u01b0\\u1eddi|chet nguoi|thi\\u1ec7t h\\u1ea1i|thiet hai|\\u00f4 nhi\\u1ec5m|o nhiem|b\\u1ecb th\\u01b0\\u01a1ng|bi thuong|th\\u01b0\\u01a1ng t\\u00edch|thuong tich|th\\u01b0\\u01a1ng t\\u1eadt|thuong tat|m\\u1ea5t kh\\u1ea3 n\\u0103ng lao \\u0111\\u1ed9ng|mat kha nang lao dong|s\\u1eadp c\\u00f4ng tr\\u00ecnh|sap cong trinh|\\u00f4 nhi\\u1ec5m ngu\\u1ed3n n\\u01b0\\u1edbc|o nhiem nguon nuoc)\\b"),
            ("quan_he_lao_dong", "\\b(?:l\\u00e0m theo ch\\u1ec9 \\u0111\\u1ea1o(?: c\\u1ea5p tr\\u00ean)?|lam theo chi dao(?: cap tren)?|\\u00e9p bu\\u1ed9c|ep buoc|d\\u1ecda \\u0111u\\u1ed5i vi\\u1ec7c|doa duoi viec|s\\u1ee3 m\\u1ea5t vi\\u1ec7c|so mat viec|h\\u1ee3p \\u0111\\u1ed3ng lao \\u0111\\u1ed9ng|hop dong lao dong|th\\u1eed vi\\u1ec7c|thu viec|sa th\\u1ea3i|sa thai|\\u0111\\u01a1n ph\\u01b0\\u01a1ng ch\\u1ea5m d\\u1ee9t|don phuong cham dut)\\b"),
            ("khac", "\\b(?:m\\u1eb9 \\u0111\\u01a1n th\\u00e2n|me don than|n\\u1ed9p thu\\u1ebf(?: \\u0111\\u1ee7)?|nop thue(?: du)?|l\\u00e0m t\\u1eeb thi\\u1ec7n(?: h\\u00e0ng t\\u1ef7 \\u0111\\u1ed3ng)?|lam tu thien(?: hang ty dong)?|m\\u00e1y l\\u1ecdc h\\u1ecfng|may loc hong|kh\\u00f4ng c\\u00f3 gi\\u1ea5y ph\\u00e9p|khong co giay phep|kh\\u00f4ng \\u0111\\u0103ng k\\u00fd|khong dang ky|kh\\u00f4ng b\\u1ea3o hi\\u1ec3m|khong bao hiem|\\u0111\\u00e3 kh\\u1eafc ph\\u1ee5c|da khac phuc|\\u0111\\u00e3 b\\u1ed3i th\\u01b0\\u1eddng|da boi thuong)\\b"),
        ]
        for kind, pattern in patterns:
            for match in re.finditer(pattern, query, flags=re.IGNORECASE):
                value = self._normalize_text(match.group(0))
                key = (kind, self._normalize_key(value))
                if key[1] and key not in seen:
                    seen.add(key)
                    conditions.append(KeyCondition(kind=kind, value=value, source_span=value))
        return conditions[:8]

    def _heuristic_named_law_hints(self, query: str) -> List[str]:
        hints: List[str] = []
        seen = set()
        for pattern in LAW_HINT_PATTERNS:
            for match in re.finditer(pattern, query):
                value = self._normalize_text(match.group(1)).rstrip(" ,.;:?!")
                key = self._normalize_key(value)
                if key and key not in seen:
                    seen.add(key)
                    hints.append(value)
                    if len(hints) >= 4:
                        return hints
        return hints

    def _heuristic_explicit_axes(self, query: str) -> List[str]:
        normalized = self._normalize_key(query)
        axes: List[str] = []
        for tag, keywords in {
            "hinh_su": ("hinh su", "trach nhiem hinh su", "xu ly hinh su", "mien trach nhiem hinh su"),
            "dan_su": ("dan su", "trach nhiem dan su"),
            "hanh_chinh": ("hanh chinh", "xu phat", "tien phat", "muc phat", "tru vao tien phat"),
            "boi_thuong": ("boi thuong", "den tien", "boi hoan"),
        }.items():
            if any(self._normalize_key(keyword) in normalized for keyword in keywords):
                axes.append(tag)
        if "boi_thuong" in axes and "dan_su" not in axes:
            axes.append("dan_su")
        return axes[:4]

    def _heuristic_question_focus(self, query: str) -> str:
        normalized = self._normalize_key(query)
        for focus, keywords in QUESTION_FOCUS_HINTS:
            if any(self._normalize_key(keyword) in normalized for keyword in keywords):
                return focus
        return DEFAULT_QUESTION_FOCUS

    def _heuristic_main_act(self, query: str) -> str:
        normalized = self._normalize_key(query)
        for canonical, aliases in ACTION_ALIASES:
            if any(self._normalize_key(alias) in normalized for alias in aliases):
                return canonical
        return ""

    def _extract_question_segments(self, query: str) -> List[str]:
        text = self._normalize_text(query)
        if not text:
            return []
        chunks = [item.strip() for item in re.split(r"\?+", text) if item.strip()]
        segments: List[str] = []
        seen = set()
        for chunk in chunks:
            value = chunk
            for marker in ("Xin hỏi", "xin hỏi", "Hỏi:", "hỏi:", "Hỏi", "hỏi", "Hoi:", "hoi:", "Hoi", "hoi"):
                if marker in value:
                    value = value.split(marker, 1)[1].strip()
            value = re.sub(r"^[\-\s:;,]+", "", value).strip()
            normalized = self._normalize_key(value)
            if not normalized:
                continue
            if not any(token in normalized for token in ("co duoc", "duoc mien", "mien trach nhiem", "boi thuong", "hinh su", "dan su", "hanh chinh", "ai vi pham ai", "hay ca hai", "thuoc ve", "tien phat", "muc phat", "dieu kien", "bao nhieu tuoi", "do tuoi", "den tien", "chiu trach nhiem")):
                continue
            if normalized not in seen:
                seen.add(normalized)
                segments.append(value)
        return segments[:4]

    def _canonical_actor_ref(self, value: str, actors: Sequence[SurfaceActor]) -> str:
        target = self._normalize_key(value)
        for actor in actors:
            if self._normalize_key(actor.name) == target:
                return actor.name
        return self._normalize_text(value)

    def _canonical_condition_ref(self, value: str, conditions: Sequence[KeyCondition]) -> str:
        target = self._normalize_key(value)
        for condition in conditions:
            if self._normalize_key(condition.value) == target:
                return condition.value
        return self._normalize_text(value)

    def _condition_matches_question(self, condition: KeyCondition, question_text: str) -> bool:
        normalized_question = self._normalize_key(question_text)
        normalized_value = self._normalize_key(condition.value)
        if normalized_value and normalized_value in normalized_question:
            return True
        if condition.kind == "tuoi" and any(token in normalized_question for token in ("tuoi", "do tuoi", "duoi 16", "duoi 18")):
            return True
        if condition.kind == "so_tien" and any(token in normalized_question for token in ("tien phat", "muc phat", "bao nhieu tien", "so tien", "muc boi thuong", "gia tri")):
            return True
        if condition.kind == "quan_he_lao_dong" and any(token in normalized_question for token in ("chi dao", "cap tren", "nhan vien", "nguoi lao dong", "duoi viec", "mien trach nhiem")):
            return True
        if condition.kind == "khac" and any(token in normalized_value for token in ("tu thien", "nop thue")) and any(token in normalized_question for token in ("tru vao tien phat", "giam phat", "tien phat")):
            return True
        return False

    def _select_actor_refs_for_question(self, question_text: str, actors: Sequence[SurfaceActor]) -> List[str]:
        normalized_question = self._normalize_key(question_text)
        selected: List[str] = []
        for actor in actors:
            actor_key = self._normalize_key(actor.name)
            if actor_key and actor_key in normalized_question:
                selected.append(actor.name)
        if selected:
            return selected[:3]
        if any(token in normalized_question for token in ("tien phat", "muc phat", "giam phat", "tru vao tien phat")):
            for actor in actors:
                if actor.actor_type == "to_chuc" and actor.name not in selected:
                    selected.append(actor.name)
            if selected:
                return selected[:2]
        if any(token in normalized_question for token in ("nhan vien", "nguoi lao dong", "tai xe")):
            for actor in actors:
                if actor.actor_type == "nhan_vien":
                    selected.append(actor.name)
        if any(token in normalized_question for token in ("giam doc", "cap tren", "quan ly")):
            for actor in actors:
                if actor.actor_type == "quan_ly" and actor.name not in selected:
                    selected.append(actor.name)
        if any(token in normalized_question for token in ("cong ty", "doanh nghiep", "phap nhan", "to chuc")):
            for actor in actors:
                if actor.actor_type == "to_chuc" and actor.name not in selected:
                    selected.append(actor.name)
        if selected:
            return selected[:3]
        return [actor.name for actor in actors[:2]]

    def _select_condition_refs_for_question(self, question_text: str, conditions: Sequence[KeyCondition]) -> List[str]:
        refs: List[str] = []
        for condition in conditions:
            if self._condition_matches_question(condition, question_text):
                refs.append(condition.value)
        return refs[:4]

    def _infer_intent_and_axis(self, question_text: str, default_axes: Sequence[str]) -> Tuple[str, str]:
        normalized = self._normalize_key(question_text)
        if any(token in normalized for token in ("boi thuong", "den tien")):
            return "boi_thuong", "boi_thuong"
        if any(token in normalized for token in ("hanh chinh", "tien phat", "muc phat", "tru vao tien phat", "giam phat")):
            return "trach_nhiem_hanh_chinh", "hanh_chinh"
        if any(token in normalized for token in ("mien trach nhiem hinh su", "trach nhiem hinh su", "hinh su")):
            return "trach_nhiem_hinh_su", "hinh_su"
        if any(token in normalized for token in ("dan su", "trach nhiem dan su")):
            return "trach_nhiem_dan_su", "dan_su"
        if any(token in normalized for token in ("dieu kien", "bao nhieu tuoi", "do tuoi")):
            axis = default_axes[0] if default_axes else ""
            return "dieu_kien_ap_dung", axis
        if any(token in normalized for token in ("ai vi pham ai", "thuoc ve ai", "hay ca hai", "ai chiu")):
            axis = default_axes[0] if default_axes else ""
            return "xac_dinh_chu_the", axis
        axis = default_axes[0] if default_axes else ""
        if axis == "boi_thuong":
            return "boi_thuong", axis
        if axis == "hanh_chinh":
            return "trach_nhiem_hanh_chinh", axis
        if axis == "hinh_su":
            return "trach_nhiem_hinh_su", axis
        if axis == "dan_su":
            return "trach_nhiem_dan_su", axis
        return "xac_dinh_chu_the", axis

    def _build_question_units_from_segment(
        self,
        segment: str,
        main_act: str,
        actors: Sequence[SurfaceActor],
        conditions: Sequence[KeyCondition],
        axes: Sequence[str],
    ) -> QuestionUnit:
        intent, legal_axis = self._infer_intent_and_axis(segment, axes)
        actor_refs = self._select_actor_refs_for_question(segment, actors)
        condition_refs = self._select_condition_refs_for_question(segment, conditions)
        label = self._normalize_text(segment)
        if len(label) > 96:
            label = label[:96].rstrip(" ,.;:")
        if not label:
            label = "Y hoi phap ly"
        return QuestionUnit(
            question_label=label,
            actor_refs=actor_refs,
            act_span=main_act,
            condition_refs=condition_refs,
            legal_axis=legal_axis,
            user_asked_text=self._normalize_text(segment),
            intent=intent,
        )

    def _heuristic_question_units(
        self,
        query: str,
        main_act: str,
        actors: Sequence[SurfaceActor],
        conditions: Sequence[KeyCondition],
        axes: Sequence[str],
    ) -> List[QuestionUnit]:
        segments = self._extract_question_segments(query)
        units = [self._build_question_units_from_segment(segment, main_act, actors, conditions, axes) for segment in segments]
        if units:
            return units[:3]
        fallback_segment = query if len(query) <= 160 else query[:160].rstrip(" ,.;:")
        return [self._build_question_units_from_segment(fallback_segment, main_act, actors, conditions, axes)]

    def _parse_surface_actors(self, raw_actors: object) -> List[SurfaceActor]:
        actors: List[SurfaceActor] = []
        if not isinstance(raw_actors, list):
            return actors
        for item in raw_actors:
            if not isinstance(item, dict):
                continue
            name = self._normalize_text(item.get("name", ""))
            source_span = self._normalize_text(item.get("source_span", "")) or name
            actor_type = self._normalize_text(item.get("actor_type", "khong_ro")) or "khong_ro"
            if not name:
                continue
            if actor_type not in ALLOWED_ACTOR_TYPES:
                actor_type = self._default_actor_type(name)
            actors.append(SurfaceActor(name=name, actor_type=actor_type, source_span=source_span))
        return actors[:6]

    def _parse_conditions(self, raw_conditions: object) -> List[KeyCondition]:
        conditions: List[KeyCondition] = []
        if not isinstance(raw_conditions, list):
            return conditions
        for item in raw_conditions:
            if not isinstance(item, dict):
                continue
            kind = self._normalize_text(item.get("kind", "khac")) or "khac"
            value = self._normalize_text(item.get("value", ""))
            source_span = self._normalize_text(item.get("source_span", "")) or value
            if not value:
                continue
            if kind not in ALLOWED_CONDITION_KINDS:
                kind = "khac"
            conditions.append(KeyCondition(kind=kind, value=value, source_span=source_span))
        return conditions[:8]

    def _parse_named_law_hints(self, raw_hints: object) -> List[str]:
        hints: List[str] = []
        seen = set()
        if not isinstance(raw_hints, list):
            return hints
        for value in raw_hints:
            text = self._normalize_text(value)
            key = self._normalize_key(text)
            if text and key and key not in seen:
                seen.add(key)
                hints.append(text)
        return hints[:4]

    def _parse_question_units(
        self,
        raw_units: object,
        main_act: str,
        actors: Sequence[SurfaceActor],
        conditions: Sequence[KeyCondition],
        axes: Sequence[str],
    ) -> List[QuestionUnit]:
        units: List[QuestionUnit] = []
        if not isinstance(raw_units, list):
            return units
        for item in raw_units:
            if not isinstance(item, dict):
                continue
            question_label = self._normalize_text(item.get("question_label", ""))
            act_span = self._normalize_text(item.get("act_span", "")) or main_act
            legal_axis = self._normalize_text(item.get("legal_axis", ""))
            if legal_axis and legal_axis not in ALLOWED_ISSUE_TAGS:
                legal_axis = ""
            user_asked_text = self._normalize_text(item.get("user_asked_text", ""))
            intent = self._normalize_text(item.get("intent", "xac_dinh_chu_the")) or "xac_dinh_chu_the"
            if intent not in ALLOWED_CLAUSE_INTENTS:
                intent, fallback_axis = self._infer_intent_and_axis(user_asked_text or question_label, axes)
                legal_axis = legal_axis or fallback_axis
            actor_refs = []
            for actor_ref in item.get("actor_refs", []) if isinstance(item.get("actor_refs", []), list) else []:
                canonical = self._canonical_actor_ref(str(actor_ref), actors)
                if canonical and canonical not in actor_refs:
                    actor_refs.append(canonical)
            condition_refs = []
            for condition_ref in item.get("condition_refs", []) if isinstance(item.get("condition_refs", []), list) else []:
                canonical = self._canonical_condition_ref(str(condition_ref), conditions)
                if canonical and canonical not in condition_refs:
                    condition_refs.append(canonical)
            if not user_asked_text:
                user_asked_text = question_label
            if not question_label:
                question_label = user_asked_text or "Y hoi phap ly"
            if not act_span:
                act_span = main_act
            if not legal_axis and axes:
                legal_axis = axes[0]
            if not actor_refs:
                actor_refs = self._select_actor_refs_for_question(user_asked_text or question_label, actors)
            if condition_refs:
                filtered_condition_refs = []
                for value in condition_refs:
                    matched_condition = next(
                        (condition for condition in conditions if self._normalize_key(condition.value) == self._normalize_key(value)),
                        KeyCondition(kind="khac", value=value),
                    )
                    if self._condition_matches_question(matched_condition, user_asked_text or question_label):
                        filtered_condition_refs.append(value)
                if filtered_condition_refs:
                    condition_refs = filtered_condition_refs
            if not condition_refs:
                condition_refs = self._select_condition_refs_for_question(user_asked_text or question_label, conditions)
            units.append(QuestionUnit(
                question_label=question_label,
                actor_refs=actor_refs[:3],
                act_span=act_span,
                condition_refs=condition_refs[:4],
                legal_axis=legal_axis,
                user_asked_text=user_asked_text,
                intent=intent,
            ))
        return units[:3]

    def _merge_surface_actors(self, preferred: Sequence[SurfaceActor], fallback: Sequence[SurfaceActor]) -> List[SurfaceActor]:
        merged: List[SurfaceActor] = []
        seen = set()
        for group in (preferred, fallback):
            for actor in group:
                key = self._normalize_key(actor.name)
                if not key:
                    continue
                replaced = False
                for idx, existing in enumerate(list(merged)):
                    existing_key = self._normalize_key(existing.name)
                    if not existing_key:
                        continue
                    if actor.actor_type == existing.actor_type and key in existing_key:
                        replaced = True
                        break
                    if actor.actor_type == existing.actor_type and existing_key in key:
                        merged[idx] = actor
                        seen.discard(existing_key)
                        seen.add(key)
                        replaced = True
                        break
                if replaced:
                    continue
                if key not in seen:
                    seen.add(key)
                    merged.append(actor)
        return merged[:6]

    def _merge_conditions(self, preferred: Sequence[KeyCondition], fallback: Sequence[KeyCondition]) -> List[KeyCondition]:
        merged: List[KeyCondition] = []
        seen = set()
        for group in (preferred, fallback):
            for condition in group:
                key = (condition.kind, self._normalize_key(condition.value))
                if key[1] and key not in seen:
                    seen.add(key)
                    merged.append(condition)
        return merged[:8]

    def _merge_simple_lists(self, preferred: Sequence[str], fallback: Sequence[str], limit: int) -> List[str]:
        values: List[str] = []
        seen = set()
        for group in (preferred, fallback):
            for item in group:
                text = self._normalize_text(item)
                key = self._normalize_key(text)
                if text and key and key not in seen:
                    seen.add(key)
                    values.append(text)
        return values[:limit]

    def _merge_question_units(self, preferred: Sequence[QuestionUnit], fallback: Sequence[QuestionUnit]) -> List[QuestionUnit]:
        merged: List[QuestionUnit] = []
        seen = set()
        for group in (preferred, fallback):
            for unit in group:
                key = "|".join(
                    [
                        self._normalize_key(unit.intent),
                        self._normalize_key(unit.legal_axis),
                        self._normalize_key(unit.act_span),
                        ",".join(sorted(self._normalize_key(actor) for actor in unit.actor_refs)),
                        ",".join(sorted(self._normalize_key(condition) for condition in unit.condition_refs)),
                    ]
                )
                if key and key not in seen:
                    seen.add(key)
                    merged.append(unit)
        return merged[:3]

    def _augment_question_units(
        self,
        units: Sequence[QuestionUnit],
        question_focus: str,
        main_act: str,
        actors: Sequence[SurfaceActor],
        conditions: Sequence[KeyCondition],
        axes: Sequence[str],
    ) -> List[QuestionUnit]:
        merged = list(units)
        if question_focus == "xac_dinh_ai_vi_pham_ai":
            merged = [
                unit
                for unit in merged
                if not (
                    unit.intent == "trach_nhiem_hinh_su"
                    and len((unit.user_asked_text or unit.question_label).split()) >= 12
                )
            ]
        age_conditions = [condition.value for condition in conditions if condition.kind == "tuoi"]
        if age_conditions and not any(unit.intent == "dieu_kien_ap_dung" for unit in merged):
            axis = axes[0] if axes else ""
            primary_actor = actors[0].name if actors else ""
            merged.append(
                QuestionUnit(
                    question_label="Điều kiện về độ tuổi áp dụng",
                    actor_refs=[primary_actor] if primary_actor else [],
                    act_span=main_act,
                    condition_refs=age_conditions[:4],
                    legal_axis=axis,
                    user_asked_text="điều kiện về độ tuổi áp dụng",
                    intent="dieu_kien_ap_dung",
                )
            )
        return merged[:3]

    def _is_generic_main_act(self, value: str) -> bool:
        normalized = self._normalize_key(value)
        if not normalized:
            return True
        return any(token in normalized for token in GENERIC_MAIN_ACT_PATTERNS)

    def _contains_action_signal(self, value: str) -> bool:
        normalized = self._normalize_key(value)
        if not normalized:
            return False
        if any(token in normalized for token in ("xa", "hiep dam", "gay tai nan", "tong hong", "danh", "lua dao", "trom", "giat")):
            return True
        return any(self._normalize_key(alias) in normalized for _, aliases in ACTION_ALIASES for alias in aliases)

    def _select_main_act(self, raw_main_act: str, query: str) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        normalized_raw = self._normalize_text(raw_main_act)
        if normalized_raw and not self._is_generic_main_act(normalized_raw) and self._contains_action_signal(normalized_raw):
            reasons.append("llm_main_act")
            return normalized_raw, reasons
        heuristic = self._heuristic_main_act(query)
        if heuristic:
            reasons.append("heuristic_action_alias")
            return heuristic, reasons
        if normalized_raw:
            reasons.append("llm_main_act_rejected_generic")
        return "", reasons

    def _normalize_axes(self, preferred: Sequence[str], fallback: Sequence[str]) -> List[str]:
        axes: List[str] = []
        seen = set()
        for group in (preferred, fallback):
            for axis in group:
                value = self._normalize_key(axis)
                if value in ALLOWED_ISSUE_TAGS and value not in seen:
                    seen.add(value)
                    axes.append(value)
        return axes[:4]

    def _build_clause_units_from_question_units(self, units: Sequence[QuestionUnit]) -> List[ClauseUnit]:
        clause_units: List[ClauseUnit] = []
        for unit in units:
            clause_units.append(ClauseUnit(
                clause_label=unit.question_label,
                actor_refs=list(unit.actor_refs),
                act_span=unit.act_span,
                condition_refs=list(unit.condition_refs),
                intent=unit.intent if unit.intent in ALLOWED_CLAUSE_INTENTS else "xac_dinh_chu_the",
            ))
        return clause_units[:6]

    def _parse_extraction(self, raw_output: str, user_query: str) -> StructuredRewriteExtraction:
        payload: Dict[str, object] = {}
        clean_output = str(raw_output or "").strip()
        clean_output = re.sub(r"^\s*```(?:json)?\s*", "", clean_output, flags=re.IGNORECASE)
        clean_output = re.sub(r"\s*```\s*$", "", clean_output, flags=re.IGNORECASE)
        text = self._normalize_text(clean_output)
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", clean_output, flags=re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        payload = {}
        heuristic_actors = self._heuristic_surface_actors(user_query)
        heuristic_conditions = self._heuristic_conditions(user_query)
        heuristic_axes = self._heuristic_explicit_axes(user_query)
        heuristic_focus = self._heuristic_question_focus(user_query)
        heuristic_hints = self._heuristic_named_law_hints(user_query)

        llm_actors = self._parse_surface_actors(payload.get("surface_actors", []))
        llm_conditions = self._parse_conditions(payload.get("conditions", []))
        llm_hints = self._parse_named_law_hints(payload.get("named_law_hints", []))
        axes = self._normalize_axes(payload.get("explicit_legal_axes", []) if isinstance(payload.get("explicit_legal_axes", []), list) else [], heuristic_axes)
        main_act, reasons = self._select_main_act(str(payload.get("main_act", "")), user_query)
        question_focus = self._normalize_text(payload.get("question_focus", "")) or heuristic_focus
        if question_focus not in {"nhieu_y_hoi", "xac_dinh_ai_vi_pham_ai", "chu_the_chiu_trach_nhiem", "muc_che_tai", "dieu_kien_ap_dung", "khai_niem_dinh_nghia"}:
            question_focus = heuristic_focus

        actors = self._merge_surface_actors(llm_actors, heuristic_actors)
        conditions = self._merge_conditions(llm_conditions, heuristic_conditions)
        named_law_hints = self._merge_simple_lists(llm_hints, heuristic_hints, 4)
        llm_units = self._parse_question_units(payload.get("question_units", []), main_act, actors, conditions, axes)
        heuristic_units = self._heuristic_question_units(user_query, main_act or self._heuristic_main_act(user_query), actors, conditions, axes)
        question_units = self._merge_question_units(llm_units, heuristic_units)
        question_units = self._augment_question_units(
            units=question_units,
            question_focus=question_focus,
            main_act=main_act,
            actors=actors,
            conditions=conditions,
            axes=axes,
        )
        clause_units = self._build_clause_units_from_question_units(question_units)
        needs_decomposition = bool(payload.get("needs_decomposition")) or len(question_units) >= 2 or self._should_rewrite(user_query)

        return StructuredRewriteExtraction(
            needs_decomposition=needs_decomposition,
            main_act=main_act,
            surface_actors=actors,
            conditions=conditions,
            explicit_legal_axes=axes,
            question_focus=question_focus,
            clause_units=clause_units,
            named_law_hints=named_law_hints,
            question_units=question_units,
            main_act_candidate_reasons=reasons,
        )

    def _anchor_snippet(self, query: str, main_act: str) -> str:
        text = self._normalize_text(query)
        if not text:
            return ""
        if main_act:
            normalized = self._normalize_key(text)
            main_key = self._normalize_key(main_act)
            idx = normalized.find(main_key)
            if idx >= 0:
                snippet = text[idx: idx + max(len(main_act) + 72, 100)].strip(' ,.;:')
                return snippet
        return text[:120].strip(' ,.;:')

    def _question_unit_hint_matches(self, unit: QuestionUnit, hint: str) -> bool:
        unit_text = self._normalize_key(" ".join([unit.question_label, unit.user_asked_text, " ".join(unit.condition_refs)]))
        hint_key = self._normalize_key(hint)
        if not hint_key:
            return False
        if "luat lao dong" in hint_key:
            return any(token in unit_text for token in ("chi dao", "nhan vien", "nguoi lao dong", "mien trach nhiem"))
        return hint_key in unit_text

    def _append_named_law_hint(self, query: str, unit: QuestionUnit, named_law_hints: Sequence[str]) -> str:
        if not named_law_hints:
            return query
        for hint in named_law_hints:
            if self._question_unit_hint_matches(unit, hint):
                candidate = f"{query}; người dùng nêu thêm {hint}"
                if len(candidate) <= 280:
                    return candidate
        return query

    def _build_query_for_question_unit(
        self,
        unit: QuestionUnit,
        main_act: str,
        actors: Sequence[SurfaceActor],
        conditions: Sequence[KeyCondition],
        named_law_hints: Sequence[str],
        original_query: str,
    ) -> str:
        act = self._normalize_text(unit.act_span or main_act)
        actor_text = ", ".join([self._normalize_text(actor) for actor in unit.actor_refs if self._normalize_text(actor)])
        condition_text = ", ".join([
            self._normalize_text(condition)
            for condition in unit.condition_refs
            if self._normalize_text(condition)
        ])
        axis_label = ISSUE_LABELS.get(unit.legal_axis, "")
        intent = unit.intent
        asked = self._normalize_text(unit.user_asked_text or unit.question_label)

        if intent == "trach_nhiem_hanh_chinh" and any(token in self._normalize_key(asked) for token in ("tru vao tien phat", "giam phat", "tien phat", "muc phat")):
            query = f"Tình tiết được xem xét giảm hoặc trừ tiền phạt đối với hành vi {act}"
            if actor_text:
                query += f" của {actor_text}"
            if condition_text:
                query += f"; tình tiết được nêu: {condition_text}"
        elif intent == "trach_nhiem_hinh_su" and any(token in self._normalize_key(asked) for token in ("mien trach nhiem", "co duoc mien")):
            query = f"Điều kiện xem xét miễn hoặc giảm {axis_label or 'trách nhiệm hình sự'} đối với {actor_text or 'chủ thể được nêu'} khi {act}"
            if condition_text:
                query += f"; điều kiện: {condition_text}"
        elif intent == "dieu_kien_ap_dung":
            query = f"Điều kiện áp dụng đối với hành vi {act}"
            if actor_text:
                query += f" của {actor_text}"
            if condition_text:
                query += f": {condition_text}"
        elif intent == "boi_thuong":
            query = f"Chủ thể và căn cứ {ISSUE_LABELS.get('boi_thuong')} đối với hành vi {act}"
            if actor_text:
                query += f"; chủ thể liên quan: {actor_text}"
            if condition_text:
                query += f"; điều kiện: {condition_text}"
        elif intent == "xac_dinh_chu_the":
            query = f"Xác định chủ thể liên quan đến hành vi {act}"
            if actor_text:
                query += f"; các chủ thể được nêu: {actor_text}"
            if condition_text:
                query += f"; tình tiết: {condition_text}"
        else:
            query = f"{unit.question_label}: {act}"
            if actor_text:
                query += f"; chủ thể: {actor_text}"
            if condition_text:
                query += f"; điều kiện: {condition_text}"

        if not act:
            query = asked or self._anchor_snippet(original_query, main_act)
        query = self._append_named_law_hint(query, unit, named_law_hints)
        return self._normalize_text(query)

    def _synthesize_query_plan(
        self,
        extraction: StructuredRewriteExtraction,
        original_query: str,
    ) -> List[Dict[str, object]]:
        plan: List[Dict[str, object]] = []
        for index, unit in enumerate(extraction.question_units[: self.max_subqueries], start=1):
            query = self._build_query_for_question_unit(
                unit=unit,
                main_act=extraction.main_act,
                actors=extraction.surface_actors,
                conditions=extraction.conditions,
                named_law_hints=extraction.named_law_hints,
                original_query=original_query,
            )
            if not query:
                continue
            uses_named_law_hint = any(self._normalize_key(hint) in self._normalize_key(query) for hint in extraction.named_law_hints)
            plan.append({
                "label": unit.question_label,
                "query": query,
                "source_question_unit": unit.question_label,
                "uses_named_law_hint": bool(uses_named_law_hint),
                "priority": index,
            })
        seen = set()
        deduped: List[Dict[str, object]] = []
        for item in plan:
            key = self._normalize_key(str(item.get("query", "")))
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped[: self.max_subqueries]

    def _quality_gate(
        self,
        extraction: StructuredRewriteExtraction,
        query_plan: Sequence[Dict[str, object]],
        original_query: str,
    ) -> List[str]:
        reasons: List[str] = []
        if not extraction.main_act:
            if extraction.surface_actors:
                reasons.append("missing_main_act_for_scenario")
        elif self._is_generic_main_act(extraction.main_act):
            reasons.append("generic_main_act")
        if not query_plan:
            reasons.append("empty_query_plan")
            return reasons

        original_key = self._normalize_key(original_query)
        main_act_key = self._normalize_key(extraction.main_act)
        actor_keys = [self._normalize_key(actor.name) for actor in extraction.surface_actors]
        condition_values = [condition.value for condition in extraction.conditions if condition.kind in {"tuoi", "so_tien", "so_lan", "quan_he_lao_dong"}]
        matched_units = set()

        for item in query_plan:
            query = self._normalize_text(item.get("query", ""))
            query_key = self._normalize_key(query)
            if not query_key:
                reasons.append("blank_query")
                continue
            if main_act_key and main_act_key not in query_key:
                reasons.append("query_missing_main_act")
            if "hanh vi duoc neu" in query_key or "tinh huong duoc neu" in query_key:
                reasons.append("generic_query_template")
            if any(token in query_key for token in DRIFT_TOKENS):
                reasons.append("topic_drift")
            if len(query_key.split()) < 5:
                reasons.append("query_too_short")
            if actor_keys and all(token not in query_key for token in actor_keys if token):
                reasons.append("query_missing_actor_anchor")
            if extraction.named_law_hints:
                hint_hits = sum(1 for hint in extraction.named_law_hints if self._normalize_key(hint) in query_key)
                main_hits = 1 if main_act_key and main_act_key in query_key else 0
                if hint_hits and not main_hits:
                    reasons.append("named_law_override")
            if any(token in query_key for token in ("co toi", "pham toi", "vo toi", "nan nhan", "duoc mien trach nhiem hinh su")):
                if not any(token in query_key for token in ("dieu kien", "xem xet", "co duoc", "co phai")):
                    reasons.append("premature_legal_conclusion")
            if any(self._normalize_key(str(item.get("source_question_unit", ""))) == self._normalize_key(unit.question_label) for unit in extraction.question_units):
                matched_units.add(self._normalize_key(str(item.get("source_question_unit", ""))))

        for value in condition_values:
            value_key = self._normalize_key(value)
            if value_key and not any(value_key in self._normalize_key(str(item.get("query", ""))) for item in query_plan):
                question_sensitive = any(value_key in self._normalize_key(unit.user_asked_text) or value_key in self._normalize_key(" ".join(unit.condition_refs)) for unit in extraction.question_units)
                if question_sensitive:
                    reasons.append("condition_not_preserved")
                    break
        if extraction.question_units and len(matched_units) < len(extraction.question_units[: self.max_subqueries]):
            reasons.append("question_unit_coverage_failed")
        if extraction.main_act and self._normalize_key(extraction.main_act) not in original_key:
            reasons.append("main_act_not_grounded_in_original_query")
        return sorted(set(reasons))

    def rewrite_query(self, user_query: str, force: bool = False) -> QueryRewriteResult:
        original_query = self._normalize_text(user_query)
        started_at = time.perf_counter()
        raw_output = ""

        heuristic_main_act = self._heuristic_main_act(original_query)
        heuristic_actors = self._heuristic_surface_actors(original_query)
        heuristic_conditions = self._heuristic_conditions(original_query)
        heuristic_axes = self._heuristic_explicit_axes(original_query)
        heuristic_focus = self._heuristic_question_focus(original_query)
        heuristic_hints = self._heuristic_named_law_hints(original_query)

        if not force and not self._should_rewrite(original_query):
            extraction = StructuredRewriteExtraction(
                needs_decomposition=False,
                main_act=heuristic_main_act,
                surface_actors=heuristic_actors,
                conditions=heuristic_conditions,
                explicit_legal_axes=heuristic_axes,
                question_focus=heuristic_focus,
                clause_units=[],
                named_law_hints=heuristic_hints,
                question_units=[],
                main_act_candidate_reasons=["rewrite_gate_skipped"],
            )
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            return QueryRewriteResult(
                original_query=original_query,
                queries=[original_query],
                status="skipped",
                raw_response="",
                fallback_reason="complexity_gate",
                latency_ms=latency_ms,
                quality_passed=True,
                retrieval_query_set=[original_query],
                actor_mentions=[actor.name for actor in extraction.surface_actors],
                issue_tags=list(extraction.explicit_legal_axes),
                question_focus=extraction.question_focus,
                needs_decomposition=extraction.needs_decomposition,
                main_act=extraction.main_act,
                surface_actors=[actor.to_public_dict() for actor in extraction.surface_actors],
                conditions=[condition.to_public_dict() for condition in extraction.conditions],
                clause_units=[clause.to_public_dict() for clause in extraction.clause_units],
                named_law_hints=list(extraction.named_law_hints),
                question_units=[unit.to_public_dict() for unit in extraction.question_units],
                retrieval_query_plan=[],
                main_act_candidate_reasons=list(extraction.main_act_candidate_reasons),
            )

        prompt = self._build_prompt(original_query)
        raw_output = self._call_llm(prompt)
        extraction = self._parse_extraction(raw_output, original_query)
        query_plan = self._synthesize_query_plan(extraction, original_query)
        rejection_reasons = self._quality_gate(extraction, query_plan, original_query)
        quality_passed = not rejection_reasons

        retrieval_queries = [str(item.get("query", "")).strip() for item in query_plan if str(item.get("query", "")).strip()]
        if not quality_passed:
            retrieval_queries = []
        retrieval_query_set = list(dict.fromkeys(retrieval_queries + [original_query]))[: self.max_subqueries + 1]
        missing_numbers = sorted(set(self._extract_numbers(original_query)) - set(self._extract_numbers(" ".join(retrieval_query_set))))
        if missing_numbers and quality_passed:
            rejection_reasons = sorted(set(rejection_reasons + ["number_not_preserved"]))
            quality_passed = False
            retrieval_query_set = [original_query]
            query_plan = []

        status = "ok" if quality_passed and retrieval_query_set else "rejected"
        fallback_reason = ""
        if status != "ok":
            status = "rejected"
            fallback_reason = ",".join(rejection_reasons) if rejection_reasons else "quality_gate"
            retrieval_query_set = [original_query]
            query_plan = []

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return QueryRewriteResult(
            original_query=original_query,
            queries=list(retrieval_query_set),
            status=status,
            raw_response=raw_output,
            missing_numbers=missing_numbers,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
            quality_passed=quality_passed,
            retrieval_query_set=list(retrieval_query_set),
            actor_mentions=[actor.name for actor in extraction.surface_actors],
            issue_tags=list(extraction.explicit_legal_axes),
            question_focus=extraction.question_focus,
            needs_decomposition=extraction.needs_decomposition,
            rejection_reasons=list(rejection_reasons),
            main_act=extraction.main_act,
            surface_actors=[actor.to_public_dict() for actor in extraction.surface_actors],
            conditions=[condition.to_public_dict() for condition in extraction.conditions],
            clause_units=[clause.to_public_dict() for clause in extraction.clause_units],
            named_law_hints=list(extraction.named_law_hints),
            question_units=[unit.to_public_dict() for unit in extraction.question_units],
            retrieval_query_plan=list(query_plan),
            main_act_candidate_reasons=list(extraction.main_act_candidate_reasons),
        )
