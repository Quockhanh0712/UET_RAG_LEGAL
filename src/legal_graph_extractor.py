"""
Rule-based graph extraction for Vietnamese legal documents.

No LLM is used in this module. All relations are extracted with deterministic
regex/keyword rules for auditability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.legal_data_processing import DOC_TYPE_HIERARCHY, normalize_doc_id, strip_accents


REL_AMENDS = "AMENDS"
REL_REPLACES = "REPLACES"
REL_ABROGATES = "ABROGATES"
REL_CITES = "CITES"


RELATION_RULES: Dict[str, Tuple[str, ...]] = {
    REL_ABROGATES: (
        "bai bo",
        "huy bo",
        "het hieu luc",
    ),
    REL_REPLACES: (
        "thay the",
        "thay cho",
    ),
    REL_AMENDS: (
        "sua doi",
        "bo sung",
        "dieu chinh",
    ),
}


DOC_REF_PATTERN = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-Z0-9Đ\-]+(?:\s*[-/]\s*[A-Z0-9Đ\-]+)*\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GraphRelation:
    source_doc_id: str
    target_doc_id: str
    relation_type: str
    evidence_text: str
    evidence_source: str = "mo_ta"
    confidence_rule: str = ""
    snapshot_date: str = ""

    @property
    def evidence(self) -> str:
        # Backward-compatible alias for older callers.
        return self.evidence_text


def extract_doc_refs(text: str) -> List[str]:
    refs = [normalize_doc_id(match.group(0)) for match in DOC_REF_PATTERN.finditer(text or "")]
    dedup: List[str] = []
    seen: Set[str] = set()
    for ref in refs:
        if ref not in seen and ref != "UNKNOWN":
            seen.add(ref)
            dedup.append(ref)
    return dedup


def _detect_relation_type(sentence_text: str) -> Optional[str]:
    sentence_ascii = strip_accents(sentence_text).lower()
    for relation_type in (REL_ABROGATES, REL_REPLACES, REL_AMENDS):
        keywords = RELATION_RULES[relation_type]
        if any(keyword in sentence_ascii for keyword in keywords):
            return relation_type
    return None


class LegalGraphExtractor:
    """
    Extract graph relations from normalized documents and legal chunks.
    """

    def __init__(self):
        self._doc_type_edges = self._build_doc_type_hierarchy_edges()

    @staticmethod
    def _build_doc_type_hierarchy_edges() -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for idx in range(len(DOC_TYPE_HIERARCHY) - 1):
            edges.append((DOC_TYPE_HIERARCHY[idx], DOC_TYPE_HIERARCHY[idx + 1]))
        return edges

    def get_doc_type_hierarchy_edges(self) -> List[Tuple[str, str]]:
        return list(self._doc_type_edges)

    def extract_amendment_relations(
        self,
        doc_id: str,
        mo_ta: str,
        snapshot_date: Optional[str] = None,
    ) -> List[GraphRelation]:
        source_doc_id = normalize_doc_id(doc_id)
        if source_doc_id == "UNKNOWN":
            return []

        text = mo_ta or ""
        if not text.strip():
            return []

        relations: List[GraphRelation] = []
        seen: Set[Tuple[str, str, str]] = set()
        snapshot = snapshot_date or date.today().isoformat()

        # Sentence-level extraction for precise relation typing
        segments = [segment.strip() for segment in re.split(r"[;\n\.]+", text) if segment.strip()]
        for segment in segments:
            ref_matches = list(DOC_REF_PATTERN.finditer(segment))
            if not ref_matches:
                continue

            for match in ref_matches:
                ref = normalize_doc_id(match.group(0))
                context_start = max(0, match.start() - 80)
                local_context = segment[context_start : match.end()]
                relation_type = _detect_relation_type(local_context) or _detect_relation_type(segment) or REL_CITES
                if ref == source_doc_id:
                    continue
                key = (source_doc_id, ref, relation_type)
                if key in seen:
                    continue
                seen.add(key)
                confidence_rule = (
                    "segment_keyword_docref"
                    if relation_type in {REL_AMENDS, REL_REPLACES, REL_ABROGATES}
                    else "segment_docref_only"
                )
                relations.append(
                    GraphRelation(
                        source_doc_id=source_doc_id,
                        target_doc_id=ref,
                        relation_type=relation_type,
                        evidence_text=segment[:300],
                        evidence_source="mo_ta",
                        confidence_rule=confidence_rule,
                        snapshot_date=snapshot,
                    )
                )

        # Fallback: if relation keyword exists but refs only appear globally
        if not relations:
            relation_type = _detect_relation_type(text)
            if relation_type is not None:
                for ref in extract_doc_refs(text):
                    if ref == source_doc_id:
                        continue
                    key = (source_doc_id, ref, relation_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        GraphRelation(
                            source_doc_id=source_doc_id,
                            target_doc_id=ref,
                            relation_type=relation_type,
                            evidence_text=text[:300],
                            evidence_source="mo_ta",
                            confidence_rule="document_keyword_docref",
                            snapshot_date=snapshot,
                        )
                    )

        return relations

    def extract_relations_for_documents(
        self,
        records: Iterable[Dict[str, Any]],
        snapshot_date: Optional[str] = None,
    ) -> List[GraphRelation]:
        all_relations: List[GraphRelation] = []
        for record in records:
            doc_id = record.get("doc_id") or record.get("so_ky_hieu") or record.get("van_ban_id")
            mo_ta = record.get("mo_ta", "")
            all_relations.extend(
                self.extract_amendment_relations(
                    doc_id=str(doc_id),
                    mo_ta=str(mo_ta),
                    snapshot_date=snapshot_date,
                )
            )
        return all_relations
