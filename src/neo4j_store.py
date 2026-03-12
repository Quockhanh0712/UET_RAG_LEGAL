"""
Neo4j persistence and graph retrieval for legal documents.

The implementation is optional at runtime: if neo4j driver or credentials are
missing, methods become no-op and return safe defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.legal_data_processing import DOC_TYPE_HIERARCHY
from src.legal_graph_extractor import GraphRelation, REL_ABROGATES, REL_AMENDS, REL_CITES, REL_REPLACES

logger = logging.getLogger(__name__)


NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jLegalStore:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

        if not self.uri or not self.user or not self.password:
            logger.info("[NEO4J] Disabled: missing credentials")
            return

        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("[NEO4J] neo4j package not installed. Graph features disabled.")
            return

        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("[NEO4J] Connected to %s", self.uri)
        except Exception as exc:
            logger.warning("[NEO4J] Connection failed, graph features disabled: %s", exc)
            self._driver = None

    @property
    def enabled(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def _execute_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        with self._driver.session(database=self.database) as session:
            session.run(query, params or {})

    def _execute_read(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        with self._driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [dict(row) for row in result]

    def create_constraints(self) -> None:
        if not self.enabled:
            return

        statements = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT provision_id_unique IF NOT EXISTS FOR (p:Provision) REQUIRE p.provision_id IS UNIQUE",
            "CREATE CONSTRAINT doctype_name_unique IF NOT EXISTS FOR (t:DocType) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT agency_name_unique IF NOT EXISTS FOR (a:Agency) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT signer_name_unique IF NOT EXISTS FOR (s:Signer) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT lawfamily_name_unique IF NOT EXISTS FOR (f:LawFamily) REQUIRE f.name IS UNIQUE",
        ]
        for statement in statements:
            self._execute_write(statement)
        logger.info("[NEO4J] Constraints ensured")

    def upsert_doc_type_hierarchy(self) -> None:
        if not self.enabled:
            return

        types = [{"name": name, "rank": idx + 1} for idx, name in enumerate(DOC_TYPE_HIERARCHY)]
        types.append({"name": "OTHER", "rank": len(DOC_TYPE_HIERARCHY) + 1})
        edges = []
        for idx in range(len(DOC_TYPE_HIERARCHY) - 1):
            edges.append({"superior": DOC_TYPE_HIERARCHY[idx], "inferior": DOC_TYPE_HIERARCHY[idx + 1]})

        self._execute_write(
            """
            UNWIND $types AS t
            MERGE (dt:DocType {name: t.name})
            SET dt.rank = t.rank
            """,
            {"types": types},
        )
        self._execute_write(
            """
            UNWIND $edges AS e
            MATCH (sup:DocType {name: e.superior})
            MATCH (inf:DocType {name: e.inferior})
            MERGE (sup)-[:TYPE_SUPERIOR_TO]->(inf)
            """,
            {"edges": edges},
        )

    def upsert_documents(self, documents: Sequence[Dict[str, Any]]) -> None:
        if not self.enabled or not documents:
            return

        self._execute_write(
            """
            UNWIND $docs AS d
            MERGE (doc:Document {doc_id: d.doc_id})
            SET doc.so_ky_hieu = d.so_ky_hieu,
                doc.ten_van_ban = d.ten_van_ban,
                doc.loai_van_ban = d.loai_van_ban,
                doc.doc_type = d.doc_type,
                doc.doc_type_rank = d.doc_type_rank,
                doc.co_quan_ban_hanh = d.co_quan_ban_hanh,
                doc.chuc_danh = d.chuc_danh,
                doc.nguoi_ky = d.nguoi_ky,
                doc.ngay_ban_hanh = d.ngay_ban_hanh,
                doc.ngay_co_hieu_luc = d.ngay_co_hieu_luc,
                doc.ngay_het_hieu_luc = d.ngay_het_hieu_luc,
                doc.is_active = d.is_active,
                doc.law_family = d.law_family,
                doc.citation_base = d.citation_base,
                doc.mo_ta = d.mo_ta,
                doc.updated_at = datetime()
            MERGE (dtype:DocType {name: coalesce(d.doc_type, 'OTHER')})
            SET dtype.rank = coalesce(d.doc_type_rank, 999)
            MERGE (doc)-[:OF_TYPE]->(dtype)
            FOREACH (_ IN CASE WHEN d.co_quan_ban_hanh <> '' THEN [1] ELSE [] END |
                MERGE (agency:Agency {name: d.co_quan_ban_hanh})
                MERGE (doc)-[:ISSUED_BY]->(agency)
            )
            FOREACH (_ IN CASE WHEN d.nguoi_ky <> '' THEN [1] ELSE [] END |
                MERGE (signer:Signer {name: d.nguoi_ky})
                SET signer.chuc_danh = d.chuc_danh
                MERGE (doc)-[:SIGNED_BY]->(signer)
            )
            FOREACH (_ IN CASE WHEN d.law_family <> '' THEN [1] ELSE [] END |
                MERGE (family:LawFamily {name: d.law_family})
                MERGE (doc)-[:BELONGS_TO_FAMILY]->(family)
            )
            """,
            {"docs": list(documents)},
        )

    def upsert_provisions(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if not self.enabled or not chunks:
            return

        self._execute_write(
            """
            UNWIND $chunks AS c
            MATCH (doc:Document {doc_id: c.doc_id})
            MERGE (p:Provision {provision_id: c.provision_id})
            SET p.doc_id = c.doc_id,
                p.dieu_so = c.dieu_so,
                p.khoan_so = c.khoan_so,
                p.diem_so = c.diem_so,
                p.chuong = c.chuong,
                p.ten_chuong = c.ten_chuong,
                p.tieu_de_dieu = c.tieu_de_dieu,
                p.path = c.path,
                p.content = c.content,
                p.updated_at = datetime()
            MERGE (doc)-[:HAS_PROVISION]->(p)
            FOREACH (_ IN CASE WHEN c.parent_provision_id <> '' THEN [1] ELSE [] END |
                MERGE (parent:Provision {provision_id: c.parent_provision_id})
                MERGE (parent)-[:CONTAINS]->(p)
            )
            """,
            {"chunks": list(chunks)},
        )

    def upsert_amendment_relations(self, relations: Sequence[GraphRelation]) -> None:
        if not self.enabled or not relations:
            return

        grouped: Dict[str, List[Dict[str, Any]]] = {
            REL_AMENDS: [],
            REL_REPLACES: [],
            REL_ABROGATES: [],
            REL_CITES: [],
        }
        for rel in relations:
            if rel.relation_type in grouped:
                grouped[rel.relation_type].append(
                    {
                        "source_doc_id": rel.source_doc_id,
                        "target_doc_id": rel.target_doc_id,
                        "evidence_text": rel.evidence_text,
                        "evidence_source": rel.evidence_source,
                        "confidence_rule": rel.confidence_rule,
                        "snapshot_date": rel.snapshot_date,
                    }
                )

        for relation_type, payload in grouped.items():
            if not payload:
                continue
            self._execute_write(
                f"""
                UNWIND $rels AS r
                MERGE (src:Document {{doc_id: r.source_doc_id}})
                MERGE (tgt:Document {{doc_id: r.target_doc_id}})
                MERGE (src)-[rel:{relation_type}]->(tgt)
                SET rel.evidence_text = r.evidence_text,
                    rel.evidence_source = coalesce(r.evidence_source, 'mo_ta'),
                    rel.confidence_rule = coalesce(r.confidence_rule, ''),
                    rel.snapshot_date = coalesce(r.snapshot_date, ''),
                    rel.updated_at = datetime()
                """,
                {"rels": payload},
            )

    def expand_related_doc_ids(
        self,
        anchor_doc_ids: Sequence[str],
        hops: int = 1,
        limit: int = 20,
        include_same_type: bool = True,
        mode: str = "local",
    ) -> List[str]:
        if not self.enabled or not anchor_doc_ids:
            return []

        hops = max(1, min(int(hops), 1))
        mode = (mode or "local").lower().strip()
        relation_pattern = "AMENDS|REPLACES|ABROGATES|CITES"
        if mode == "legal_impact":
            relation_pattern = "AMENDS|REPLACES|ABROGATES"
        elif mode == "temporal":
            relation_pattern = "AMENDS|REPLACES|ABROGATES"
        elif mode == "local":
            relation_pattern = "CITES|AMENDS|REPLACES|ABROGATES"

        relation_query = f"""
        MATCH (d:Document)
        WHERE d.doc_id IN $doc_ids
        MATCH (d)-[:{relation_pattern}*1..{hops}]-(n:Document)
        WHERE n.doc_id IS NOT NULL AND n.doc_id <> d.doc_id
        RETURN DISTINCT n.doc_id AS doc_id
        LIMIT $limit
        """
        rows = self._execute_read(relation_query, {"doc_ids": list(anchor_doc_ids), "limit": int(limit)})
        related = {row["doc_id"] for row in rows if row.get("doc_id")}

        if include_same_type and len(related) < limit:
            same_type_query = """
            MATCH (d:Document)-[:OF_TYPE]->(t:DocType)<-[:OF_TYPE]-(n:Document)
            WHERE d.doc_id IN $doc_ids AND n.doc_id <> d.doc_id
            RETURN DISTINCT n.doc_id AS doc_id
            LIMIT $limit
            """
            rows = self._execute_read(
                same_type_query,
                {"doc_ids": list(anchor_doc_ids), "limit": int(limit)},
            )
            for row in rows:
                doc_id = row.get("doc_id")
                if doc_id:
                    related.add(doc_id)

        return list(related)[:limit]


_graph_store: Optional[Neo4jLegalStore] = None


def get_graph_store() -> Neo4jLegalStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = Neo4jLegalStore()
    return _graph_store
