from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from src.document_ingestion.indexer import UserDocumentIndexer
from src.document_ingestion.models import IngestedChunk


@dataclass
class _FakeDenseVector:
    size: int


class _FakeCollectionInfo:
    def __init__(self, size: int):
        self.config = SimpleNamespace(params=SimpleNamespace(vectors={"dense": _FakeDenseVector(size=size)}))


class _FakeClient:
    def __init__(self, collections=None, aliases=None):
        self.collections = dict(collections or {})
        self.aliases = dict(aliases or {})
        self.upsert_calls = []
        self.deleted = []
        self.payload_index_calls = []
        self.query_calls = []

    def get_collection(self, collection_name: str):
        target = self.aliases.get(collection_name, collection_name)
        if target not in self.collections:
            raise RuntimeError(f"collection not found: {collection_name}")
        return _FakeCollectionInfo(self.collections[target])

    def create_collection(self, collection_name: str, vectors_config, sparse_vectors_config, on_disk_payload):
        dense_cfg = vectors_config["dense"]
        self.collections[collection_name] = int(dense_cfg.size)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema: str):
        self.payload_index_calls.append((collection_name, field_name, field_schema))

    def update_collection_aliases(self, change_aliases_operations):
        for operation in change_aliases_operations:
            if hasattr(operation, "create_alias"):
                create = operation.create_alias
                self.aliases[create.alias_name] = create.collection_name
            elif hasattr(operation, "delete_alias"):
                delete = operation.delete_alias
                self.aliases.pop(delete.alias_name, None)

    def get_aliases(self):
        aliases = [
            SimpleNamespace(alias_name=alias_name, collection_name=collection_name)
            for alias_name, collection_name in self.aliases.items()
        ]
        return SimpleNamespace(aliases=aliases)

    def upsert(self, *, collection_name: str, points, wait: bool):
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points_count": len(points),
                "wait": wait,
            }
        )

    def delete(self, *, collection_name: str, points_selector):
        self.deleted.append((collection_name, points_selector))

    def query_points(self, *, collection_name: str, query, using: str, limit: int, query_filter, with_payload: bool):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "limit": limit,
                "query_filter": query_filter,
                "with_payload": with_payload,
            }
        )
        point = SimpleNamespace(
            payload={
                "content": "Tai lieu nguoi dung",
                "filename": "tai_lieu.txt",
                "doc_id": "doc-1",
                "page_number": 1,
            },
            score=0.87,
        )
        return SimpleNamespace(points=[point])


class _FakeStore:
    def __init__(self, client: _FakeClient, collection_name: str, dense_dim: int, payload_profile: str):
        self.client = client
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.payload_profile = payload_profile
        self.bm25_encoder = SimpleNamespace(vocab={}, encode=lambda text: ([], []))
        self.hybrid_calls = []

    def create_collection(self, force_recreate: bool = False):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": SimpleNamespace(size=self.dense_dim)},
            sparse_vectors_config={"sparse": object()},
            on_disk_payload=True,
        )

    def _create_payload_indexes(self):
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )

    def resolve_alias(self, alias_name: str):
        return self.client.aliases.get(alias_name)

    def set_alias(self, alias_name: str, target_collection: str):
        self.client.aliases[alias_name] = target_collection

    def swap_alias(self, alias_name: str, new_collection: str):
        self.client.aliases[alias_name] = new_collection

    @staticmethod
    def _build_filter(filter_conditions):
        return dict(filter_conditions or {})

    def hybrid_search(
        self,
        *,
        query: str,
        query_embedding,
        top_k: int,
        filter_conditions,
        collection_name: str,
    ):
        self.hybrid_calls.append(
            {
                "query": query,
                "query_embedding": query_embedding,
                "top_k": top_k,
                "filter_conditions": dict(filter_conditions or {}),
                "collection_name": collection_name,
            }
        )
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                content="Tai lieu nguoi dung",
                score=0.91,
                metadata={
                    "chunk_id": "chunk-1",
                    "content": "Tai lieu nguoi dung",
                    "filename": "tai_lieu.txt",
                    "doc_id": "doc-1",
                    "page_number": 1,
                },
            )
        ]


def _build_indexer(
    monkeypatch,
    client: _FakeClient,
    *,
    dense_dim: int = 1024,
    embedding_model_name: str = "unit/test-embedding",
) -> UserDocumentIndexer:
    import src.document_ingestion.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "_get_embedding_dimension", lambda: dense_dim)
    monkeypatch.setattr(indexer_module, "_current_embedding_model_name", lambda: embedding_model_name)
    monkeypatch.setattr(
        indexer_module,
        "_embed_texts",
        lambda texts, text_type: [[0.01] * dense_dim for _ in texts],
    )
    monkeypatch.setattr(
        indexer_module,
        "_embed_query",
        lambda query: [0.02] * dense_dim,
    )
    monkeypatch.setattr(
        indexer_module,
        "QdrantLegalStore",
        lambda collection_name, dense_dim, payload_profile: _FakeStore(
            client,
            collection_name,
            dense_dim,
            payload_profile,
        ),
    )
    return UserDocumentIndexer()


def _build_chunk() -> IngestedChunk:
    return IngestedChunk(
        chunk_id="chunk-1",
        content="Tai lieu nguoi dung",
        doc_id="doc-1",
        user_id="user-1",
        session_id="session-1",
        filename="tai_lieu.txt",
    )


def test_upsert_uses_versioned_collection_when_legacy_collection_dim_mismatches(monkeypatch):
    client = _FakeClient(collections={"user_docs_private": 768})
    indexer = _build_indexer(monkeypatch, client)

    indexer.upsert([_build_chunk()])

    assert indexer.versioned_collection_name in client.collections
    assert client.collections[indexer.versioned_collection_name] == 1024
    assert client.upsert_calls[0]["collection_name"] == indexer.versioned_collection_name
    assert indexer._active_collection_name == indexer.versioned_collection_name


def test_versioned_collection_name_uses_legal_embedding_model_and_dimension(monkeypatch):
    client = _FakeClient()
    indexer = _build_indexer(
        monkeypatch,
        client,
        dense_dim=768,
        embedding_model_name="truro7/vn-law-embedding",
    )

    assert indexer.versioned_collection_name == "user_docs_private__truro7_vn_law_embedding__d768"


def test_upsert_binds_logical_alias_to_versioned_collection_on_clean_deploy(monkeypatch):
    client = _FakeClient()
    indexer = _build_indexer(monkeypatch, client)

    indexer.upsert([_build_chunk()])

    assert client.aliases["user_docs_private"] == indexer.versioned_collection_name
    assert client.upsert_calls[0]["collection_name"] == "user_docs_private"
    assert indexer.versioned_collection_name in client.collections
    assert indexer._active_collection_name == "user_docs_private"


def test_upsert_repairs_alias_when_alias_target_dim_mismatches(monkeypatch):
    client = _FakeClient(
        collections={"user_docs_private__old__d768": 768},
        aliases={"user_docs_private": "user_docs_private__old__d768"},
    )
    indexer = _build_indexer(monkeypatch, client)

    indexer.upsert([_build_chunk()])

    assert client.aliases["user_docs_private"] == indexer.versioned_collection_name
    assert client.collections[indexer.versioned_collection_name] == 1024
    assert client.upsert_calls[0]["collection_name"] == "user_docs_private"


def test_upsert_avoids_legacy_base_collection_even_when_dimension_matches(monkeypatch):
    client = _FakeClient(collections={"user_docs_private": 768})
    indexer = _build_indexer(
        monkeypatch,
        client,
        dense_dim=768,
        embedding_model_name="truro7/vn-law-embedding",
    )

    indexer.upsert([_build_chunk()])

    assert indexer.versioned_collection_name in client.collections
    assert client.upsert_calls[0]["collection_name"] == indexer.versioned_collection_name
    assert indexer._active_collection_name == indexer.versioned_collection_name


def test_delete_routes_to_active_versioned_collection_when_legacy_collection_is_incompatible(monkeypatch):
    client = _FakeClient(collections={"user_docs_private": 768})
    indexer = _build_indexer(monkeypatch, client)

    indexer.delete_document(doc_id="doc-1", user_id="user-1")

    assert client.deleted[0][0] == indexer.versioned_collection_name


def test_search_omits_session_filter_for_user_scope(monkeypatch):
    client = _FakeClient()
    indexer = _build_indexer(monkeypatch, client)
    indexer.store.bm25_encoder.vocab = {"tai_lieu": 1}

    results = indexer.search(
        query="tai lieu",
        user_id="user-1",
        session_id="session-2",
        top_k=3,
        use_hybrid=True,
    )

    assert len(results) == 1
    assert indexer.store.hybrid_calls[0]["filter_conditions"] == {"user_id": "user-1"}
    assert indexer.store.hybrid_calls[0]["collection_name"] == indexer._active_collection()
