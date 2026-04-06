from __future__ import annotations

import logging
from pathlib import Path

from .chunking import ChunkRouter
from .extractors import ExtractorRouter
from .indexer import UserDocumentIndexer
from .models import (
    IngestOutcome,
    IngestRequest,
    STATUS_FAILED,
    STATUS_NEEDS_OCR,
    STATUS_READY,
)
from .ocr import OcrProviderRouter
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


class DocumentIngestionProcessor:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        extractor_router: ExtractorRouter | None = None,
        ocr_router: OcrProviderRouter | None = None,
        chunk_router: ChunkRouter | None = None,
        indexer: UserDocumentIndexer | None = None,
    ):
        self.artifact_store = artifact_store
        self.extractor_router = extractor_router or ExtractorRouter()
        self.ocr_router = ocr_router or OcrProviderRouter(artifact_store)
        self.chunk_router = chunk_router or ChunkRouter()
        self.indexer = indexer or UserDocumentIndexer()

    def ingest_saved_file(self, request: IngestRequest) -> IngestOutcome:
        try:
            if request.content_mode == "ocr":
                extracted = self.ocr_router.process(request.ocr_provider, request.saved_file_path, request.artifact_dir)
            else:
                extracted = self.extractor_router.extract(request.saved_file_path)
                if extracted.needs_ocr:
                    if extracted.normalized_text:
                        self.artifact_store.save_text(
                            request.artifact_dir,
                            "extracted_text",
                            "document.txt",
                            extracted.normalized_text,
                        )
                    return IngestOutcome(
                        status=STATUS_NEEDS_OCR,
                        parser_used=extracted.parser_used,
                        page_count=extracted.page_count,
                        text_chars=extracted.text_chars,
                        ocr_provider=extracted.ocr_provider,
                    )

            self.artifact_store.save_text(request.artifact_dir, "extracted_text", "document.txt", extracted.text)
            self.artifact_store.save_text(
                request.artifact_dir,
                "normalized_text",
                "document.txt",
                extracted.normalized_text,
            )
            chunks = self.chunk_router.chunk(extracted, request)
            self.artifact_store.save_chunks(request.artifact_dir, chunks)
            self.indexer.upsert(chunks)
            logger.info(
                "[INGEST] Indexed %s chunks for doc_id=%s parser=%s mode=%s",
                len(chunks),
                request.doc_id,
                extracted.parser_used,
                request.content_mode,
            )
            return IngestOutcome(
                status=STATUS_READY,
                chunks=chunks,
                parser_used=extracted.parser_used,
                ocr_provider=extracted.ocr_provider,
                page_count=extracted.page_count,
                text_chars=extracted.text_chars,
            )
        except Exception as exc:
            logger.exception("[INGEST] Failed for doc_id=%s", request.doc_id)
            return IngestOutcome(
                status=STATUS_FAILED,
                parser_used="",
                page_count=0,
                text_chars=0,
                error_message=str(exc),
            )

    def delete_document(self, *, doc_id: str, user_id: str, artifact_dir: Path) -> None:
        self.indexer.delete_document(doc_id=doc_id, user_id=user_id)
        self.artifact_store.delete_artifacts(artifact_dir)
