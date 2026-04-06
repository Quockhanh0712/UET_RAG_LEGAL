from __future__ import annotations

from importlib import import_module


_ATTR_TO_MODULE = {
    "ArtifactStore": "storage",
    "ChunkRouter": "chunking",
    "GenericDocumentChunker": "chunking",
    "ExtractorRouter": "extractors",
    "PdfTextExtractor": "extractors",
    "DocxTextExtractor": "extractors",
    "TxtTextExtractor": "extractors",
    "DefaultOcrProvider": "ocr",
    "OcrProviderRouter": "ocr",
    "UserDocumentIndexer": "indexer",
    "DocumentIngestionProcessor": "service",
    "IngestRequest": "models",
    "IngestOutcome": "models",
    "ExtractedDocument": "models",
    "ExtractedPage": "models",
    "IngestedChunk": "models",
    "UnsupportedDocumentError": "extractors",
}

__all__ = sorted(_ATTR_TO_MODULE)


def __getattr__(name: str):
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
