from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict

from .extractors import PdfTextExtractor
from .models import ExtractedDocument
from .storage import ArtifactStore


class OcrProviderError(RuntimeError):
    """Raised when OCR processing cannot be completed."""


class DefaultOcrProvider:
    name = "default"

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store
        self._pdf_extractor = PdfTextExtractor(min_text_chars=1)

    def process(self, file_path: str | Path, artifact_dir: Path) -> ExtractedDocument:
        source = Path(file_path)
        if source.suffix.lower() != ".pdf":
            raise OcrProviderError("OCR default provider currently supports PDF inputs only.")

        executable = shutil.which("ocrmypdf")
        if not executable:
            raise OcrProviderError("OCRmyPDF is not installed or not available on PATH.")

        ocr_dir = self.artifact_store.ensure_subdir(artifact_dir, "ocr_output")
        output_pdf = ocr_dir / "ocr.pdf"
        result = subprocess.run(
            [
                executable,
                "--force-ocr",
                "--skip-text",
                str(source),
                str(output_pdf),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and not output_pdf.exists():
            message = (result.stderr or result.stdout or "").strip()
            raise OcrProviderError(message or "OCRmyPDF failed to process the document.")

        extracted = self._pdf_extractor.extract(output_pdf)
        extracted.parser_used = "pymupdf+ocrmypdf"
        extracted.ocr_provider = self.name
        extracted.needs_ocr = False
        return extracted


class PaddleOcrProvider:
    name = "paddleocr"

    def process(self, file_path: str | Path, artifact_dir: Path) -> ExtractedDocument:
        raise OcrProviderError("PaddleOCR provider is not implemented yet.")


class OcrProviderRouter:
    def __init__(self, artifact_store: ArtifactStore):
        default_provider = DefaultOcrProvider(artifact_store)
        self._providers: Dict[str, object] = {
            "default": default_provider,
            "ocrmypdf": default_provider,
            "paddleocr": PaddleOcrProvider(),
        }

    def process(self, provider_name: str, file_path: str | Path, artifact_dir: Path) -> ExtractedDocument:
        normalized_name = str(provider_name or "default").strip().lower()
        provider = self._providers.get(normalized_name)
        if provider is None:
            raise OcrProviderError(f"Unsupported OCR provider: {provider_name}")
        return provider.process(file_path, artifact_dir)
