from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

from .models import IngestedChunk


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return cleaned or fallback


class ArtifactStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def document_dir(self, user_id: str, session_id: str, doc_id: str) -> Path:
        safe_user = _safe_component(user_id, "user")
        safe_session = _safe_component(session_id, "session")
        safe_doc = _safe_component(doc_id, "document")
        directory = self.base_dir / safe_user / safe_session / safe_doc
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def original_file_path(self, user_id: str, session_id: str, doc_id: str, filename: str) -> Path:
        directory = self.document_dir(user_id, session_id, doc_id) / "original"
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_component(Path(filename).name, "upload.bin")
        return directory / safe_name

    def save_original(
        self,
        *,
        user_id: str,
        session_id: str,
        doc_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> Path:
        target = self.original_file_path(user_id, session_id, doc_id, filename)
        target.write_bytes(file_bytes)
        return target

    def save_text(self, artifact_dir: Path, subdir: str, filename: str, content: str) -> Path:
        directory = Path(artifact_dir) / subdir
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        target.write_text(content or "", encoding="utf-8")
        return target

    def save_chunks(self, artifact_dir: Path, chunks: Iterable[IngestedChunk]) -> Path:
        target = Path(artifact_dir) / "chunks.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        return target

    def ensure_subdir(self, artifact_dir: Path, name: str) -> Path:
        directory = Path(artifact_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def delete_artifacts(self, artifact_dir: Path) -> None:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
