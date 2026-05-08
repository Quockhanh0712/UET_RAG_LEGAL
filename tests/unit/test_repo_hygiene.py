from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]


LIVE_DOCS = [
    PROJECT_ROOT / "docs" / "BACKEND_ARCHITECTURE_MAINTENANCE.md",
    PROJECT_ROOT / "docs" / "LEGAL_SYSTEM_OVERVIEW.md",
    PROJECT_ROOT / "docs" / "LEGAL_PIPELINE_CURRENT_STATE.md",
    PROJECT_ROOT / "docs" / "LEGAL_REWRITE_V2.md",
]

OPS_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "ops" / "run_hybrid_index_production.py",
    PROJECT_ROOT / "scripts" / "ops" / "upload_qdrant_cloud_embeddings.py",
    PROJECT_ROOT / "scripts" / "ops" / "patch_qdrant_metadata.py",
]


def test_live_repo_paths_exist():
    required = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "web_app" / "backend" / "main.py",
        PROJECT_ROOT / "web_app" / "frontend" / "package.json",
        PROJECT_ROOT / "tests" / "unit",
        *LIVE_DOCS,
        *OPS_SCRIPTS,
    ]

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    assert not missing, f"Missing live repo paths: {missing}"


def test_generated_noise_artifacts_are_absent_from_repo_tree():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    forbidden_tracked = []
    for path in tracked:
        normalized = path.replace("/", "\\")
        if (
            normalized == ".pytest_cache"
            or normalized.startswith(".pytest_cache\\")
            or normalized == "web_app\\frontend\\dist"
            or normalized.startswith("web_app\\frontend\\dist\\")
            or normalized == "web_app\\frontend\\tmp_frontend_vite.log"
            or "\\__pycache__\\" in normalized
            or normalized.endswith("\\__pycache__")
        ):
            forbidden_tracked.append(path)

    assert not forbidden_tracked, f"Generated noise artifacts are tracked: {forbidden_tracked}"


def test_historical_docs_are_archived():
    archived = [
        PROJECT_ROOT / "docs" / "archive" / "plans" / "requery_qdrant_refactor_v1",
        PROJECT_ROOT / "docs" / "archive" / "refactors" / "backend_rag_2026_04_02",
        PROJECT_ROOT / "docs" / "archive" / "handover" / "WORKSPACE_HANDOVER_2026-03-11.md",
        PROJECT_ROOT / "docs" / "archive" / "source-snapshots",
    ]
    legacy_locations = [
        PROJECT_ROOT / "docs" / "plan" / "requery_qdrant_refactor_v1",
        PROJECT_ROOT / "docs" / "refactor_backend_rag_2026_04_02",
        PROJECT_ROOT / "docs" / "WORKSPACE_HANDOVER_2026-03-11.md",
        PROJECT_ROOT / "docs" / "src",
    ]

    missing_archive = [str(path.relative_to(PROJECT_ROOT)) for path in archived if not path.exists()]
    lingering_legacy = [str(path.relative_to(PROJECT_ROOT)) for path in legacy_locations if path.exists()]

    assert not missing_archive, f"Missing archived docs: {missing_archive}"
    assert not lingering_legacy, f"Historical docs still in live locations: {lingering_legacy}"
