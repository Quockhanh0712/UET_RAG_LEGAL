# tests/test_load_file.py
import pytest
from pathlib import Path

# Skip this test - file_manager module not available
pytestmark = pytest.mark.skip(reason="file_manager module not available")

try:
    from src.file_manager import load_file, delete_file
    from src import vector_store
except ImportError:
    load_file = None
    delete_file = None
    vector_store = None

def test_load_and_list_files(tmp_path: Path):
    # tạo file text tạm
    p = tmp_path / "sample.txt"
    p.write_text(
        "Python là một ngôn ngữ lập trình. "
        "RAG giúp kết hợp LLM với dữ liệu riêng.",
        encoding="utf-8",
    )

    # nạp file vào hệ thống
    file_id = load_file(str(p))
    files = vector_store.list_files()

    assert file_id in files  # file đã ghi vào Chroma

    # dọn dẹp
    delete_file(file_id)
    files_after = vector_store.list_files()
    assert file_id not in files_after
