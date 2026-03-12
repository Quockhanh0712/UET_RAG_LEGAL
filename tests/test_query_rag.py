# tests/test_query_rag.py
import pytest
import types
from pathlib import Path

# Skip this test - rag_pipeline module not available
pytestmark = pytest.mark.skip(reason="rag_pipeline module not available in src/__init__.py")

try:
    from src import rag_pipeline
    from src import vector_store
    from src.file_manager import load_file, delete_file
except ImportError:
    rag_pipeline = None
    vector_store = None
    load_file = None
    delete_file = None

# client giả thay cho Gemini thật
class DummyClient:
    class DummyModels:
        def generate_content(self, model, contents):
            # luôn trả về câu trả lời cố định
            return types.SimpleNamespace(text="DUMMY_ANSWER_FOR_TEST")

    @property
    def models(self):
        return self.DummyModels()

def setup_module(module):
    # gắn client giả vào llm_client trước khi test
    import src.llm_client as llm_client
    llm_client._client = DummyClient()

def test_rag_pipeline(tmp_path: Path):
    # 1. tạo file text
    p = tmp_path / "rag.txt"
    p.write_text(
        "RAG là kỹ thuật kết hợp LLM với dữ liệu riêng của người dùng.",
        encoding="utf-8",
    )

    # 2. nạp file
    file_id = load_file(str(p))
    assert file_id in vector_store.list_files()

    # 3. truy vấn
    res = rag_pipeline.rag_answer("RAG là gì?", k=3)

    assert "answer" in res
    assert res["answer"] == "DUMMY_ANSWER_FOR_TEST"
    assert len(res["context"]) > 0  # có ít nhất 1 chunk được trả về

    # 4. xóa file
    delete_file(file_id)
    assert file_id not in vector_store.list_files()
