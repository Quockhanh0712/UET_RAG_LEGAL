"""
Pytest Configuration and Shared Fixtures.
"""
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web_app"))
sys.path.insert(0, str(PROJECT_ROOT / "web_app" / "backend"))


# ==================== Async Event Loop ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== FastAPI Test Client ====================

@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for API testing."""
    try:
        from backend.main import app
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Cannot import backend.main: {e}")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def initialized_app():
    """Initialize app with database."""
    try:
        from backend.main import app
        from backend.db.database import init_db, close_db
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Cannot import backend modules: {e}")
    
    await init_db()
    yield app
    await close_db()


# ==================== Core Component Fixtures ====================

@pytest.fixture
def qdrant_connector():
    """Get Qdrant connector instance."""
    try:
        from backend.core.qdrant_store import get_qdrant_connector
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend.core.qdrant_store not available: {e}")
    return get_qdrant_connector()


@pytest.fixture
def embedding_model():
    """Get embedding model instance (lazy loaded)."""
    try:
        from backend.core.embeddings import get_embedding_model
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend.core.embeddings not available: {e}")
    return get_embedding_model()


@pytest.fixture
def reranker_model():
    """Get reranker model instance (lazy loaded)."""
    try:
        from backend.core.reranker import get_reranker
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend.core.reranker not available: {e}")
    return get_reranker()


@pytest.fixture
def llm_client():
    """Get LLM client instance."""
    try:
        from backend.core.llm_client import get_llm_client
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend.core.llm_client not available: {e}")
    return get_llm_client()


@pytest.fixture
def rag_pipeline():
    """Get RAG pipeline instance."""
    try:
        from backend.core.rag_pipeline import get_rag_pipeline
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"backend.core.rag_pipeline not available: {e}")
    return get_rag_pipeline()


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_query():
    """Sample Vietnamese legal query."""
    return "Tội tham nhũng bị phạt bao nhiêu năm tù?"


@pytest.fixture
def sample_texts():
    """Sample texts for embedding tests."""
    return [
        "Điều 353 quy định về tội tham ô tài sản",
        "Người phạm tội có thể bị phạt tù từ 2 đến 7 năm",
        "Hình phạt bổ sung bao gồm phạt tiền và cấm đảm nhiệm chức vụ"
    ]


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "test_user_001"


@pytest.fixture
def sample_session_id():
    """Sample session ID for testing."""
    return "test_session_001"


# ==================== Pytest Configuration ====================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
