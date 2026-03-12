# test_gpu_embedding.py
import pytest
import torch

# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

def test_gpu_embedding_benchmark():
    """Test GPU embedding performance"""
    from src.embeddings import get_model, embed_texts
    import time
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    model = get_model()
    print(f"Model device: {model.device}")
    
    # Benchmark
    texts = ["Test sentence"] * 100
    start = time.time()
    embeddings = embed_texts(texts, batch_size=32)
    end = time.time()
    
    print(f"Embedded {len(texts)} texts in {end-start:.2f}s")
    print(f"Speed: {len(texts)/(end-start):.1f} it/s")
    
    assert len(embeddings) == len(texts)
    assert embeddings[0] is not None