"""
Test script for Advanced RAG modules.
Run with: python test_advanced_rag.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Advanced RAG Imports")
    print("=" * 60)
    
    try:
        from src.retrieval import (
            BM25Index,
            Reranker,
            HybridRetriever,
            SemanticChunker,
            ParentChildChunker,
            AdvancedChunker,
            QueryTransformer,
            CitationExtractor,
            AdvancedRAGPipeline,
        )
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_bm25():
    """Test BM25 index."""
    print("\n" + "=" * 60)
    print("Testing BM25 Index")
    print("=" * 60)
    
    try:
        from src.retrieval import BM25Index
        
        bm25 = BM25Index()
        
        # Test documents
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Python is a popular programming language for AI.",
        ]
        doc_ids = ["1", "2", "3"]
        metadatas = [
            {"source": "ml.txt"},
            {"source": "dl.txt"},
            {"source": "python.txt"},
        ]
        
        # Add documents
        bm25.add_documents(docs, doc_ids, metadatas)
        print(f"✅ Added {len(docs)} documents")
        
        # Search - returns tuple (docs, ids, metas, scores)
        result_docs, result_ids, result_metas, result_scores = bm25.search("machine learning", k=2)
        print(f"✅ Search returned {len(result_docs)} results")
        
        for doc, score in zip(result_docs, result_scores):
            print(f"   - Score: {score:.4f}, Doc: {doc[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ BM25 error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunking():
    """Test chunking strategies."""
    print("\n" + "=" * 60)
    print("Testing Advanced Chunking")
    print("=" * 60)
    
    try:
        from src.retrieval import SemanticChunker, ParentChildChunker
        
        text = """
        Machine learning is a branch of artificial intelligence. It allows computers to learn from data.
        
        Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
        These networks can learn complex patterns in large datasets.
        
        Natural language processing enables computers to understand human language.
        It combines linguistics with machine learning techniques.
        Applications include translation, sentiment analysis, and chatbots.
        """
        
        # Test semantic chunker
        semantic = SemanticChunker(max_chunk_size=100)
        semantic_chunks = semantic.chunk(text)
        print(f"✅ Semantic chunker: {len(semantic_chunks)} chunks")
        
        # Test parent-child chunker (using correct param names)
        parent_child = ParentChildChunker(parent_size=200, child_size=50)
        pc_chunks = parent_child.chunk(text)
        print(f"✅ Parent-Child chunker: {len(pc_chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_transform():
    """Test query transformation (requires LLM)."""
    print("\n" + "=" * 60)
    print("Testing Query Transformation")
    print("=" * 60)
    
    try:
        from src.retrieval import QueryTransformer
        
        transformer = QueryTransformer()
        
        # Test basic initialization
        print(f"✅ Query transformer initialized")
        print(f"   - Rewrite enabled: {transformer.rewrite_enabled}")
        print(f"   - Expansion enabled: {transformer.expansion_enabled}")
        
        # Note: Actual transformation requires LLM
        print("   - (Full test requires LLM connection)")
        
        return True
        
    except Exception as e:
        print(f"❌ Query transform error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citations():
    """Test citation extraction."""
    print("\n" + "=" * 60)
    print("Testing Citation System")
    print("=" * 60)
    
    try:
        from src.retrieval import CitationExtractor, format_sources_markdown
        
        extractor = CitationExtractor()
        
        # Test documents
        docs = [
            {
                "id": "chunk1",
                "document": "Machine learning is a subset of AI.",
                "metadata": {"source": "ml_book.pdf", "page": 1}
            },
            {
                "id": "chunk2", 
                "document": "Deep learning uses neural networks.",
                "metadata": {"source": "dl_guide.txt", "paragraph": 3}
            }
        ]
        
        scores = [0.85, 0.72]
        
        citations = extractor.extract_citations(docs, scores)
        print(f"✅ Extracted {len(citations)} citations")
        
        for c in citations:
            print(f"   - {c.source_file} | Score: {c.relevance_score:.2f}")
        
        # Test markdown formatting
        md = format_sources_markdown(citations)
        print(f"✅ Markdown formatted ({len(md)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reranker_init():
    """Test reranker initialization (without model loading)."""
    print("\n" + "=" * 60)
    print("Testing Reranker Initialization")
    print("=" * 60)
    
    try:
        from src.retrieval import Reranker
        
        reranker = Reranker()
        print(f"✅ Reranker initialized")
        print(f"   - Model: {reranker.model_name}")
        print(f"   - Device: {reranker.device}")
        print(f"   - Model loaded: {reranker.model is not None}")
        print("   - (Full test requires GPU memory)")
        
        return True
        
    except Exception as e:
        print(f"❌ Reranker error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 ADVANCED RAG MODULE TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    results["imports"] = test_imports()
    results["bm25"] = test_bm25()
    results["chunking"] = test_chunking()
    results["query_transform"] = test_query_transform()
    results["citations"] = test_citations()
    results["reranker"] = test_reranker_init()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n   Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
