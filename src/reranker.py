"""
Reranker Module for Vietnamese Legal RAG

Uses Cross-Encoder model to re-rank retrieved documents based on
relevance to the query. This provides more accurate ranking than
initial retrieval methods.

Supports:
- Vietnamese Cross-Encoder models (bkai-foundation-models/vietnamese-bi-encoder)
- Multilingual Cross-Encoder fallback (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Custom scoring with RRF integration
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Default models
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", 
    "thanhtantran/Vietnamese_Reranker"  # Vietnamese-specific reranker
)

# Alternative Vietnamese models:
# - "thanhtantran/Vietnamese_Reranker" - Specifically for Vietnamese (RECOMMENDED)
# - "itdainb/PhoRanker" - Another Vietnamese option
# - "BAAI/bge-reranker-v2-m3" - Multilingual (good for Vietnamese)
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" - Fast but English-focused


@dataclass
class RerankResult:
    """Result from reranking"""
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]
    rank: int


class LegalReranker:
    """
    Cross-Encoder Reranker for Vietnamese Legal Documents
    
    Re-ranks documents based on semantic relevance to query using
    cross-encoder architecture (query-document pair scoring).
    """
    
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        use_gpu: bool = True,
        max_length: int = 512,
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.device = "cpu"
    
    def load(self):
        """Load the reranker model"""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"[RERANKER] Loading {self.model_name}...")
        start = time.time()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine device first
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            # Load model directly to target device to avoid meta tensor issues
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            
            self._loaded = True
            elapsed = time.time() - start
            logger.info(f"[RERANKER] Loaded on {self.device} in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"[RERANKER] Failed to load model: {e}")
            # Will use fallback scoring
            self._loaded = False
    
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs
        
        Args:
            query: Search query
            documents: List of document texts
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not self._loaded:
            self.load()
        
        if not self._loaded or not documents:
            return [0.0] * len(documents)
        
        import torch
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            # Create query-document pairs
            pairs = [[query, doc[:2000]] for doc in batch_docs]  # Truncate long docs
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Compute scores
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Get logits (for cross-encoder, usually single output)
                    if hasattr(outputs, 'logits'):
                        batch_scores = outputs.logits.squeeze(-1)
                    else:
                        batch_scores = outputs[0].squeeze(-1)
                    
                    # Normalize to 0-1 range using sigmoid
                    batch_scores = torch.sigmoid(batch_scores)
                    scores.extend(batch_scores.cpu().tolist())
                    
            except Exception as e:
                logger.warning(f"[RERANKER] Batch scoring failed: {e}")
                # Fallback: return neutral scores
                scores.extend([0.5] * len(batch_docs))
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_key: str = "content",
        score_key: str = "score",
        rrf_weight: float = 0.5
    ) -> List[RerankResult]:
        """
        Re-rank documents using cross-encoder scores
        
        Combines original retrieval score with reranker score using RRF-style fusion.
        
        Args:
            query: Search query
            documents: List of document dicts with content and scores
            top_k: Number of top results to return (None = all)
            content_key: Key for document content in dict
            score_key: Key for original score in dict
            rrf_weight: Weight for combining scores (0=original, 1=rerank)
            
        Returns:
            List of RerankResult sorted by final score
        """
        if not documents:
            return []
        
        start = time.time()
        
        # Extract contents
        contents = [doc.get(content_key, "") for doc in documents]
        original_scores = [doc.get(score_key, 0.0) for doc in documents]
        
        # Compute reranker scores
        rerank_scores = self.compute_scores(query, contents)
        
        # Combine scores using weighted combination
        results = []
        for i, doc in enumerate(documents):
            # Normalize original score to 0-1 range if needed
            orig_score = original_scores[i]
            if orig_score > 1.0:
                orig_score = min(1.0, orig_score / 2.0)  # Simple normalization
            
            rerank_score = rerank_scores[i]
            
            # Weighted combination
            final_score = (1 - rrf_weight) * orig_score + rrf_weight * rerank_score
            
            results.append(RerankResult(
                content=contents[i],
                original_score=original_scores[i],
                rerank_score=rerank_score,
                final_score=final_score,
                metadata=doc,
                rank=0  # Will be set after sorting
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        elapsed = time.time() - start
        logger.info(f"[RERANKER] Re-ranked {len(results)} documents in {elapsed:.2f}s")
        
        if top_k:
            return results[:top_k]
        return results


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion algorithm
    
    Combines multiple rankings into a single ranking using RRF formula:
    RRF(d) = Σ 1 / (k + rank(d))
    
    Args:
        rankings: List of rankings, each is [(doc_id, score), ...]
        k: Constant for RRF formula (default 60)
        
    Returns:
        Fused ranking as [(doc_id, rrf_score), ...]
    """
    rrf_scores = {}
    
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


class HybridReranker:
    """
    Advanced hybrid search with RRF and Cross-Encoder reranking
    
    Pipeline:
    1. Dense search (semantic) -> ranking 1
    2. Sparse search (BM25) -> ranking 2
    3. RRF fusion -> combined ranking
    4. Cross-Encoder rerank -> final ranking
    """
    
    def __init__(
        self,
        reranker_model: str = RERANKER_MODEL,
        use_reranker: bool = True,
        rrf_k: int = 60,
        rerank_weight: float = 0.6
    ):
        self.rrf_k = rrf_k
        self.rerank_weight = rerank_weight
        self.use_reranker = use_reranker
        
        self.reranker = None
        if use_reranker:
            self.reranker = LegalReranker(model_name=reranker_model)
    
    def fuse_and_rerank(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 10,
        id_key: str = "chunk_id",
        content_key: str = "content",
        score_key: str = "score"
    ) -> List[RerankResult]:
        """
        Fuse dense and sparse results with RRF, then rerank
        
        Args:
            query: Search query
            dense_results: Results from dense vector search
            sparse_results: Results from sparse (BM25) search
            top_k: Number of final results
            id_key: Key for document ID
            content_key: Key for content
            score_key: Key for score
            
        Returns:
            Final reranked results
        """
        # Prepare rankings for RRF
        dense_ranking = [
            (r.get(id_key, str(i)), r.get(score_key, 0.0))
            for i, r in enumerate(dense_results)
        ]
        sparse_ranking = [
            (r.get(id_key, str(i)), r.get(score_key, 0.0))
            for i, r in enumerate(sparse_results)
        ]
        
        # Apply RRF
        rrf_results = reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking],
            k=self.rrf_k
        )
        
        # Build doc_id to document mapping
        all_docs = {}
        for r in dense_results:
            doc_id = r.get(id_key, "")
            if doc_id:
                all_docs[doc_id] = r
        for r in sparse_results:
            doc_id = r.get(id_key, "")
            if doc_id and doc_id not in all_docs:
                all_docs[doc_id] = r
        
        # Get top candidates from RRF
        rrf_candidates = []
        for doc_id, rrf_score in rrf_results[:top_k * 2]:  # Get more for reranking
            if doc_id in all_docs:
                doc = all_docs[doc_id].copy()
                doc["rrf_score"] = rrf_score
                rrf_candidates.append(doc)
        
        # Apply reranker if enabled
        if self.use_reranker and self.reranker:
            results = self.reranker.rerank(
                query=query,
                documents=rrf_candidates,
                top_k=top_k,
                content_key=content_key,
                score_key="rrf_score",
                rrf_weight=self.rerank_weight
            )
            return results
        else:
            # Just return RRF results without reranking
            return [
                RerankResult(
                    content=doc.get(content_key, ""),
                    original_score=doc.get(score_key, 0.0),
                    rerank_score=doc.get("rrf_score", 0.0),
                    final_score=doc.get("rrf_score", 0.0),
                    metadata=doc,
                    rank=i + 1
                )
                for i, doc in enumerate(rrf_candidates[:top_k])
            ]


# Singleton instance
_reranker_instance = None


def get_reranker(use_reranker: bool = True) -> Optional[HybridReranker]:
    """Get or create reranker singleton"""
    global _reranker_instance
    
    if not use_reranker:
        return None
    
    if _reranker_instance is None:
        _reranker_instance = HybridReranker(use_reranker=use_reranker)
    
    return _reranker_instance


if __name__ == "__main__":
    # Test reranker
    logging.basicConfig(level=logging.INFO)
    
    reranker = LegalReranker()
    
    query = "Hình phạt cho tội giết người là gì?"
    documents = [
        {"content": "Điều 123. Tội giết người. Người nào giết người thì bị phạt tù từ 12 năm đến 20 năm, tù chung thân hoặc tử hình.", "score": 0.9},
        {"content": "Điều 134. Tội cố ý gây thương tích. Người nào cố ý gây thương tích cho người khác...", "score": 0.7},
        {"content": "Điều 155. Tội làm nhục người khác. Người nào xúc phạm nghiêm trọng nhân phẩm...", "score": 0.5},
    ]
    
    results = reranker.rerank(query, documents)
    
    print("\nRerank Results:")
    for r in results:
        print(f"Rank {r.rank}: score={r.final_score:.3f} (orig={r.original_score:.3f}, rerank={r.rerank_score:.3f})")
        print(f"  Content: {r.content[:100]}...")
