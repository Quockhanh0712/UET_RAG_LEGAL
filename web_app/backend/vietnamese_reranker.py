"""
Vietnamese Reranker Module

Sử dụng model: AITeamVN/Vietnamese_Reranker
- Cross-encoder reranking cho Vietnamese text
- GPU accelerated với FP16
- Toggle on/off
- Batch processing

Usage:
    reranker = VietnameseReranker()
    reranker.load()
    results = reranker.rerank(query, documents, top_k=5)
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Single reranked result"""
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    rank_before: int
    rank_after: int
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "metadata": self.metadata or {}
        }


class VietnameseReranker:
    """
    Vietnamese Cross-Encoder Reranker
    
    Model: AITeamVN/Vietnamese_Reranker
    - Trained on Vietnamese legal/general text
    - Max length: 2304 tokens
    - Output: relevance score (higher = more relevant)
    
    Usage:
        reranker = VietnameseReranker(use_gpu=True)
        reranker.load()
        
        # Compute scores
        scores = reranker.compute_scores(query, documents)
        
        # Rerank results
        reranked = reranker.rerank(query, results, top_k=5, alpha=0.7)
    """
    
    MODEL_NAME = "AITeamVN/Vietnamese_Reranker"
    MAX_LENGTH = 2304
    
    def __init__(
        self,
        model_name: str = None,
        use_gpu: bool = True,
        max_length: int = 2304,
        batch_size: int = 8
    ):
        """
        Initialize Vietnamese Reranker
        
        Args:
            model_name: HuggingFace model name (default: AITeamVN/Vietnamese_Reranker)
            use_gpu: Use CUDA if available
            max_length: Max sequence length (default: 2304)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name or self.MODEL_NAME
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False
    
    def load(self) -> bool:
        """Load reranker model"""
        if self._loaded:
            logger.info("[RERANKER] Already loaded")
            return True
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            logger.info(f"[RERANKER] Loading {self.model_name}...")
            start = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu and torch.cuda.is_available() else torch.float32
            )
            
            # Move to GPU if available
            if self.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to(self.device)
                logger.info(f"[RERANKER] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("[RERANKER] Using CPU")
            
            self.model.eval()
            self._loaded = True
            
            elapsed = time.time() - start
            logger.info(f"[RERANKER] Loaded in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"[RERANKER] Failed to load: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
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
            logger.warning("[RERANKER] Model not loaded, loading now...")
            self.load()
        
        if not documents:
            return []
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            # Create query-document pairs
            pairs = [[query, doc] for doc in batch_docs]
            
            try:
                with torch.no_grad():
                    # Tokenize
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get scores
                    outputs = self.model(**inputs, return_dict=True)
                    batch_scores = outputs.logits.view(-1).float().cpu().tolist()
                    
                    scores.extend(batch_scores)
                    
            except Exception as e:
                logger.error(f"[RERANKER] Error scoring batch: {e}")
                # Return zeros for failed batch
                scores.extend([0.0] * len(batch_docs))
        
        # Normalize scores to 0-1 range using sigmoid
        normalized_scores = []
        for score in scores:
            # Sigmoid normalization
            norm_score = 1 / (1 + torch.exp(torch.tensor(-score)).item())
            normalized_scores.append(norm_score)
        
        return normalized_scores
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
        alpha: float = 0.5,
        content_key: str = "content",
        score_key: str = "score"
    ) -> Tuple[List[RerankedResult], float]:
        """
        Rerank search results using cross-encoder scores
        
        Final score = alpha * rerank_score + (1 - alpha) * original_score
        
        Args:
            query: Search query
            results: List of result dicts with content and score
            top_k: Number of results to return
            alpha: Weight for reranker score (0.5 = balanced, 50% reranker + 50% original)
            content_key: Key for document content in result dict
            score_key: Key for original score in result dict
            
        Returns:
            Tuple of (reranked_results, rerank_time)
        """
        if not results:
            return [], 0.0
        
        start = time.time()
        
        # Extract contents and original scores
        contents = [r.get(content_key, "") for r in results]
        original_scores = [r.get(score_key, 0.0) for r in results]
        
        # Compute reranker scores
        rerank_scores = self.compute_scores(query, contents)
        
        # Combine scores and create results
        reranked = []
        for i, (result, orig_score, rerank_score) in enumerate(zip(results, original_scores, rerank_scores)):
            final_score = alpha * rerank_score + (1 - alpha) * orig_score
            
            reranked.append(RerankedResult(
                content=result.get(content_key, ""),
                original_score=orig_score,
                rerank_score=rerank_score,
                final_score=final_score,
                rank_before=i + 1,
                rank_after=0,  # Will be set after sorting
                metadata=result.get("metadata") or result.get("payload") or {}
            ))
        
        # Sort by final score (descending)
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Update rank_after
        for i, r in enumerate(reranked):
            r.rank_after = i + 1
        
        rerank_time = time.time() - start
        logger.info(f"[RERANKER] Reranked {len(results)} docs in {rerank_time:.3f}s")
        
        return reranked[:top_k], rerank_time


class AnswerScorer:
    """
    Score generated answers against retrieved contexts
    
    Uses the same Vietnamese Reranker to compute semantic similarity
    between the generated answer and the retrieved documents.
    
    Metrics:
    - Answer-Context Relevance: How well answer matches context
    - Query-Answer Relevance: How well answer addresses query
    - Extractive Score: % of answer found in context
    """
    
    def __init__(self, reranker: VietnameseReranker = None):
        """
        Initialize Answer Scorer
        
        Args:
            reranker: VietnameseReranker instance (will create new if None)
        """
        self.reranker = reranker
    
    def _get_reranker(self) -> VietnameseReranker:
        """Get or create reranker"""
        if self.reranker is None:
            self.reranker = VietnameseReranker()
            self.reranker.load()
        return self.reranker
    
    def compute_extractive_score(
        self,
        answer: str,
        contexts: List[str],
        min_ngram: int = 3
    ) -> float:
        """
        Compute extractive score (% of answer n-grams found in contexts)
        
        Args:
            answer: Generated answer
            contexts: List of retrieved context texts
            min_ngram: Minimum n-gram size to check
            
        Returns:
            Float 0-1 (1 = 100% extractive)
        """
        if not answer or not contexts:
            return 0.0
        
        # Normalize
        answer_lower = answer.lower().strip()
        context_combined = " ".join(contexts).lower()
        
        # Split answer into words
        answer_words = answer_lower.split()
        
        if len(answer_words) < min_ngram:
            # Check if entire answer is in context
            return 1.0 if answer_lower in context_combined else 0.0
        
        # Check n-grams
        matches = 0
        total = 0
        
        for n in range(min_ngram, min(len(answer_words) + 1, 10)):
            for i in range(len(answer_words) - n + 1):
                ngram = " ".join(answer_words[i:i + n])
                total += 1
                if ngram in context_combined:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def score_answer(
        self,
        query: str,
        answer: str,
        contexts: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        [DEPRECATED] Use QualityMonitor.evaluate_answer() instead.
        
        This method is kept for backward compatibility with benchmarks.
        For production use, QualityMonitor (BERTScore-based) is the primary scorer.
        
        Args:
            query: Original user query
            answer: Generated answer
            contexts: List of retrieved context dicts
            top_k: Number of contexts to consider
            
        Returns:
            Dict with scores and metrics
        """
        if not answer or not contexts:
            return {
                "overall_score": 0.0,
                "query_answer_score": 0.0,
                "answer_context_score": 0.0,
                "extractive_score": 0.0,
                "context_scores": [],
                "grade": "F",
                "feedback": "Không có câu trả lời hoặc ngữ cảnh"
            }
        
        reranker = self._get_reranker()
        
        # Get context texts
        context_texts = [
            c.get("content", "") if isinstance(c, dict) else str(c)
            for c in contexts[:top_k]
        ]
        
        # 1. Query-Answer relevance
        query_answer_scores = reranker.compute_scores(query, [answer])
        query_answer_score = query_answer_scores[0] if query_answer_scores else 0.0
        
        # 2. Answer-Context relevance (answer vs each context)
        context_scores = reranker.compute_scores(answer, context_texts)
        answer_context_score = max(context_scores) if context_scores else 0.0
        avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0.0
        
        # 3. Extractive score
        extractive_score = self.compute_extractive_score(answer, context_texts)
        
        # 4. Overall score (weighted average)
        overall_score = (
            0.4 * query_answer_score +      # Answer addresses query
            0.4 * answer_context_score +    # Answer grounded in context
            0.2 * extractive_score          # Answer uses context
        )
        
        # 5. Grade
        if overall_score >= 0.8:
            grade = "A"
            feedback = "Câu trả lời xuất sắc, chính xác và đầy đủ căn cứ"
        elif overall_score >= 0.6:
            grade = "B"
            feedback = "Câu trả lời tốt, có căn cứ từ tài liệu"
        elif overall_score >= 0.4:
            grade = "C"
            feedback = "Câu trả lời trung bình, cần bổ sung thêm căn cứ"
        elif overall_score >= 0.2:
            grade = "D"
            feedback = "Câu trả lời yếu, thiếu căn cứ hoặc không chính xác"
        else:
            grade = "F"
            feedback = "Câu trả lời không phù hợp với ngữ cảnh"
        
        return {
            "overall_score": round(overall_score, 4),
            "query_answer_score": round(query_answer_score, 4),
            "answer_context_score": round(answer_context_score, 4),
            "avg_context_score": round(avg_context_score, 4),
            "extractive_score": round(extractive_score, 4),
            "context_scores": [round(s, 4) for s in context_scores],
            "grade": grade,
            "feedback": feedback
        }


# Singleton instances
_reranker_instance: Optional[VietnameseReranker] = None
_scorer_instance: Optional[AnswerScorer] = None


def get_reranker() -> VietnameseReranker:
    """Get singleton reranker instance"""
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = VietnameseReranker(use_gpu=True)
        _reranker_instance.load()
    
    return _reranker_instance


def get_answer_scorer() -> AnswerScorer:
    """Get singleton answer scorer instance"""
    global _scorer_instance
    
    if _scorer_instance is None:
        _scorer_instance = AnswerScorer(get_reranker())
    
    return _scorer_instance


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Vietnamese Reranker Test")
    print("=" * 60)
    
    # Test reranker
    reranker = get_reranker()
    
    query = "Trí tuệ nhân tạo là gì?"
    docs = [
        "Trí tuệ nhân tạo là công nghệ giúp máy móc suy nghĩ và học hỏi như con người.",
        "Giấc ngủ giúp cơ thể và não bộ nghỉ ngơi, hồi phục năng lượng.",
        "AI có thể được sử dụng trong nhiều lĩnh vực như y tế, giáo dục, pháp luật."
    ]
    
    scores = reranker.compute_scores(query, docs)
    print(f"\nQuery: {query}")
    print("\nScores:")
    for doc, score in zip(docs, scores):
        print(f"  {score:.4f}: {doc[:60]}...")
    
    # Test answer scorer
    print("\n" + "=" * 60)
    print("Answer Scorer Test")
    print("=" * 60)
    
    scorer = get_answer_scorer()
    
    answer = "Trí tuệ nhân tạo là công nghệ giúp máy móc có khả năng học hỏi và suy nghĩ."
    contexts = [{"content": doc} for doc in docs]
    
    result = scorer.score_answer(query, answer, contexts)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    print(f"\nScores:")
    for k, v in result.items():
        print(f"  {k}: {v}")
