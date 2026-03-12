"""
Quality Monitor Module
=====================
Automatic answer quality evaluation using BERTScore and semantic analysis.

Features:
- BERTScore semantic similarity measurement
- Hallucination detection (claims not in context)
- Factuality verification (accuracy vs source)
- Context relevance scoring
- Vietnamese feedback generation
- Integration with database for metrics tracking

Author: AI Assistant
Date: 2025-12-27
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for answer quality metrics"""
    overall_score: float
    bert_score: float
    hallucination_score: float
    factuality_score: float
    context_relevance: float
    grade: str
    feedback: str
    details: Dict[str, Any]


class QualityMonitor:
    """
    Monitor and evaluate answer quality using semantic analysis
    """
    
    def __init__(self):
        """Initialize Quality Monitor"""
        self.bert_scorer = None
        self._bert_scorer_initialized = False
        logger.info("✅ Quality Monitor initialized")
    
    def _init_bert_scorer(self):
        """Lazy initialization of BERTScorer"""
        if self._bert_scorer_initialized:
            return
        
        try:
            from bert_score import BERTScorer
            # Use multilingual BERT for Vietnamese (without baseline rescaling)
            self.bert_scorer = BERTScorer(
                model_type="bert-base-multilingual-cased",
                lang="vi",
                rescale_with_baseline=False,
                device="cuda"
            )
            self._bert_scorer_initialized = True
            logger.info("✅ BERTScorer initialized with multilingual-BERT")
        except ImportError:
            logger.error("❌ bert-score not installed. Install with: pip install bert-score")
            self._bert_scorer_initialized = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize BERTScorer: {e}")
            self._bert_scorer_initialized = False
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        prompt: str,
        top_k: int = 5
    ) -> QualityMetrics:
        """
        Evaluate answer quality across multiple dimensions
        
        Args:
            query: User's question
            answer: Generated answer
            contexts: Top-k context documents used
            prompt: Full prompt sent to LLM
            top_k: Number of contexts used
        
        Returns:
            QualityMetrics with all evaluation scores
        """
        logger.info(f"🔍 Evaluating answer quality (top_k={top_k})")
        
        # Combine contexts for analysis
        combined_context = " ".join(contexts[:top_k])
        
        # 1. BERTScore: Semantic similarity between answer and contexts
        bert_score = self._compute_bert_score(answer, combined_context)
        
        # 2. Hallucination Detection: Claims not supported by context
        hallucination_score = self._detect_hallucinations(answer, combined_context)
        
        # 3. Factuality: Answer accuracy vs source material
        factuality_score = self._check_factuality(answer, contexts[:top_k])
        
        # 4. Context Relevance: How well contexts match query
        context_relevance = self._compute_context_relevance(query, combined_context)
        
        # 5. Overall Score: Weighted combination
        overall_score = self._compute_overall_score(
            bert_score, hallucination_score, factuality_score, context_relevance
        )
        
        # 6. Grade and Feedback
        grade, feedback = self._generate_grade_and_feedback(
            overall_score, bert_score, hallucination_score, 
            factuality_score, context_relevance
        )
        
        # Collect details
        details = {
            "query_length": len(query),
            "answer_length": len(answer),
            "contexts_count": len(contexts),
            "combined_context_length": len(combined_context),
            "answer_sentences": len(self._split_sentences(answer)),
            "context_sentences": len(self._split_sentences(combined_context))
        }
        
        metrics = QualityMetrics(
            overall_score=overall_score,
            bert_score=bert_score,
            hallucination_score=hallucination_score,
            factuality_score=factuality_score,
            context_relevance=context_relevance,
            grade=grade,
            feedback=feedback,
            details=details
        )
        
        logger.info(f"✅ Evaluation complete: Grade={grade}, Overall={overall_score:.3f}")
        return metrics
    
    def _compute_bert_score(self, answer: str, context: str) -> float:
        """
        Compute BERTScore F1 between answer and context
        
        Returns:
            Score between 0.0-1.0
        """
        if not answer or not context:
            return 0.0
        
        try:
            # Initialize BERTScorer if needed
            self._init_bert_scorer()
            
            if not self._bert_scorer_initialized or self.bert_scorer is None:
                # Fallback to simple overlap if BERTScore unavailable
                logger.warning("⚠️ BERTScore unavailable, using fallback method")
                return self._compute_token_overlap(answer, context)
            
            # Compute BERTScore
            P, R, F1 = self.bert_scorer.score([answer], [context])
            score = float(F1[0].item())
            
            logger.debug(f"BERTScore: P={P[0]:.3f}, R={R[0]:.3f}, F1={score:.3f}")
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ BERTScore computation failed: {e}")
            # Fallback to token overlap
            return self._compute_token_overlap(answer, context)
    
    def _compute_token_overlap(self, text1: str, text2: str) -> float:
        """
        Fallback method: Compute token overlap ratio
        
        Returns:
            Jaccard similarity between 0.0-1.0
        """
        tokens1 = set(self._tokenize(text1.lower()))
        tokens2 = set(self._tokenize(text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_hallucinations(self, answer: str, context: str) -> float:
        """
        Detect hallucinations: claims in answer not supported by context
        
        Returns:
            Score between 0.0-1.0 (higher = less hallucination)
        """
        if not answer or not context:
            return 1.0
        
        # Split answer into sentences
        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return 1.0
        
        # For each sentence, check if it's grounded in context
        grounded_count = 0
        context_lower = context.lower()
        
        for sentence in answer_sentences:
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue
            
            # Extract key phrases (3+ words)
            key_phrases = self._extract_key_phrases(sentence)
            
            # Check if at least some key phrases appear in context
            matches = sum(1 for phrase in key_phrases if phrase in context_lower)
            
            if matches > 0 or len(key_phrases) == 0:
                grounded_count += 1
        
        # Ratio of grounded sentences
        total_sentences = len([s for s in answer_sentences if len(s.strip()) >= 10])
        if total_sentences == 0:
            return 1.0
        
        score = grounded_count / total_sentences
        logger.debug(f"Hallucination check: {grounded_count}/{total_sentences} grounded")
        
        return max(0.0, min(1.0, score))
    
    def _check_factuality(self, answer: str, contexts: List[str]) -> float:
        """
        Check factual consistency between answer and source contexts
        
        Returns:
            Score between 0.0-1.0 (higher = more factual)
        """
        if not answer or not contexts:
            return 0.5
        
        # Extract numeric facts, dates, and specific terms
        answer_facts = self._extract_facts(answer)
        
        if not answer_facts:
            # No specific facts to verify, assume reasonable
            return 0.7
        
        # Check how many facts appear in contexts
        verified_count = 0
        for fact in answer_facts:
            for context in contexts:
                if fact.lower() in context.lower():
                    verified_count += 1
                    break
        
        score = verified_count / len(answer_facts)
        logger.debug(f"Factuality check: {verified_count}/{len(answer_facts)} facts verified")
        
        return max(0.0, min(1.0, score))
    
    def _compute_context_relevance(self, query: str, context: str) -> float:
        """
        Compute how relevant the context is to the query
        
        Returns:
            Score between 0.0-1.0
        """
        if not query or not context:
            return 0.0
        
        # Tokenize query and context
        query_tokens = set(self._tokenize(query.lower()))
        context_tokens = self._tokenize(context.lower())
        
        if not query_tokens:
            return 0.0
        
        # Count query token appearances in context
        matches = sum(1 for token in query_tokens if token in context_tokens)
        
        # Ratio of query tokens found
        score = matches / len(query_tokens)
        
        logger.debug(f"Context relevance: {matches}/{len(query_tokens)} query tokens found")
        return max(0.0, min(1.0, score))
    
    def _compute_overall_score(
        self,
        bert_score: float,
        hallucination_score: float,
        factuality_score: float,
        context_relevance: float
    ) -> float:
        """
        Compute weighted overall quality score
        
        Weights:
        - BERTScore: 35% (semantic similarity)
        - Hallucination: 30% (avoid false claims)
        - Factuality: 25% (accuracy)
        - Context Relevance: 10% (source quality)
        """
        weights = {
            'bert': 0.35,
            'hallucination': 0.30,
            'factuality': 0.25,
            'relevance': 0.10
        }
        
        overall = (
            bert_score * weights['bert'] +
            hallucination_score * weights['hallucination'] +
            factuality_score * weights['factuality'] +
            context_relevance * weights['relevance']
        )
        
        return max(0.0, min(1.0, overall))
    
    def _generate_grade_and_feedback(
        self,
        overall_score: float,
        bert_score: float,
        hallucination_score: float,
        factuality_score: float,
        context_relevance: float
    ) -> Tuple[str, str]:
        """
        Generate letter grade (A-F) and Vietnamese feedback
        
        Returns:
            (grade, feedback) tuple
        """
        # Determine grade
        if overall_score >= 0.9:
            grade = "A"
            base_feedback = "Xuất sắc! Câu trả lời chất lượng cao"
        elif overall_score >= 0.8:
            grade = "B"
            base_feedback = "Tốt! Câu trả lời đạt yêu cầu"
        elif overall_score >= 0.7:
            grade = "C"
            base_feedback = "Khá! Câu trả lời cần cải thiện"
        elif overall_score >= 0.6:
            grade = "D"
            base_feedback = "Trung bình! Câu trả lời có nhiều vấn đề"
        else:
            grade = "F"
            base_feedback = "Kém! Câu trả lời không đạt yêu cầu"
        
        # Add specific issues
        issues = []
        if bert_score < 0.7:
            issues.append("độ tương đồng ngữ nghĩa thấp")
        if hallucination_score < 0.7:
            issues.append("có dấu hiệu ảo giác (hallucination)")
        if factuality_score < 0.7:
            issues.append("độ chính xác thông tin chưa cao")
        if context_relevance < 0.5:
            issues.append("tài liệu nguồn không đủ liên quan")
        
        if issues:
            feedback = f"{base_feedback}. Vấn đề: {', '.join(issues)}."
        else:
            feedback = f"{base_feedback}. Không phát hiện vấn đề nghiêm trọng."
        
        return grade, feedback
    
    # Utility methods
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter for Vietnamese
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenizer"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases (3+ word sequences)"""
        words = self._tokenize(text.lower())
        phrases = []
        
        # Extract n-grams (n=3 to 5)
        for n in range(3, 6):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                phrases.append(phrase)
        
        return phrases
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual elements: numbers, dates, specific terms"""
        facts = []
        
        # Numbers (including dates)
        numbers = re.findall(r'\d+[.,/\d]*', text)
        facts.extend(numbers)
        
        # Vietnamese legal terms (common patterns)
        legal_patterns = [
            r'Điều \d+',
            r'Khoản \d+',
            r'Luật số \d+',
            r'Nghị định số \d+',
            r'Thông tư số \d+',
            r'Quyết định số \d+'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            facts.extend(matches)
        
        return list(set(facts))  # Remove duplicates


# Singleton instance
_quality_monitor_instance = None


def get_quality_monitor() -> QualityMonitor:
    """Get or create Quality Monitor singleton"""
    global _quality_monitor_instance
    if _quality_monitor_instance is None:
        _quality_monitor_instance = QualityMonitor()
    return _quality_monitor_instance


# Test function
def test_quality_monitor():
    """Test Quality Monitor functionality"""
    monitor = get_quality_monitor()
    
    # Test case
    query = "Điều 10 Luật Đất đai quy định gì?"
    answer = "Theo Điều 10 Luật Đất đai năm 2013, đất đai thuộc sở hữu toàn dân do Nhà nước đại diện chủ sở hữu và thống nhất quản lý."
    contexts = [
        "Điều 10. Chế độ sở hữu đất đai. Đất đai thuộc sở hữu toàn dân do Nhà nước đại diện chủ sở hữu và thống nhất quản lý.",
        "Nhà nước giao đất, cho thuê đất, công nhận quyền sử dụng đất cho các đối tượng theo quy định."
    ]
    prompt = f"Câu hỏi: {query}\n\nTài liệu tham khảo:\n{contexts[0]}\n\nTrả lời:"
    
    print("\n" + "="*80)
    print("TESTING QUALITY MONITOR")
    print("="*80)
    
    metrics = monitor.evaluate_answer(query, answer, contexts, prompt, top_k=2)
    
    print(f"\n📊 Quality Metrics:")
    print(f"   Overall Score: {metrics.overall_score:.3f}")
    print(f"   Grade: {metrics.grade}")
    print(f"   Feedback: {metrics.feedback}")
    print(f"\n📈 Detailed Scores:")
    print(f"   BERTScore: {metrics.bert_score:.3f}")
    print(f"   Hallucination Score: {metrics.hallucination_score:.3f}")
    print(f"   Factuality Score: {metrics.factuality_score:.3f}")
    print(f"   Context Relevance: {metrics.context_relevance:.3f}")
    print(f"\n📝 Details:")
    for key, value in metrics.details.items():
        print(f"   {key}: {value}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run test
    test_quality_monitor()
