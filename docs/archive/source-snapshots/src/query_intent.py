"""
Query Intent Classifier for Legal RAG

Classifies queries into categories to optimize retrieval strategy:
- LOOKUP: Tra cứu Điều luật cụ thể (prioritize exact match, BM25)
- EXPLAIN: Giải thích quy định (prioritize semantic, dense)
- CONDITION: Hỏi điều kiện, thủ tục (balanced)
- COMPARE: So sánh, đối chiếu (need multiple docs)
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent categories"""
    LOOKUP = "lookup"           # Tra cứu Điều luật cụ thể
    EXPLAIN = "explain"         # Giải thích ý nghĩa, khái niệm
    CONDITION = "condition"     # Điều kiện, thủ tục, yêu cầu
    COMPARE = "compare"         # So sánh, đối chiếu
    GENERAL = "general"         # Câu hỏi chung


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: QueryIntent
    confidence: float
    entities: dict  # Extracted entities (Điều số, văn bản, etc.)
    
    # Search weights based on intent
    @property
    def dense_weight(self) -> float:
        weights = {
            QueryIntent.LOOKUP: 0.3,      # More BM25
            QueryIntent.EXPLAIN: 0.7,     # More semantic
            QueryIntent.CONDITION: 0.5,   # Balanced
            QueryIntent.COMPARE: 0.6,     # Slightly semantic
            QueryIntent.GENERAL: 0.5
        }
        return weights.get(self.intent, 0.5)
    
    @property
    def sparse_weight(self) -> float:
        return 1.0 - self.dense_weight
    
    @property
    def top_k(self) -> int:
        """Recommended top_k based on intent"""
        top_ks = {
            QueryIntent.LOOKUP: 5,        # Fewer, more precise
            QueryIntent.EXPLAIN: 10,      # More context
            QueryIntent.CONDITION: 8,
            QueryIntent.COMPARE: 15,      # Need multiple sources
            QueryIntent.GENERAL: 10
        }
        return top_ks.get(self.intent, 10)


class QueryIntentClassifier:
    """
    Rule-based Query Intent Classifier for Vietnamese Legal Queries
    
    Uses keyword patterns and entity extraction to classify query intent.
    """
    
    def __init__(self):
        # Patterns for LOOKUP intent (tra cứu Điều cụ thể)
        self.lookup_patterns = [
            r'điều\s*\d+',
            r'khoản\s*\d+',
            r'điểm\s*[a-z]',
            r'quy định\s+tại',
            r'theo\s+điều',
            r'căn cứ',
            r'chiếu\s+theo',
            r'tra cứu',
            r'tìm\s+điều',
            r'nội dung\s+(điều|khoản)',
            r'luật\s+\w+\s+năm\s+\d{4}',
            r'nghị định\s+\d+',
            r'thông tư\s+\d+',
        ]
        
        # Patterns for EXPLAIN intent
        self.explain_patterns = [
            r'là gì',
            r'nghĩa là',
            r'có nghĩa',
            r'thế nào',
            r'như thế nào',
            r'giải thích',
            r'định nghĩa',
            r'khái niệm',
            r'hiểu\s+(như|thế)',
            r'tại sao',
            r'vì sao',
            r'lý do',
            r'mục đích',
            r'ý nghĩa',
        ]
        
        # Patterns for CONDITION intent
        self.condition_patterns = [
            r'điều kiện',
            r'thủ tục',
            r'yêu cầu',
            r'cần (những )?gì',
            r'phải (có|làm)',
            r'hồ sơ',
            r'giấy tờ',
            r'quy trình',
            r'các bước',
            r'làm (thế )?nào',
            r'cách (để )?',
            r'được phép',
            r'không được',
            r'bị cấm',
            r'cho phép',
            r'xử phạt',
            r'mức phạt',
            r'thời hạn',
            r'thời gian',
        ]
        
        # Patterns for COMPARE intent
        self.compare_patterns = [
            r'khác (nhau|biệt)',
            r'so sánh',
            r'giống (nhau)?',
            r'tương tự',
            r'phân biệt',
            r'đối chiếu',
            r'so với',
            r'hay là',
            r'hoặc là',
            r'nào (tốt|đúng|phù hợp)',
        ]
        
        # Entity patterns
        self.entity_patterns = {
            'dieu_so': r'(?:điều|đ\.?)\s*(\d+)',
            'khoan_so': r'(?:khoản|k\.?)\s*(\d+)',
            'diem_so': r'(?:điểm)\s*([a-z])',
            'luat_name': r'luật\s+([^,\.]+?)(?:\s+năm\s+\d{4})?',
            'nghi_dinh': r'nghị định\s*(?:số\s*)?(\d+(?:/\d+)?(?:/NĐ-CP)?)',
            'thong_tu': r'thông tư\s*(?:số\s*)?(\d+(?:/\d+)?(?:/TT-\w+)?)',
            'nam': r'năm\s+(\d{4})',
        }
        
        logger.info("[INTENT] Query Intent Classifier initialized")
    
    def _match_patterns(self, query: str, patterns: List[str]) -> int:
        """Count pattern matches in query"""
        query_lower = query.lower()
        count = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                count += 1
        return count
    
    def _extract_entities(self, query: str) -> dict:
        """Extract legal entities from query"""
        query_lower = query.lower()
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                # Clean and store
                if entity_type == 'dieu_so':
                    entities[entity_type] = [f"Điều {m}" for m in matches]
                elif entity_type == 'khoan_so':
                    entities[entity_type] = [f"Khoản {m}" for m in matches]
                elif entity_type == 'diem_so':
                    entities[entity_type] = [f"Điểm {m}" for m in matches]
                else:
                    entities[entity_type] = matches
        
        return entities
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent
        
        Returns:
            IntentResult with intent, confidence, and entities
        """
        # Extract entities first
        entities = self._extract_entities(query)
        
        # Count pattern matches for each intent
        scores = {
            QueryIntent.LOOKUP: self._match_patterns(query, self.lookup_patterns),
            QueryIntent.EXPLAIN: self._match_patterns(query, self.explain_patterns),
            QueryIntent.CONDITION: self._match_patterns(query, self.condition_patterns),
            QueryIntent.COMPARE: self._match_patterns(query, self.compare_patterns),
        }
        
        # Boost LOOKUP if specific legal entities found
        if entities.get('dieu_so') or entities.get('nghi_dinh') or entities.get('thong_tu'):
            scores[QueryIntent.LOOKUP] += 2
        
        # Find best intent
        max_score = max(scores.values())
        
        if max_score == 0:
            intent = QueryIntent.GENERAL
            confidence = 0.5
        else:
            intent = max(scores, key=scores.get)
            # Normalize confidence (max reasonable score ~5)
            confidence = min(max_score / 5.0, 1.0)
        
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities
        )
        
        logger.info(f"[INTENT] Query: '{query[:50]}...' -> {intent.value} ({confidence:.2f})")
        logger.debug(f"[INTENT] Entities: {entities}")
        logger.debug(f"[INTENT] Weights: dense={result.dense_weight}, sparse={result.sparse_weight}")
        
        return result
    
    def get_retrieval_params(self, query: str) -> Tuple[float, float, int, dict]:
        """
        Get optimized retrieval parameters based on query intent
        
        Returns:
            (dense_weight, sparse_weight, top_k, filter_hints)
        """
        result = self.classify(query)
        
        # Build filter hints from entities
        filter_hints = {}
        
        if result.entities.get('nghi_dinh'):
            filter_hints['loai_van_ban'] = 'Nghị định'
        elif result.entities.get('thong_tu'):
            filter_hints['loai_van_ban'] = 'Thông tư'
        elif result.entities.get('luat_name'):
            filter_hints['loai_van_ban'] = 'Luật'
        
        return (
            result.dense_weight,
            result.sparse_weight,
            result.top_k,
            filter_hints
        )


# Singleton
_classifier: Optional[QueryIntentClassifier] = None

def get_intent_classifier() -> QueryIntentClassifier:
    """Get singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = QueryIntentClassifier()
    return _classifier


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    classifier = QueryIntentClassifier()
    
    # Test queries
    test_queries = [
        "Điều 128 Bộ luật Hình sự quy định gì?",
        "Thế nào là tội phạm nghiêm trọng?",
        "Điều kiện để được cấp giấy phép kinh doanh là gì?",
        "So sánh trách nhiệm hình sự và trách nhiệm hành chính",
        "Cho tôi biết về luật doanh nghiệp",
        "Nghị định 100/2019 về xử phạt vi phạm giao thông",
        "Khoản 2 Điều 15 quy định như thế nào?",
        "Thủ tục đăng ký kết hôn cần những gì?",
    ]
    
    print("\n" + "="*80)
    print("QUERY INTENT CLASSIFICATION TEST")
    print("="*80)
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Dense/Sparse: {result.dense_weight:.1f}/{result.sparse_weight:.1f}")
        print(f"  Top-K: {result.top_k}")
        if result.entities:
            print(f"  Entities: {result.entities}")
