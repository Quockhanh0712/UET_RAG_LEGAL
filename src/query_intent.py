"""
Query Intent Classifier for Legal RAG System
Classifies user queries to adjust search weights and retrieval strategy
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of legal query intents"""
    DEFINITION = "definition"      # "X là gì?", định nghĩa
    PENALTY = "penalty"            # Hỏi về mức phạt, hình phạt
    PROCEDURE = "procedure"        # Thủ tục, quy trình
    CONDITION = "condition"        # Điều kiện, yêu cầu
    COMPARISON = "comparison"      # So sánh, khác biệt
    CASE_SPECIFIC = "case"         # Tình huống cụ thể
    AMENDMENT = "amendment"        # Truy vấn sửa đổi/thay thế/bãi bỏ
    TEMPORAL = "temporal"          # Truy vấn hiệu lực theo thời điểm
    GENERAL = "general"            # Câu hỏi chung


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: QueryIntent
    confidence: float
    dense_weight: float = 0.6    # Weight for dense vector search
    sparse_weight: float = 0.4   # Weight for BM25 sparse search
    top_k: int = 10              # Recommended number of results
    graph_mode: str = "none"     # none | local | legal_impact | temporal
    
    @property
    def intent_name(self) -> str:
        return self.intent.value


# Intent patterns with corresponding weights
INTENT_PATTERNS = {
    QueryIntent.DEFINITION: {
        "patterns": [
            r"là gì\??$",
            r"định nghĩa",
            r"khái niệm",
            r"nghĩa là",
            r"được hiểu như thế nào",
            r"giải thích"
        ],
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "top_k": 5,
        "graph_mode": "none",
    },
    QueryIntent.PENALTY: {
        "patterns": [
            r"bị phạt",
            r"mức phạt",
            r"hình phạt",
            r"xử phạt",
            r"phạt bao nhiêu",
            r"phạt tù",
            r"phạt tiền",
            r"bị xử lý",
            r"chịu trách nhiệm"
        ],
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "top_k": 10,
        "graph_mode": "none",
    },
    QueryIntent.PROCEDURE: {
        "patterns": [
            r"thủ tục",
            r"quy trình",
            r"các bước",
            r"cách thức",
            r"làm thế nào",
            r"như thế nào",
            r"hồ sơ cần",
            r"giấy tờ cần"
        ],
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "top_k": 8,
        "graph_mode": "none",
    },
    QueryIntent.CONDITION: {
        "patterns": [
            r"điều kiện",
            r"yêu cầu",
            r"khi nào",
            r"trường hợp nào",
            r"được phép",
            r"có quyền",
            r"phải đáp ứng"
        ],
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "top_k": 8,
        "graph_mode": "none",
    },
    QueryIntent.COMPARISON: {
        "patterns": [
            r"khác nhau",
            r"giống nhau",
            r"so sánh",
            r"phân biệt",
            r"khác biệt",
            r"hay là",
            r"hoặc"
        ],
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "top_k": 15,
        "graph_mode": "local",
    },
    QueryIntent.CASE_SPECIFIC: {
        "patterns": [
            r"tình huống",
            r"trường hợp",
            r"ví dụ",
            r"cụ thể",
            r"nếu tôi",
            r"khi tôi",
            r"giả sử"
        ],
        "dense_weight": 0.65,
        "sparse_weight": 0.35,
        "top_k": 10,
        "graph_mode": "local",
    },
    QueryIntent.AMENDMENT: {
        "patterns": [
            r"sửa đổi",
            r"bo sung",
            r"bổ sung",
            r"thay thế",
            r"bãi bỏ",
            r"hết hiệu lực",
            r"văn bản nào sửa",
            r"được thay bởi",
            r"được sửa bởi"
        ],
        "dense_weight": 0.4,
        "sparse_weight": 0.6,
        "top_k": 12,
        "graph_mode": "legal_impact",
    },
    QueryIntent.TEMPORAL: {
        "patterns": [
            r"còn hiệu lực",
            r"đến ngày",
            r"tại ngày",
            r"as of",
            r"thời điểm",
            r"năm \d{4}",
            r"trước năm",
            r"sau năm"
        ],
        "dense_weight": 0.45,
        "sparse_weight": 0.55,
        "top_k": 12,
        "graph_mode": "temporal",
    }
}


class QueryIntentClassifier:
    """Classifies legal queries to optimize search strategy"""
    
    def __init__(self):
        self._compiled_patterns = {}
        self._compile_patterns()
        logger.info("[INTENT] QueryIntentClassifier initialized")
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        for intent, config in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE | re.UNICODE)
                for pattern in config["patterns"]
            ]
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent and return optimized search parameters
        
        Args:
            query: User's question
            
        Returns:
            IntentResult with intent type and search weights
        """
        query_lower = query.lower().strip()
        
        # Check each intent pattern
        best_intent = QueryIntent.GENERAL
        best_confidence = 0.0
        
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    # Found a match
                    config = INTENT_PATTERNS[intent]
                    confidence = 0.8 + (0.1 * len(pattern.pattern) / 20)  # Longer patterns = higher confidence
                    
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = min(confidence, 1.0)
                    break
        
        # Get weights for the detected intent
        if best_intent in INTENT_PATTERNS:
            config = INTENT_PATTERNS[best_intent]
            result = IntentResult(
                intent=best_intent,
                confidence=best_confidence,
                dense_weight=config["dense_weight"],
                sparse_weight=config["sparse_weight"],
                top_k=config["top_k"],
                graph_mode=config.get("graph_mode", "none"),
            )
        else:
            # Default for general queries
            result = IntentResult(
                intent=QueryIntent.GENERAL,
                confidence=0.5,
                dense_weight=0.6,
                sparse_weight=0.4,
                top_k=10,
                graph_mode="none",
            )
        
        logger.debug(f"[INTENT] Query: '{query[:50]}...' -> {result.intent_name} (conf={result.confidence:.2f})")
        return result


# Singleton instance
_intent_classifier: Optional[QueryIntentClassifier] = None


def get_intent_classifier() -> QueryIntentClassifier:
    """Get or create the singleton intent classifier"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = QueryIntentClassifier()
    return _intent_classifier
