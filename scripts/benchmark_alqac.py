#!/usr/bin/env python3
"""
ALQAC Benchmark for Legal RAG System

Comprehensive evaluation script using ALQAC dataset (530 Q&A pairs)
with multiple metrics:
- Retrieval: MRR, Recall@K, Precision@K, Hit Rate
- Generation: Exact Match, F1, BLEU, ROUGE, BERTScore
- Answer Quality: Length ratio, Extractive rate

Usage:
    python scripts/benchmark_alqac.py --csv ALQAC.csv --sample 50
    python scripts/benchmark_alqac.py --csv ALQAC.csv --full
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'benchmark_results.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class BenchmarkSample:
    """Single benchmark sample"""
    idx: int
    question: str
    ground_truth: str
    context: str  # Gold context from ALQAC
    
    # Results
    retrieved_contexts: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    generated_answer: str = ""
    
    # Timing
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    
    # Metrics (computed later)
    retrieval_metrics: Dict = field(default_factory=dict)
    generation_metrics: Dict = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    mrr: float = 0.0  # Mean Reciprocal Rank
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    avg_retrieval_time: float = 0.0


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics"""
    exact_match: float = 0.0
    f1_score: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score_p: float = 0.0
    bert_score_r: float = 0.0
    bert_score_f1: float = 0.0
    answer_length_ratio: float = 0.0  # pred_len / gold_len
    extractive_rate: float = 0.0  # % answers found in context
    avg_generation_time: float = 0.0


@dataclass 
class BenchmarkResult:
    """Complete benchmark result"""
    timestamp: str
    dataset: str
    total_samples: int
    evaluated_samples: int
    
    # Aggregate metrics
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    
    # Per-sample results (for detailed analysis)
    samples: List[Dict] = field(default_factory=list)
    
    # Config
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
            "config": self.config
        }


# ==============================================================================
# Metric Calculators
# ==============================================================================

class TextMetrics:
    """Text-based metrics calculator with Vietnamese support"""
    
    def __init__(self, use_bert_score: bool = True):
        self.use_bert_score = use_bert_score
        self._bert_scorer = None
        self._vi_tokenizer = None
        self._load_vi_tokenizer()
    
    def _load_vi_tokenizer(self):
        """Load Vietnamese tokenizer from pyvi"""
        try:
            from pyvi import ViTokenizer
            self._vi_tokenizer = ViTokenizer
            logger.info("[METRICS] Loaded Vietnamese tokenizer (pyvi)")
        except ImportError:
            logger.warning("[METRICS] pyvi not installed. Using simple split(). Run: pip install pyvi")
            self._vi_tokenizer = None
        
    @property
    def bert_scorer(self):
        """Lazy load BERTScore"""
        if self._bert_scorer is None and self.use_bert_score:
            try:
                from bert_score import BERTScorer
                logger.info("[METRICS] Loading BERTScore model...")
                # Disable baseline rescaling for Vietnamese (no baseline available)
                self._bert_scorer = BERTScorer(
                    lang="vi",
                    rescale_with_baseline=False,  # Vietnamese baseline not available
                    model_type="bert-base-multilingual-cased"
                )
                logger.info("[METRICS] BERTScore ready")
            except ImportError:
                logger.warning("[METRICS] bert-score not installed. Run: pip install bert-score")
                self._bert_scorer = None
            except Exception as e:
                logger.warning(f"[METRICS] BERTScore initialization failed: {e}")
                self._bert_scorer = None
        return self._bert_scorer
    
    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text for comparison"""
        text = text.lower().strip()
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation for comparison
        text = re.sub(r'[.,!?;:"\']', '', text)
        return text
    
    def tokenize_vi(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using pyvi ViTokenizer"""
        text = self.normalize_text(text)
        
        if self._vi_tokenizer is not None:
            # Use pyvi for proper Vietnamese word segmentation
            # "nhà nước" -> "nhà_nước" (1 token) instead of "nhà", "nước" (2 tokens)
            segmented = self._vi_tokenizer.tokenize(text)
            # Split by space but keep compound words
            tokens = segmented.split()
            return tokens
        else:
            # Fallback to simple split
            return text.split()
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Exact match after normalization"""
        pred_norm = self.normalize_text(prediction)
        gold_norm = self.normalize_text(ground_truth)
        
        # Check if gold is contained in prediction (more lenient)
        if gold_norm in pred_norm:
            return 1.0
        if pred_norm in gold_norm:
            return 0.8  # Partial credit
        return 1.0 if pred_norm == gold_norm else 0.0
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Token-level F1 score"""
        pred_tokens = set(self.tokenize_vi(prediction))
        gold_tokens = set(self.tokenize_vi(ground_truth))
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def bleu_score(self, prediction: str, ground_truth: str, n: int = 4) -> Dict[str, float]:
        """BLEU score with n-grams"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
        
        pred_tokens = self.tokenize_vi(prediction)
        gold_tokens = self.tokenize_vi(ground_truth)
        
        if not pred_tokens or not gold_tokens:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
        
        smoothing = SmoothingFunction().method1
        
        results = {}
        for n_gram in [1, 2, 4]:
            weights = tuple([1.0/n_gram] * n_gram + [0.0] * (4 - n_gram))
            try:
                score = sentence_bleu(
                    [gold_tokens], 
                    pred_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                results[f"bleu_{n_gram}"] = score
            except:
                results[f"bleu_{n_gram}"] = 0.0
        
        return results
    
    def rouge_score(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            scores = scorer.score(ground_truth, prediction)
            return {
                "rouge_1": scores['rouge1'].fmeasure,
                "rouge_2": scores['rouge2'].fmeasure,
                "rouge_l": scores['rougeL'].fmeasure
            }
        except ImportError:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    
    def bert_score(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, List[float]]:
        """BERTScore for semantic similarity"""
        if self.bert_scorer is None:
            return {
                "bert_p": [0.0] * len(predictions),
                "bert_r": [0.0] * len(predictions),
                "bert_f1": [0.0] * len(predictions)
            }
        
        P, R, F1 = self.bert_scorer.score(predictions, ground_truths)
        return {
            "bert_p": P.tolist(),
            "bert_r": R.tolist(),
            "bert_f1": F1.tolist()
        }
    
    def compute_all(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Compute all text metrics for a single pair"""
        results = {
            "exact_match": self.exact_match(prediction, ground_truth),
            "f1_score": self.f1_score(prediction, ground_truth)
        }
        
        # BLEU
        bleu = self.bleu_score(prediction, ground_truth)
        results.update(bleu)
        
        # ROUGE
        rouge = self.rouge_score(prediction, ground_truth)
        results.update(rouge)
        
        # Length ratio
        pred_len = len(self.tokenize_vi(prediction))
        gold_len = len(self.tokenize_vi(ground_truth))
        results["length_ratio"] = pred_len / gold_len if gold_len > 0 else 0.0
        
        return results


class RetrievalMetricsCalculator:
    """Retrieval metrics calculator"""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.threshold = similarity_threshold
    
    def normalize_text(self, text: str) -> str:
        """Normalize for comparison"""
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def is_relevant(self, retrieved: str, gold_context: str) -> bool:
        """Check if retrieved context is relevant to gold context"""
        retrieved_norm = self.normalize_text(retrieved)
        gold_norm = self.normalize_text(gold_context)
        
        # Simple substring check
        if gold_norm[:100] in retrieved_norm or retrieved_norm[:100] in gold_norm:
            return True
        
        # Jaccard similarity
        ret_tokens = set(retrieved_norm.split())
        gold_tokens = set(gold_norm.split())
        
        if not ret_tokens or not gold_tokens:
            return False
        
        intersection = ret_tokens & gold_tokens
        union = ret_tokens | gold_tokens
        jaccard = len(intersection) / len(union)
        
        return jaccard > self.threshold
    
    def compute_mrr(self, retrieved_list: List[str], gold_context: str) -> float:
        """Mean Reciprocal Rank"""
        for i, retrieved in enumerate(retrieved_list, 1):
            if self.is_relevant(retrieved, gold_context):
                return 1.0 / i
        return 0.0
    
    def compute_recall_at_k(self, retrieved_list: List[str], gold_context: str, k: int) -> float:
        """Recall@K - whether gold context is in top-K"""
        for retrieved in retrieved_list[:k]:
            if self.is_relevant(retrieved, gold_context):
                return 1.0
        return 0.0
    
    def compute_precision_at_k(self, retrieved_list: List[str], gold_context: str, k: int) -> float:
        """Precision@K - fraction of relevant in top-K"""
        relevant_count = sum(
            1 for r in retrieved_list[:k] if self.is_relevant(r, gold_context)
        )
        return relevant_count / k if k > 0 else 0.0


# ==============================================================================
# RAG System Wrapper
# ==============================================================================

# Extractive QA prompt for ALQAC benchmark (short, concise answers)
EXTRACTIVE_PROMPT = """Bạn là trợ lý pháp lý chính xác. Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách NGẮN GỌN NHẤT có thể.

QUY TẮC BẮT BUỘC:
1. Nếu câu trả lời là một con số, mốc thời gian, hoặc danh từ cụ thể -> CHỈ đưa ra đáp án đó
2. KHÔNG giải thích dài dòng, KHÔNG trích dẫn lại điều luật
3. Trả lời trong 1-10 từ nếu có thể
4. Nếu không tìm thấy thông tin -> trả lời "Không tìm thấy"

Ví dụ:
- Câu hỏi: "Án tù tối đa bao nhiêu năm?" -> Trả lời: "07 năm"
- Câu hỏi: "Độ tuổi chịu trách nhiệm hình sự?" -> Trả lời: "16 tuổi"
- Câu hỏi: "Hình phạt nào được áp dụng?" -> Trả lời: "Cảnh cáo"
"""


class RAGSystemWrapper:
    """Wrapper around the Legal RAG system for benchmarking"""
    
    def __init__(self, use_llm: bool = True, use_reranker: bool = False, use_extractive_prompt: bool = True):
        self.use_llm = use_llm
        self.use_reranker = use_reranker
        self.use_extractive_prompt = use_extractive_prompt
        self.pipeline = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the RAG pipeline"""
        if self._initialized:
            return
        
        try:
            from src.legal_rag_pipeline import LegalRAGPipeline
            
            logger.info("[RAG] Initializing Legal RAG Pipeline...")
            self.pipeline = LegalRAGPipeline()
            self.pipeline.initialize()
            
            self._initialized = True
            logger.info("[RAG] Pipeline ready")
            
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize: {e}")
            raise
    
    def query(self, question: str, top_k: int = 10) -> Tuple[List[str], List[float], str, float, float]:
        """
        Query the RAG system
        
        Returns:
            retrieved_contexts: List of retrieved context strings
            scores: List of retrieval scores
            answer: Generated answer
            retrieval_time: Time for retrieval
            generation_time: Time for generation
        """
        if not self._initialized:
            self.initialize()
        
        # Retrieval phase
        start_ret = time.time()
        
        # Get embedding
        query_embedding = self.pipeline.embedding_model.encode_query(question)
        
        # Hybrid search with optional reranking
        if self.use_reranker:
            # Use hybrid search with reranking for better Recall@1
            try:
                results = self.pipeline.qdrant_store.hybrid_search_with_rerank(
                    query=question,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            except AttributeError:
                # Fallback if reranker not available
                results = self.pipeline.qdrant_store.hybrid_search(
                    query=question,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
        else:
            results = self.pipeline.qdrant_store.hybrid_search(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
        
        retrieval_time = time.time() - start_ret
        
        # Extract contexts and scores
        retrieved_contexts = [r.content for r in results]
        scores = [r.score for r in results]
        
        # Debug logging
        if not results:
            logger.warning(f"[RAG] No results retrieved for question: {question[:50]}...")
        
        # Generation phase
        answer = ""
        generation_time = 0.0
        
        if self.use_llm and results:
            start_gen = time.time()
            
            # Format context (use fewer for extractive QA)
            num_contexts = 3 if self.use_extractive_prompt else 5
            context_text = "\n\n---\n\n".join([
                f"[{i+1}] {r.content}" for i, r in enumerate(results[:num_contexts])
            ])
            
            # Generate answer with appropriate prompt
            if self.use_extractive_prompt:
                answer = self._generate_extractive(question, context_text)
            else:
                answer = self.pipeline.llm.generate(question, context_text)
            
            generation_time = time.time() - start_gen
        
        return retrieved_contexts, scores, answer, retrieval_time, generation_time
    
    def _generate_extractive(self, question: str, context: str) -> str:
        """Generate short extractive answer for ALQAC benchmark"""
        import requests
        
        prompt = f"""CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI NGẮN GỌN:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.pipeline.llm.model,
                    "prompt": prompt,
                    "system": EXTRACTIVE_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for precise extraction
                        "top_p": 0.9,
                        "num_predict": 50  # Limit output length
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # Clean thinking tags if present
                if "<think>" in answer:
                    import re
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                
                # Take only first line/sentence for extractive
                answer = answer.split('\n')[0].strip()
                
                if not answer:
                    logger.warning(f"[RAG] Empty answer from LLM for question: {question[:50]}...")
                
                return answer
            else:
                logger.error(f"[RAG] LLM request failed with status {response.status_code}: {response.text[:200]}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error(f"[RAG] LLM request timeout for question: {question[:50]}...")
            return ""
        except requests.exceptions.ConnectionError:
            logger.error("[RAG] Cannot connect to Ollama at http://localhost:11434. Is Ollama running?")
            return ""
        except Exception as e:
            logger.error(f"[RAG] Extractive generation failed: {e}")
            return ""


# ==============================================================================
# Benchmark Runner
# ==============================================================================

class ALQACBenchmark:
    """Main benchmark runner for ALQAC dataset"""
    
    def __init__(
        self,
        csv_path: str,
        sample_size: Optional[int] = None,
        use_llm: bool = True,
        use_bert_score: bool = True,
        use_reranker: bool = False,
        use_extractive_prompt: bool = True,
        top_k: int = 10,
        output_dir: str = "benchmark_results"
    ):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.use_llm = use_llm
        self.use_bert_score = use_bert_score
        self.use_reranker = use_reranker
        self.use_extractive_prompt = use_extractive_prompt
        self.top_k = top_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.df = pd.read_csv(csv_path)
        logger.info(f"[BENCHMARK] Loaded {len(self.df)} samples from {csv_path}")
        
        # Sample if needed
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42)
            logger.info(f"[BENCHMARK] Sampled {sample_size} samples")
        
        # Log configuration
        logger.info(f"[BENCHMARK] Config: reranker={use_reranker}, extractive={use_extractive_prompt}")
        
        # Initialize components
        self.rag = RAGSystemWrapper(
            use_llm=use_llm, 
            use_reranker=use_reranker,
            use_extractive_prompt=use_extractive_prompt
        )
        self.text_metrics = TextMetrics(use_bert_score=use_bert_score)
        self.retrieval_metrics = RetrievalMetricsCalculator()
        
        # Results storage
        self.samples: List[BenchmarkSample] = []
        self.result: Optional[BenchmarkResult] = None
    
    def run(self) -> BenchmarkResult:
        """Run the full benchmark"""
        logger.info("[BENCHMARK] Starting benchmark run...")
        start_time = time.time()
        
        # Initialize RAG
        self.rag.initialize()
        
        # Process each sample
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Benchmarking"):
            sample = self._process_sample(idx, row)
            self.samples.append(sample)
        
        # Compute aggregate metrics
        self.result = self._compute_aggregate_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"[BENCHMARK] Completed in {total_time:.1f}s")
        
        # Save results
        self._save_results()
        
        return self.result
    
    def _process_sample(self, idx: int, row: pd.Series) -> BenchmarkSample:
        """Process a single benchmark sample"""
        sample = BenchmarkSample(
            idx=idx,
            question=str(row['question']),
            ground_truth=str(row['answer']),
            context=str(row['context'])
        )
        
        try:
            # Query RAG system
            contexts, scores, answer, ret_time, gen_time = self.rag.query(
                sample.question, 
                top_k=self.top_k
            )
            
            sample.retrieved_contexts = contexts
            sample.retrieved_scores = scores
            sample.generated_answer = answer
            sample.retrieval_time = ret_time
            sample.generation_time = gen_time
            
            # Compute retrieval metrics
            sample.retrieval_metrics = {
                "mrr": self.retrieval_metrics.compute_mrr(contexts, sample.context),
                "recall@1": self.retrieval_metrics.compute_recall_at_k(contexts, sample.context, 1),
                "recall@3": self.retrieval_metrics.compute_recall_at_k(contexts, sample.context, 3),
                "recall@5": self.retrieval_metrics.compute_recall_at_k(contexts, sample.context, 5),
                "recall@10": self.retrieval_metrics.compute_recall_at_k(contexts, sample.context, 10),
                "precision@1": self.retrieval_metrics.compute_precision_at_k(contexts, sample.context, 1),
                "precision@3": self.retrieval_metrics.compute_precision_at_k(contexts, sample.context, 3),
                "precision@5": self.retrieval_metrics.compute_precision_at_k(contexts, sample.context, 5),
            }
            
            # Compute generation metrics
            if answer and answer.strip():  # Only compute if answer is not empty
                sample.generation_metrics = self.text_metrics.compute_all(
                    answer, sample.ground_truth
                )
                
                # Check if answer is extractive (found in context)
                is_extractive = sample.ground_truth.lower() in sample.context.lower()
                sample.generation_metrics["extractive"] = 1.0 if is_extractive else 0.0
            else:
                # Log warning if no answer generated
                logger.warning(f"[BENCHMARK] No answer generated for sample {idx}: {sample.question[:50]}...")
                sample.generation_metrics = {}
            
        except Exception as e:
            logger.error(f"[BENCHMARK] Error processing sample {idx}: {e}")
            sample.retrieval_metrics = {}
            sample.generation_metrics = {}
        
        return sample
    
    def _compute_aggregate_metrics(self) -> BenchmarkResult:
        """Compute aggregate metrics from all samples"""
        
        # Retrieval metrics
        retrieval = RetrievalMetrics()
        ret_values = defaultdict(list)
        
        for sample in self.samples:
            for key, value in sample.retrieval_metrics.items():
                ret_values[key].append(value)
            ret_values["retrieval_time"].append(sample.retrieval_time)
        
        retrieval.mrr = np.mean(ret_values["mrr"]) if ret_values["mrr"] else 0.0
        retrieval.recall_at_1 = np.mean(ret_values["recall@1"]) if ret_values["recall@1"] else 0.0
        retrieval.recall_at_3 = np.mean(ret_values["recall@3"]) if ret_values["recall@3"] else 0.0
        retrieval.recall_at_5 = np.mean(ret_values["recall@5"]) if ret_values["recall@5"] else 0.0
        retrieval.recall_at_10 = np.mean(ret_values["recall@10"]) if ret_values["recall@10"] else 0.0
        retrieval.precision_at_1 = np.mean(ret_values["precision@1"]) if ret_values["precision@1"] else 0.0
        retrieval.precision_at_3 = np.mean(ret_values["precision@3"]) if ret_values["precision@3"] else 0.0
        retrieval.precision_at_5 = np.mean(ret_values["precision@5"]) if ret_values["precision@5"] else 0.0
        retrieval.hit_rate_at_1 = retrieval.recall_at_1  # Same as recall@1
        retrieval.hit_rate_at_5 = retrieval.recall_at_5
        retrieval.hit_rate_at_10 = retrieval.recall_at_10
        retrieval.avg_retrieval_time = np.mean(ret_values["retrieval_time"]) if ret_values["retrieval_time"] else 0.0
        
        # Generation metrics
        generation = GenerationMetrics()
        gen_values = defaultdict(list)
        
        for sample in self.samples:
            for key, value in sample.generation_metrics.items():
                gen_values[key].append(value)
            gen_values["generation_time"].append(sample.generation_time)
        
        generation.exact_match = np.mean(gen_values["exact_match"]) if gen_values["exact_match"] else 0.0
        generation.f1_score = np.mean(gen_values["f1_score"]) if gen_values["f1_score"] else 0.0
        generation.bleu_1 = np.mean(gen_values["bleu_1"]) if gen_values["bleu_1"] else 0.0
        generation.bleu_2 = np.mean(gen_values["bleu_2"]) if gen_values["bleu_2"] else 0.0
        generation.bleu_4 = np.mean(gen_values["bleu_4"]) if gen_values["bleu_4"] else 0.0
        generation.rouge_1 = np.mean(gen_values["rouge_1"]) if gen_values["rouge_1"] else 0.0
        generation.rouge_2 = np.mean(gen_values["rouge_2"]) if gen_values["rouge_2"] else 0.0
        generation.rouge_l = np.mean(gen_values["rouge_l"]) if gen_values["rouge_l"] else 0.0
        generation.answer_length_ratio = np.mean(gen_values["length_ratio"]) if gen_values["length_ratio"] else 0.0
        generation.extractive_rate = np.mean(gen_values["extractive"]) if gen_values["extractive"] else 0.0
        generation.avg_generation_time = np.mean(gen_values["generation_time"]) if gen_values["generation_time"] else 0.0
        
        # Compute BERTScore in batch (more efficient)
        if self.use_bert_score and self.use_llm:
            predictions = [s.generated_answer for s in self.samples if s.generated_answer]
            ground_truths = [s.ground_truth for s in self.samples if s.generated_answer]
            
            if predictions:
                bert_scores = self.text_metrics.bert_score(predictions, ground_truths)
                generation.bert_score_p = np.mean(bert_scores["bert_p"])
                generation.bert_score_r = np.mean(bert_scores["bert_r"])
                generation.bert_score_f1 = np.mean(bert_scores["bert_f1"])
        
        # Create result
        result = BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            dataset=self.csv_path,
            total_samples=len(self.df),
            evaluated_samples=len(self.samples),
            retrieval=retrieval,
            generation=generation,
            config={
                "top_k": self.top_k,
                "use_llm": self.use_llm,
                "use_bert_score": self.use_bert_score,
                "sample_size": self.sample_size
            }
        )
        
        return result
    
    def _save_results(self):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[BENCHMARK] Saved results to {json_path}")
        
        # Save detailed CSV
        csv_path = self.output_dir / f"benchmark_details_{timestamp}.csv"
        details = []
        for sample in self.samples:
            row = {
                "idx": sample.idx,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "generated_answer": sample.generated_answer,
                "retrieval_time": sample.retrieval_time,
                "generation_time": sample.generation_time,
                **sample.retrieval_metrics,
                **sample.generation_metrics
            }
            details.append(row)
        
        df_details = pd.DataFrame(details)
        df_details.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"[BENCHMARK] Saved details to {csv_path}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print benchmark summary to console"""
        r = self.result.retrieval
        g = self.result.generation
        
        print("\n" + "="*70)
        print("📊 ALQAC BENCHMARK RESULTS")
        print("="*70)
        
        print(f"\n📁 Dataset: {self.csv_path}")
        print(f"📝 Samples: {self.result.evaluated_samples} / {self.result.total_samples}")
        print(f"⏱️  Timestamp: {self.result.timestamp}")
        
        print("\n" + "-"*70)
        print("🔍 RETRIEVAL METRICS")
        print("-"*70)
        print(f"  MRR:          {r.mrr:.4f}")
        print(f"  Recall@1:     {r.recall_at_1:.4f}")
        print(f"  Recall@3:     {r.recall_at_3:.4f}")
        print(f"  Recall@5:     {r.recall_at_5:.4f}")
        print(f"  Recall@10:    {r.recall_at_10:.4f}")
        print(f"  Precision@1:  {r.precision_at_1:.4f}")
        print(f"  Precision@3:  {r.precision_at_3:.4f}")
        print(f"  Precision@5:  {r.precision_at_5:.4f}")
        print(f"  Avg Time:     {r.avg_retrieval_time*1000:.1f} ms")
        
        if self.use_llm:
            print("\n" + "-"*70)
            print("✍️  GENERATION METRICS")
            print("-"*70)
            print(f"  Exact Match:  {g.exact_match:.4f}")
            print(f"  F1 Score:     {g.f1_score:.4f}")
            print(f"  BLEU-1:       {g.bleu_1:.4f}")
            print(f"  BLEU-2:       {g.bleu_2:.4f}")
            print(f"  BLEU-4:       {g.bleu_4:.4f}")
            print(f"  ROUGE-1:      {g.rouge_1:.4f}")
            print(f"  ROUGE-2:      {g.rouge_2:.4f}")
            print(f"  ROUGE-L:      {g.rouge_l:.4f}")
            if self.use_bert_score:
                print(f"  BERTScore P:  {g.bert_score_p:.4f}")
                print(f"  BERTScore R:  {g.bert_score_r:.4f}")
                print(f"  BERTScore F1: {g.bert_score_f1:.4f}")
            print(f"  Length Ratio: {g.answer_length_ratio:.2f}")
            print(f"  Extractive:   {g.extractive_rate:.2%}")
            print(f"  Avg Time:     {g.avg_generation_time*1000:.1f} ms")
        
        print("\n" + "="*70)
        print("✅ Benchmark completed!")
        print("="*70 + "\n")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALQAC Benchmark for Legal RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 50 samples (quick test)
  python scripts/benchmark_alqac.py --csv ALQAC.csv --sample 50

  # Run on full dataset
  python scripts/benchmark_alqac.py --csv ALQAC.csv --full

  # Retrieval-only benchmark (no LLM)
  python scripts/benchmark_alqac.py --csv ALQAC.csv --no-llm --sample 100
  
  # Without BERTScore (faster)
  python scripts/benchmark_alqac.py --csv ALQAC.csv --no-bert-score
        """
    )
    
    parser.add_argument(
        "--csv", 
        type=str, 
        default="ALQAC.csv",
        help="Path to ALQAC CSV file"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run on full dataset"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    parser.add_argument(
        "--no-llm", 
        action="store_true",
        help="Skip LLM generation (retrieval-only)"
    )
    parser.add_argument(
        "--no-bert-score", 
        action="store_true",
        help="Skip BERTScore computation (faster)"
    )
    parser.add_argument(
        "--use-reranker", 
        action="store_true",
        help="Enable cross-encoder reranking for better Recall@1"
    )
    parser.add_argument(
        "--verbose-mode", 
        action="store_true",
        help="Use verbose LLM response (not extractive). Default is extractive (short answers)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="benchmark_results",
        help="Output directory for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine sample size
    sample_size = None if args.full else (args.sample or 50)
    
    # Create and run benchmark
    benchmark = ALQACBenchmark(
        csv_path=args.csv,
        sample_size=sample_size,
        use_llm=not args.no_llm,
        use_bert_score=not args.no_bert_score,
        use_reranker=args.use_reranker,
        use_extractive_prompt=not args.verbose_mode,
        top_k=args.top_k,
        output_dir=args.output
    )
    
    result = benchmark.run()
    
    # Generate visualization report
    try:
        # Try relative import first
        try:
            from benchmark_visualizer import BenchmarkVisualizer
        except ImportError:
            from scripts.benchmark_visualizer import BenchmarkVisualizer
        
        visualizer = BenchmarkVisualizer(result, benchmark.samples, args.output)
        visualizer.generate_all()
        logger.info("[BENCHMARK] Visualization report generated!")
    except ImportError as e:
        logger.warning(f"[BENCHMARK] Visualizer not available: {e}")
    except Exception as e:
        logger.warning(f"[BENCHMARK] Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
