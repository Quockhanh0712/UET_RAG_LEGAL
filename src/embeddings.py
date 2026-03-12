from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterable, List

import torch

# Avoid torchvision import path issues in text-only embedding workloads.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AV", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
DEFAULT_NORMALIZE_EMBEDDINGS = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
DEFAULT_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "").strip()
DEFAULT_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "").strip()

# Vietnamese tokenizer behavior:
# - "auto": enable for DEk21 only
# - "true"/"false": force explicit behavior
VI_TOKENIZER_MODE = os.getenv("USE_VIETNAMESE_TOKENIZER", "auto").lower().strip()

# Device selection
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda").lower().strip()  # auto/cuda/cpu
if EMBEDDING_DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = EMBEDDING_DEVICE if EMBEDDING_DEVICE in {"cpu", "cuda"} else ("cuda" if torch.cuda.is_available() else "cpu")

if EMBEDDING_DEVICE == "cuda" and DEVICE == "cpu":
    logger.warning("[EMBEDDINGS] Requested cuda but CUDA is unavailable. Falling back to CPU.")

# Vietnamese tokenizer (lazy load)
_vi_tokenizer = None

def get_vi_tokenizer():
    """Lazy load Vietnamese tokenizer from pyvi"""
    global _vi_tokenizer
    if _vi_tokenizer is None:
        try:
            from pyvi import ViTokenizer
            _vi_tokenizer = ViTokenizer
            logger.info("[EMBEDDINGS] Loaded Vietnamese tokenizer (pyvi)")
        except ImportError:
            logger.warning("[EMBEDDINGS] pyvi not installed. Run: pip install pyvi")
            _vi_tokenizer = None
    return _vi_tokenizer

def preprocess_vietnamese(texts):
    """
    Preprocess Vietnamese texts using word segmentation.
    Required for DEk21_hcmute_embedding model.
    
    Example: "Điều kiện kết hôn" -> "Điều_kiện kết_hôn"
    """
    tokenizer = get_vi_tokenizer()
    if tokenizer is None:
        logger.warning("[EMBEDDINGS] Vietnamese tokenizer not available, using raw texts")
        return texts
    
    if isinstance(texts, str):
        return tokenizer.tokenize(texts)
    
    return [tokenizer.tokenize(text) for text in texts]

def _normalize_model_name(model_name: str | None) -> str:
    return (model_name or DEFAULT_MODEL).strip()


def _should_use_vietnamese_tokenizer(model_name: str) -> bool:
    if VI_TOKENIZER_MODE in {"true", "1", "yes"}:
        return True
    if VI_TOKENIZER_MODE in {"false", "0", "no"}:
        return False
    # Auto mode: keep legacy behavior for DEk21 only.
    return "dek21_hcmute_embedding" in model_name.lower()


def _resolve_prefix(model_name: str, text_type: str) -> str:
    text_type = (text_type or "passage").lower().strip()
    if text_type == "query" and DEFAULT_QUERY_PREFIX:
        return DEFAULT_QUERY_PREFIX if DEFAULT_QUERY_PREFIX.endswith(" ") else f"{DEFAULT_QUERY_PREFIX} "
    if text_type != "query" and DEFAULT_PASSAGE_PREFIX:
        return DEFAULT_PASSAGE_PREFIX if DEFAULT_PASSAGE_PREFIX.endswith(" ") else f"{DEFAULT_PASSAGE_PREFIX} "

    lower_name = model_name.lower()
    # E5-family models require explicit query/passage instruction prefixes.
    if "e5" in lower_name:
        return "query: " if text_type == "query" else "passage: "
    return ""


def _apply_prefix(texts: Iterable[str], prefix: str) -> List[str]:
    if not prefix:
        return list(texts)
    return [f"{prefix}{text}" if text else prefix.strip() for text in texts]


@lru_cache(maxsize=8)
def get_model(model_name: str = DEFAULT_MODEL):
    model_name = _normalize_model_name(model_name)
    logger.info(f"[EMBEDDINGS] Loading SentenceTransformer model: {model_name}")
    logger.info(f"[EMBEDDINGS] Device: {DEVICE}")
    logger.info(f"[EMBEDDINGS] Vietnamese tokenizer mode: {VI_TOKENIZER_MODE}")
    
    # Load model on GPU if available
    model = SentenceTransformer(model_name, device=DEVICE)
    
    # Enable FP16 for faster inference on GPU
    if DEVICE == "cuda" and USE_FP16:
        model.half()
        logger.info("[EMBEDDINGS] Enabled FP16 mixed precision")
    
    # Log GPU info
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[EMBEDDINGS] Using GPU: {gpu_name}")
    
    # Log model info
    logger.info(f"[EMBEDDINGS] Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    logger.info("[EMBEDDINGS] Model loaded OK")
    return model

def get_embedding_dimension(model_name: str | None = None) -> int:
    model = get_model(_normalize_model_name(model_name))
    return int(model.get_sentence_embedding_dimension())


def embed_texts(
    texts,
    batch_size: int | None = None,
    model_name: str | None = None,
    normalize_embeddings: bool | None = None,
    text_type: str = "passage",
    add_prefix: bool = True,
):
    """
    Embed texts with GPU acceleration and batch processing.
    
    For Vietnamese legal embedding (DEk21_hcmute_embedding), 
    texts will be preprocessed with word segmentation using pyvi.
    
    Args:
        texts: Single text or list of texts to embed
        batch_size: Batch size for processing (default from env)
    
    Returns:
        List of embedding vectors
    """
    model_name = _normalize_model_name(model_name)
    model = get_model(model_name)
    
    # Use batch size from parameter or env
    if batch_size is None:
        batch_size = EMBEDDING_BATCH_SIZE

    # Convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    use_vietnamese_tokenizer = _should_use_vietnamese_tokenizer(model_name)

    # Preprocess Vietnamese texts if enabled
    if use_vietnamese_tokenizer:
        texts = preprocess_vietnamese(texts)
        logger.info(f"[EMBEDDINGS] Preprocessed {len(texts)} texts with Vietnamese tokenizer")

    if add_prefix:
        prefix = _resolve_prefix(model_name, text_type=text_type)
        texts = _apply_prefix(texts, prefix)
        if prefix:
            logger.info("[EMBEDDINGS] Applied '%s' prefix for text_type=%s", prefix.strip(), text_type)

    if normalize_embeddings is None:
        normalize_embeddings = DEFAULT_NORMALIZE_EMBEDDINGS
    
    # Log for performance monitoring
    logger.info(
        "[EMBEDDINGS] Embedding %s texts with batch_size=%s normalize=%s model=%s",
        len(texts),
        batch_size,
        normalize_embeddings,
        model_name,
    )
    
    # Encode with GPU + batching + normalization
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=bool(normalize_embeddings),
        device=DEVICE,
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True
    )
    
    logger.info(f"[EMBEDDINGS] Completed embedding {len(texts)} texts")
    return embeddings.tolist()


def embed_query(
    query: str,
    model_name: str | None = None,
    normalize_embeddings: bool | None = None,
):
    return embed_texts(
        [query],
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        text_type="query",
        add_prefix=True,
    )[0]

class ChromaEmbeddingFunction:
    def __init__(self, model_name: str | None = None):
        self.model_name = _normalize_model_name(model_name)

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input, model_name=self.model_name, text_type="passage")

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input, model_name=self.model_name, text_type="query")

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input, model_name=self.model_name, text_type="passage")

    def name(self):
        return self.model_name

