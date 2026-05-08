from sentence_transformers import SentenceTransformer
from functools import lru_cache
import os
import logging
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "huyydangg/DEk21_hcmute_embedding")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Vietnamese Legal Embedding requires word segmentation
USE_VIETNAMESE_TOKENIZER = os.getenv("USE_VIETNAMESE_TOKENIZER", "true").lower() == "true"  # Default true for DEk21

# Device selection: Use GPU for faster embedding
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # cuda or cpu
DEVICE = EMBEDDING_DEVICE if EMBEDDING_DEVICE in ["cpu", "cuda"] else ("cuda" if torch.cuda.is_available() else "cpu")

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

@lru_cache(maxsize=1)
def get_model():
    logger.info(f"[EMBEDDINGS] Loading SentenceTransformer model: {DEFAULT_MODEL}")
    logger.info(f"[EMBEDDINGS] Device: {DEVICE}")
    logger.info(f"[EMBEDDINGS] Vietnamese tokenizer: {USE_VIETNAMESE_TOKENIZER}")
    
    # Load model on GPU if available
    model = SentenceTransformer(DEFAULT_MODEL, device=DEVICE)
    
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

def embed_texts(texts, batch_size=None):
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
    model = get_model()
    
    # Use batch size from parameter or env
    if batch_size is None:
        batch_size = EMBEDDING_BATCH_SIZE
    
    # Convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    
    # Preprocess Vietnamese texts if enabled
    if USE_VIETNAMESE_TOKENIZER:
        texts = preprocess_vietnamese(texts)
        logger.info(f"[EMBEDDINGS] Preprocessed {len(texts)} texts with Vietnamese tokenizer")
    
    # Log for performance monitoring
    logger.info(f"[EMBEDDINGS] Embedding {len(texts)} texts with batch_size={batch_size}")
    
    # Encode with GPU + batching + normalization
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        device=DEVICE,
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True
    )
    
    logger.info(f"[EMBEDDINGS] Completed embedding {len(texts)} texts")
    return embeddings.tolist()

class ChromaEmbeddingFunction:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_MODEL

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return embed_texts(input)

    def name(self):
        return self.model_name

