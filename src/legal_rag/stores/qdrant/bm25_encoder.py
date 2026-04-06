"""BM25 sparse encoder for Qdrant hybrid retrieval."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)


class BM25Encoder:
    """Simple BM25-style sparse encoder for Vietnamese legal text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.doc_freq = {}
        self.avg_doc_len = 0
        self.doc_count = 0
        self.total_doc_len = 0

        try:
            from pyvi import ViTokenizer

            self.tokenizer = ViTokenizer.tokenize
            logger.info("[BM25] Using pyvi Vietnamese tokenizer")
        except ImportError:
            self.tokenizer = lambda x: x
            logger.warning("[BM25] pyvi not found, using simple tokenizer")

    def tokenize(self, text: str) -> List[str]:
        text = self.tokenizer(text.lower())
        tokens = []
        for word in text.split():
            tokens.append(word.replace("_", " "))
        return tokens

    def _update_stats(self, documents: List[str]):
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_count += 1
            self.total_doc_len += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def _recompute_idf(self):
        self.avg_doc_len = self.total_doc_len / self.doc_count if self.doc_count > 0 else 1
        self.idf = {}
        for token, df in self.doc_freq.items():
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def fit(self, documents: List[str]):
        self.vocab = {}
        self.idf = {}
        self.doc_freq = {}
        self.avg_doc_len = 0
        self.doc_count = 0
        self.total_doc_len = 0
        self.partial_fit(documents)

    def partial_fit(self, documents: List[str]):
        if not documents:
            return
        self._update_stats(documents)
        self._recompute_idf()
        logger.info("[BM25] Vocabulary size: %s | docs=%s", len(self.vocab), self.doc_count)

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)

        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        indices = []
        values = []

        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score = idf * (numerator / denominator)
                indices.append(idx)
                values.append(float(score))

        return indices, values

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "doc_freq": self.doc_freq,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count,
            "total_doc_len": self.total_doc_len,
            "k1": self.k1,
            "b": self.b,
        }
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("[BM25] Saved encoder to %s", path)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.doc_freq = data.get("doc_freq", {})
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data["doc_count"]
        self.total_doc_len = data.get("total_doc_len", int(self.avg_doc_len * self.doc_count))
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        logger.info("[BM25] Loaded encoder with %s vocab", len(self.vocab))

