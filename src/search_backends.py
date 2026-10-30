from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import Counter
from typing import Iterable

from dotenv import load_dotenv


load_dotenv()
_BGE_MODEL = None
_BGE_MODEL_LOCK = threading.Lock()


STOP_WORDS = {
    "什么", "怎么", "如何", "多少", "多久", "一下", "请问", "帮我", "查询", "规定",
    "员工", "需要", "可以", "吗", "呢", "的", "是", "那", "我", "想", "知道",
}

SYNONYMS = {
    "报销": {"报销", "费用报销", "报账"},
    "期限": {"期限", "时限", "多久", "多长时间", "几天", "提交时间", "最晚"},
    "餐补": {"餐补", "伙食补贴", "吃饭补贴", "餐费补贴"},
    "密码": {"密码", "口令", "登录密码"},
    "忘记": {"忘记", "忘了", "找回", "重置"},
    "研发": {"研发", "技术", "开发"},
    "开机": {"开机", "启动", "开机启动"},
    "电源键": {"电源键", "开关键", "电源按钮"},
    "分辨率": {"分辨率", "清晰度"},
    "配网": {"配网", "联网", "网络配置"},
    "保修": {"保修", "保修期", "质保", "质保期"},
}


def normalize_terms(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    terms: set[str] = set()
    for token in cleaned.split():
        if token not in STOP_WORDS:
            terms.add(token)
            for canonical, variants in SYNONYMS.items():
                if token in variants:
                    terms.add(canonical)
                    terms.update(variants)
    for canonical, variants in SYNONYMS.items():
        if any(variant in cleaned for variant in variants):
            terms.add(canonical)
            terms.update(variants)
    return terms


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize Chinese/Latin text without requiring an external segmenter."""
    lowered = text.lower()
    tokens: list[str] = re.findall(r"[a-z0-9][a-z0-9._+-]*", lowered)
    for span in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(span) <= 4 and span not in STOP_WORDS:
            tokens.append(span)
        for size in (2, 3, 4):
            tokens.extend(
                gram for gram in (span[index:index + size] for index in range(len(span) - size + 1))
                if gram not in STOP_WORDS
            )
    tokens.extend(sorted(normalize_terms(text)))
    return tokens


class BM25Index:
    """Small in-process Okapi BM25 index used for scoped sparse retrieval."""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [tokenize_for_bm25(text) for text in documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequency.update(set(tokens))
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = tokenize_for_bm25(query)
        if not query_terms or not self.documents:
            return [0.0] * len(self.documents)
        scores: list[float] = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / max(self.average_length, 1.0)
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + normalization)
                )
            scores.append(score)
        return scores


class BGEEmbeddingProvider:
    """BGE-M3 embeddings with a deterministic n-gram embedding fallback."""

    def __init__(self, dimensions: int = 384):
        global _BGE_MODEL
        self.dimensions = dimensions
        self.backend = "hashing-ngram-embedding-fallback"
        self.model = None
        if os.getenv("ENABLE_BGE", "0") == "1":
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore

                with _BGE_MODEL_LOCK:
                    if _BGE_MODEL is None:
                        _BGE_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
                self.model = _BGE_MODEL
                self.backend = "bge-m3"
            except Exception as exc:
                raise RuntimeError(
                    "ENABLE_BGE=1，但 BGE-M3 加载失败；请检查 requirements-ml.txt、模型缓存和网络"
                ) from exc

    def dense(self, text: str) -> list[float]:
        return self.dense_many([text])[0]

    def dense_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model:
            output = self.model.encode(texts, batch_size=min(16, len(texts)), max_length=1024)
            return [self._normalize(list(vector)) for vector in output["dense_vecs"]]
        return [self._hashing_embedding(text) for text in texts]

    def _hashing_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        features = tokenize_for_bm25(text)
        for feature, frequency in Counter(features).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0 + math.log(frequency)
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return float(sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm))
