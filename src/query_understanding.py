from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import normalize_terms


@dataclass(frozen=True)
class Product:
    product_id: str
    canonical_name: str
    aliases: list[str]
    manual_doc_ids: list[str]


@dataclass
class ProductResolution:
    status: str
    rewritten_query: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


class ProductMatcher:
    """Resolve product mentions before retrieval to prevent cross-model leakage."""

    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.products = [Product(**item) for item in raw]
        self._by_id = {product.product_id: product for product in self.products}
        self._embedder = None
        try:
            if os.getenv("ENABLE_BGE", "0") == "1":
                from FlagEmbedding import BGEM3FlagModel  # type: ignore

                self._embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        except Exception:
            # The lexical matcher remains usable without local model weights.
            self._embedder = None

    def resolve(self, question: str, history: list[dict[str, str]] | None = None, selected_product_id: str | None = None) -> ProductResolution:
        history = history or []
        context = " ".join(item.get("content", "") for item in history[-4:])
        text = f"{context} {question}".strip()
        if selected_product_id:
            product = self._by_id.get(selected_product_id)
            if not product:
                return ProductResolution("unresolved", text, [], "用户选择的商品不在标准商品库")
            return self._resolved(product, text, "采用用户明确选择的商品")

        candidates: list[tuple[Product, float, list[str]]] = []
        normalized = text.lower()
        for product in self.products:
            matched = [alias for alias in product.aliases if alias.lower() in normalized]
            if matched:
                score = max(len(alias) for alias in matched) / max(len(product.canonical_name), 1)
                candidates.append((product, min(1.0, score), matched))

        if self._embedder:
            try:
                query_vector = self._embedder.encode([text], batch_size=1, max_length=256)["dense_vecs"][0]
                product_texts = [f"{p.canonical_name} {' '.join(p.aliases)}" for p in self.products]
                product_vectors = self._embedder.encode(product_texts, batch_size=4, max_length=256)["dense_vecs"]
                for product, vector in zip(self.products, product_vectors):
                    score = self._cosine(query_vector, vector)
                    if score >= 0.35 and product not in {item[0] for item in candidates}:
                        candidates.append((product, float(score), ["BGE-M3 semantic match"]))
            except Exception:
                # Retrieval remains available if model inference is unavailable.
                pass

        # A short family name such as “Pro” intentionally produces an
        # ambiguity instead of guessing X1/X2.
        family_match = bool(re.search(r"\bpro\b|专业版|终端", normalized))
        if not candidates and family_match:
            candidates = [(product, 0.45, ["Pro/终端系列"]) for product in self.products if product.product_id.startswith("P100")]
        if not candidates:
            # No product language: this may be a company policy question and
            # is safe to pass to the general knowledge route.
            has_product_signal = bool(re.search(r"型号|机型|设备|产品|说明书|manual", normalized))
            if has_product_signal:
                return ProductResolution("unresolved", text, [], "出现商品/型号语义，但无法与标准商品库确认")
            return ProductResolution("not_applicable", text, [], "问题未要求按商品型号限定")

        candidates.sort(key=lambda item: item[1], reverse=True)
        candidate_dicts = [
            {"product_id": product.product_id, "canonical_name": product.canonical_name, "score": round(score, 3), "matched_aliases": aliases}
            for product, score, aliases in candidates
        ]
        if len(candidates) > 1 and candidates[0][1] - candidates[1][1] < 0.25:
            return ProductResolution("ambiguous", text, candidate_dicts, "多个相似型号得分接近，需要用户选择")
        return self._resolved(candidates[0][0], text, "已通过标准商品库确认型号")

    def _resolved(self, product: Product, text: str, reason: str) -> ProductResolution:
        return ProductResolution(
            "resolved",
            f"{text} 商品={product.canonical_name} product_id={product.product_id}",
            [{"product_id": product.product_id, "canonical_name": product.canonical_name, "score": 1.0, "matched_aliases": product.aliases}],
            reason,
        )

    def _cosine(self, left: Any, right: Any) -> float:
        denominator = sum(value * value for value in left) ** 0.5 * sum(value * value for value in right) ** 0.5
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

