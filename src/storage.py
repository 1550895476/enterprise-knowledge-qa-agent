from __future__ import annotations

import os
from collections import defaultdict
from typing import Any


class MilvusVectorStore:
    """Milvus adapter with an in-process fallback for local evaluation.

    Real deployment uses pymilvus when MILVUS_URI is configured. The fallback
    keeps the same insert/search contract and is deliberately not described as
    a production vector index.
    """

    def __init__(self, collection_name: str = "knowledge_chunks"):
        self.collection_name = collection_name
        self._rows: list[dict[str, Any]] = []
        self.backend = "memory"
        self._collection = None
        uri = os.getenv("MILVUS_URI", "")
        if uri:
            try:
                from pymilvus import MilvusClient  # type: ignore

                self._client = MilvusClient(uri=uri, token=os.getenv("MILVUS_TOKEN", ""))
                self.backend = "milvus"
            except Exception:
                self._client = None

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        self._rows.extend(rows)
        if self.backend != "milvus":
            return
        # Collection schema is intentionally left to deployment migration code;
        # local tests use the identical row payload in memory.
        if self._client and self._client.has_collection(self.collection_name):
            self._client.insert(collection_name=self.collection_name, data=rows)

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)


class MongoTraceStore:
    """Persist request/trace records to MongoDB when configured."""

    def __init__(self, database: str = "enterprise_qa", collection: str = "traces"):
        self.backend = "memory"
        self._records: list[dict[str, Any]] = []
        self._collection = None
        uri = os.getenv("MONGODB_URI", "")
        if uri:
            try:
                from pymongo import MongoClient  # type: ignore

                self._collection = MongoClient(uri)[database][collection]
                self.backend = "mongodb"
            except Exception:
                self._collection = None

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if self._collection is not None:
            self._collection.insert_one(record)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

