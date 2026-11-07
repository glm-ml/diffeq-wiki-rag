"""Эмбеддинги и векторное хранилище Qdrant.

Поддерживаются два режима:
* `QDRANT_URL` задан — обычный сервер (docker-compose);
* `QDRANT_URL` пуст — встроенный Qdrant, данные лежат в `qdrant_data/`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.settings import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
    )


def create_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    settings = get_settings()
    exists = client.collection_exists(settings.collection_name)
    if exists and recreate:
        client.delete_collection(settings.collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def build_vectorstore(client: QdrantClient | None = None) -> QdrantVectorStore:
    settings = get_settings()
    client = client or create_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )
