"""Идемпотентная индексация: заполнить Qdrant, только если коллекция пуста."""

from __future__ import annotations

import time

from qdrant_client import QdrantClient

from app.ingest.chunking import load_corpus
from app.rag.store import build_vectorstore, create_qdrant_client, ensure_collection
from app.settings import get_settings


def open_client(attempts: int = 30, delay: float = 2.0) -> QdrantClient:
    """Дождаться Qdrant: в compose он стартует параллельно с приложением."""
    last_error: Exception | None = None
    for _ in range(attempts):
        client = create_qdrant_client()
        try:
            client.get_collections()
            return client
        except Exception as exc:  # сервер ещё не поднялся
            last_error = exc
            client.close()
            time.sleep(delay)

    raise RuntimeError(
        f"Qdrant недоступен за {attempts * delay:.0f} секунд: {last_error}"
    )


def ensure_index(corpus_path: str | None = None) -> int:
    settings = get_settings()
    client = open_client()
    try:
        ensure_collection(client, recreate=False)
        count = client.count(collection_name=settings.collection_name).count
        if count:
            print(
                f"Коллекция {settings.collection_name} уже содержит "
                f"{count} чанков — индексация пропущена"
            )
            return count

        documents = load_corpus(corpus_path or settings.corpus_path)
        store = build_vectorstore(client)
        store.add_documents(documents, batch_size=64)
        print(
            f"Проиндексировано {len(documents)} чанков "
            f"в {settings.collection_name}"
        )
        return len(documents)
    finally:
        client.close()


def main() -> None:
    ensure_index()


if __name__ == "__main__":
    main()
