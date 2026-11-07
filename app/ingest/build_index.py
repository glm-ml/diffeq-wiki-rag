"""CLI: построить коллекцию Qdrant из `data/corpus.jsonl`."""

from __future__ import annotations

import argparse
import sys

from app.ingest.chunking import load_corpus
from app.rag.store import build_vectorstore, create_qdrant_client, ensure_collection
from app.settings import get_settings

SANITY_QUERIES = (
    "В чём смысл условия Липшица?",
    "Как устроено линейное дифференциальное уравнение n-го порядка?",
    "Что такое вронскиан и зачем он нужен?",
)


def build_index(corpus_path: str | None = None, recreate: bool = True) -> int:
    settings = get_settings()
    documents = load_corpus(corpus_path or settings.corpus_path)
    client = create_qdrant_client()
    try:
        ensure_collection(client, recreate=recreate)
        store = build_vectorstore(client)
        store.add_documents(documents, batch_size=64)
        print(f"Проиндексировано {len(documents)} чанков в {settings.collection_name}")

        for query in SANITY_QUERIES:
            print(f"\n--- {query}")
            for hit in store.similarity_search(query, k=3):
                meta = hit.metadata
                print(
                    f"  • {meta.get('article')} / {meta.get('section')}"
                    f" — {hit.page_content[:90].strip()!r}"
                )
    finally:
        client.close()

    return len(documents)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Индексация корпуса в Qdrant")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args()
    build_index(args.corpus, recreate=not args.no_recreate)


if __name__ == "__main__":
    main()
