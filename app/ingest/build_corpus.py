"""CLI: скачать разделы статей и собрать `data/corpus.jsonl`."""

from __future__ import annotations

import argparse
import sys

import httpx

from app.ingest.chunking import chunk_sections, save_corpus
from app.ingest.wikipedia import (
    USER_AGENT,
    extract_sections,
    fetch_article_html,
    load_articles,
)
from app.settings import get_settings


def build_corpus(articles_path: str, output_path: str, language: str | None = None) -> int:
    language, specs = load_articles(articles_path)
    chunks = []
    with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for spec in specs:
            print(f"Загружаю: {spec.title} ...", flush=True)
            html = fetch_article_html(spec, language=language, client=client)
            sections = extract_sections(html, spec, language=language)
            documents = chunk_sections(sections)
            print(f"  разделов: {len(sections)}, чанков: {len(documents)}", flush=True)
            chunks.extend(documents)

    save_corpus(chunks, output_path)
    print(f"Готово: {len(chunks)} чанков -> {output_path}")
    return len(chunks)


def main() -> None:
    # На Windows консоль по умолчанию cp1251 и падает на "Picard–Lindelöf".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Сборка корпуса из Википедии")
    parser.add_argument("--articles", default="data/articles.yaml")
    parser.add_argument("--out", default=settings.corpus_path)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    build_corpus(args.articles, args.out, args.language)


if __name__ == "__main__":
    main()
