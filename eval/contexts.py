"""Короткие саммари разделов и chunk-specific context."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.llm import get_llm

SHORT_SUMMARY_PATH = Path("eval/results/section_summaries_short.json")
CHUNK_CONTEXT_PATH = Path("eval/results/chunk_contexts.json")
MAX_SECTION_CHARS = 2500
MAX_CHUNK_CHARS = 1600


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_chunks(corpus: list) -> list[dict]:
    return [
        {
            "chunk_id": document.metadata["chunk_id"],
            "article": document.metadata["article"],
            "section": document.metadata["section"],
            "text": document.page_content,
        }
        for document in corpus
    ]


def _short_summary_prompt(entry: dict) -> str:
    return f"""Сформулируй очень краткое описание раздела (1 предложение, максимум 20 слов),
которое объясняет, чему он посвящён и чем отличается от других разделов этой же статьи.
Верни только текст без пояснений.

Article: {entry['article']}
Section: {entry['section']}

Section text:
{entry['text'][:MAX_SECTION_CHARS]}
"""


def generate_short_section_summaries(
    sections: dict[str, dict],
    path: Path = SHORT_SUMMARY_PATH,
) -> dict:
    cache = _load(path)
    llm = get_llm()

    for key, entry in sections.items():
        if cache.get(key):
            continue
        last_error = None
        for _ in range(3):
            try:
                response = llm.invoke(_short_summary_prompt(entry))
                text = str(response.content).strip()
                if not text:
                    raise ValueError("пустой ответ")
                cache[key] = text
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5)

        if last_error is not None:
            raise RuntimeError(f"{key}: {last_error}") from last_error

        _save(cache, path)
        print(f"short summary {key}: {cache[key][:80]}...")
        time.sleep(0.15)

    return cache


def _chunk_context_prompt(chunk: dict, section_summary: str) -> str:
    return f"""Ты помогаешь поисковой системе различать фрагменты статьи.
Дана статья, раздел и конкретный чанк. Напиши 1–2 предложения (максимум 35 слов)
о том, что именно содержит этот чанк и как он соотносится с разделом.
Добавь контекст: где находится чанк, какие термины, формулы или теоремы он вводит,
чем отличается от соседних частей. Не пересказывай весь чанк и не добавляй факты,
которых нет в тексте. Верни только текст.

Article: {chunk['article']}
Section: {chunk['section']}
Section summary: {section_summary}

Chunk:
{chunk['text'][:MAX_CHUNK_CHARS]}
"""


def generate_chunk_contexts(
    corpus: list,
    short_summaries: dict,
    path: Path = CHUNK_CONTEXT_PATH,
) -> dict:
    cache = _load(path)
    llm = get_llm()
    chunks = collect_chunks(corpus)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if cache.get(chunk_id):
            continue
        key = f"{chunk['article']}::{chunk['section']}"
        summary = short_summaries.get(key, "")
        last_error = None
        for _ in range(3):
            try:
                response = llm.invoke(_chunk_context_prompt(chunk, summary))
                text = str(response.content).strip()
                if not text:
                    raise ValueError("пустой ответ")
                cache[chunk_id] = text
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5)

        if last_error is not None:
            raise RuntimeError(f"{chunk_id}: {last_error}") from last_error

        _save(cache, path)
        print(f"context {chunk_id}: {cache[chunk_id][:80]}...")
        time.sleep(0.15)

    return cache


def fit_context(
    original: str,
    context: str,
    tokenizer,
    max_tokens: int = 512,
    margin: int = 2,
) -> tuple[str, bool]:
    """Обрезать контекст так, чтобы original + context <= max_tokens.

    Исходный чанк не трогаем. Возвращает (fitted_context, was_truncated).
    """
    original_tokens = len(tokenizer.encode(original, add_special_tokens=False))
    budget = max_tokens - margin - original_tokens
    if budget <= 0 or not context.strip():
        return "", bool(context.strip())

    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) <= budget:
        return context.strip(), False

    fitted = tokenizer.decode(ids[:budget], skip_special_tokens=True).strip()
    return fitted, True
