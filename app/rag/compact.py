"""Сжатие retrieved-контекста перед отправкой в LLM.

Подход extractive: выбираем наиболее релевантные вопросу предложения из
каждого чанка. Формулировки и цитаты сохраняются, новых фактов не появляется.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
DEFAULT_MAX_TOKENS = 500


@dataclass
class CompactionStats:
    before_tokens: int
    after_tokens: int
    documents: int
    selected_sentences: int
    total_sentences: int


class SentenceScorer:
    """Семантическая близость вопроса и предложения через e5."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            prompts={"query": "query: ", "passage": "passage: "},
            default_prompt_name="passage",
        )

    def __call__(self, question: str, sentences: list[str]) -> list[float]:
        query = self.model.encode(
            [question], prompt_name="query", normalize_embeddings=True
        )[0]
        vectors = self.model.encode(
            sentences,
            prompt_name="passage",
            normalize_embeddings=True,
        )
        return (vectors @ query).tolist()


def split_sentences(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in _SENTENCE_SPLIT.split(text)
        if part.strip()
    ]
    return parts or [text.strip()]


def compact_documents(
    documents: list[Document],
    question: str,
    scorer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    encoding_name: str = "cl100k_base",
) -> tuple[list[Document], CompactionStats]:
    """Оставить в каждом чанке самые релевантные предложения.

    Гарантирует хотя бы одно предложение на чанк, чтобы не ломать нумерацию
    источников `[1]..[n]`. Бюджет общий; если лучшие предложения не влезают,
    результат может немного превысить max_tokens.
    """
    encoder = tiktoken.get_encoding(encoding_name)
    before_tokens = sum(
        len(encoder.encode(document.page_content)) for document in documents
    )

    candidates = []
    best_by_document: dict[int, int] = {}
    for document_index, document in enumerate(documents):
        sentences = split_sentences(document.page_content)
        if len(sentences) == 1:
            scores = [1.0]
        else:
            scores = scorer(question, sentences)

        for sentence_index, (sentence, score) in enumerate(zip(sentences, scores)):
            candidates.append(
                {
                    "document": document_index,
                    "sentence": sentence_index,
                    "text": sentence,
                    "score": float(score),
                    "tokens": len(encoder.encode(sentence)),
                }
            )

        best_index = max(range(len(sentences)), key=lambda index: scores[index])
        best_by_document[document_index] = best_index

    selected = []
    selected_keys = set()
    used_tokens = 0

    # Сначала по одному лучшему предложению на каждый чанк.
    for document_index, sentence_index in best_by_document.items():
        candidate = next(
            item
            for item in candidates
            if item["document"] == document_index
            and item["sentence"] == sentence_index
        )
        selected.append(candidate)
        selected_keys.add((document_index, sentence_index))
        used_tokens += candidate["tokens"]

    # Добираем самые релевантные предложения в рамках бюджета.
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        key = (candidate["document"], candidate["sentence"])
        if key in selected_keys:
            continue
        if used_tokens + candidate["tokens"] > max_tokens:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        used_tokens += candidate["tokens"]

    compacted: list[Document] = []
    for document_index, document in enumerate(documents):
        parts = [
            item["text"]
            for item in sorted(
                (item for item in selected if item["document"] == document_index),
                key=lambda item: item["sentence"],
            )
        ]
        compacted.append(
            Document(
                page_content=" ".join(parts),
                metadata={
                    **document.metadata,
                    "original_content": document.page_content,
                },
            )
        )

    after_tokens = sum(
        len(encoder.encode(document.page_content)) for document in compacted
    )
    return compacted, CompactionStats(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        documents=len(documents),
        selected_sentences=len(selected),
        total_sentences=len(candidates),
    )
