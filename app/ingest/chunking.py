"""Нарезка разделов на чанки и сохранение корпуса в JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.wikipedia import WikiSection

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_sections(
    sections: list[WikiSection],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for section in sections:
        for index, piece in enumerate(splitter.split_text(section.text)):
            parts = [section.article, section.section]
            if section.subsection:
                parts.append(section.subsection)
            parts.append(str(index))
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": section.url,
                        "article": section.article,
                        "section": section.section,
                        "subsection": section.subsection,
                        "chunk": index,
                        "chunk_id": "::".join(parts),
                    },
                )
            )
    return documents


def save_corpus(documents: list[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {"content": document.page_content, "metadata": document.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_corpus(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                Document(page_content=row["content"], metadata=row["metadata"])
            )
    return documents
