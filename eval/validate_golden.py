"""Валидация golden dataset: схема, уникальность, соответствие корпусу."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.ingest.chunking import load_corpus

GOLDEN_PATH = Path("eval/golden/qa.jsonl")
GOLD_CHUNKS_PATH = Path("eval/golden/gold_chunks.jsonl")
CORPUS_PATH = Path("data/corpus.jsonl")

REQUIRED = {
    "id",
    "topic",
    "type",
    "question",
    "expected_articles",
    "expected_sections",
    "reference_answer",
    "reference_source",
    "key_facts",
    "must_cite",
    "difficulty",
}


def main() -> int:
    rows = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus = load_corpus(CORPUS_PATH)
    corpus_articles = {document.metadata["article"] for document in corpus}
    corpus_sections = {
        (document.metadata["article"], document.metadata["section"])
        for document in corpus
    }

    errors: list[str] = []
    ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{index}: нет полей {sorted(missing)}")
            continue

        if row["id"] in ids:
            errors.append(f"{row['id']}: повторяющийся id")
        ids.add(row["id"])

        for article in row["expected_articles"]:
            if article not in corpus_articles:
                errors.append(f"{row['id']}: статьи {article!r} нет в корпусе")

        for section in row["expected_sections"]:
            if not any(
                article in row["expected_articles"] and (article, section) in corpus_sections
                for article in row["expected_articles"]
            ):
                errors.append(f"{row['id']}: раздела {section!r} нет у ожидаемых статей")

        for citation in row["must_cite"]:
            if citation not in row["expected_articles"]:
                errors.append(
                    f"{row['id']}: must_cite {citation!r} отсутствует в expected_articles"
                )

        if row["difficulty"] not in {"easy", "medium", "hard"}:
            errors.append(f"{row['id']}: неизвестная сложность {row['difficulty']!r}")

    gold_rows = [
        json.loads(line)
        for line in GOLD_CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus_chunk_ids = {
        document.metadata.get("chunk_id") for document in corpus
    }
    gold_by_id: dict[str, dict] = {}
    for row in gold_rows:
        gold_by_id[row["id"]] = row
        if not 1 <= len(row["gold_chunk_ids"]) <= 3:
            errors.append(f"{row['id']}: gold-чанков должно быть 1–3")
        if len(row["gold_chunk_ids"]) != len(row["gold_evidence"]):
            errors.append(f"{row['id']}: число id и evidence не совпадает")
        for chunk_id in row["gold_chunk_ids"]:
            if chunk_id not in corpus_chunk_ids:
                errors.append(f"{row['id']}: чанка {chunk_id!r} нет в корпусе")
        if row.get("method") not in {"auto", "manual"}:
            errors.append(f"{row['id']}: неизвестный method {row.get('method')!r}")

    for row in rows:
        if row["id"] not in gold_by_id:
            errors.append(f"{row['id']}: нет записи в gold_chunks.jsonl")

    print(f"gold-вопросов: {len(gold_rows)}")
    print(
        "gold-чанков:",
        sum(len(row["gold_chunk_ids"]) for row in gold_rows),
        "| вручную:",
        sum(1 for row in gold_rows if row.get("method") == "manual"),
    )

    print(f"вопросов: {len(rows)}")
    print("по темам:", dict(Counter(row["topic"] for row in rows)))
    print("по типам:", dict(Counter(row["type"] for row in rows)))
    print("по сложности:", dict(Counter(row["difficulty"] for row in rows)))

    if errors:
        print("\nОШИБКИ:")
        for error in errors:
            print(" -", error)
        return 1

    print("\nOK: golden dataset валиден")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
