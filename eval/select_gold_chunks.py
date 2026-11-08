"""Подбор эталонных чанков для golden dataset.

Для каждого вопроса берём кандидатов из ожидаемых статей/разделов и
ранжируем их по семантической близости к reference_answer + key_facts.
Скрипт только предлагает выбор; итоговый файл `gold_chunks.jsonl` нужно
просмотреть глазами.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os

os.chdir(ROOT)

from app.ingest.chunking import load_corpus

GOLDEN_PATH = Path("eval/golden/qa.jsonl")
CORPUS_PATH = Path("data/corpus.jsonl")
OUTPUT_PATH = Path("eval/golden/gold_chunks.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidates_for(item: dict, corpus: list) -> list:
    articles = set(item["expected_articles"])
    sections = set(item["expected_sections"])
    return [
        document
        for document in corpus
        if document.metadata["article"] in articles
        and document.metadata["section"] in sections
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--max-gold", type=int, default=2)
    parser.add_argument("--max-gold-multi", type=int, default=3)
    args = parser.parse_args()

    items = load_jsonl(GOLDEN_PATH)
    corpus = load_corpus(CORPUS_PATH)
    model = SentenceTransformer(
        MODEL_NAME,
        prompts={"query": "query: ", "passage": "passage: "},
        default_prompt_name="passage",
    )

    proposals = []
    for item in items:
        candidates = candidates_for(item, corpus)
        if not candidates:
            print(f"{item['id']}: НЕТ КАНДИДАТОВ")
            continue

        query = item["reference_answer"] + " " + " ".join(item["key_facts"])
        query_vector = model.encode(
            [query], prompt_name="query", normalize_embeddings=True
        )[0]
        chunk_vectors = model.encode(
            [document.page_content for document in candidates],
            prompt_name="passage",
            normalize_embeddings=True,
        )
        scores = chunk_vectors @ query_vector
        order = np.argsort(scores)[::-1]
        limit = args.max_gold_multi if item["type"] == "multi_hop" else args.max_gold
        selected = [candidates[index] for index in order[:limit]]

        proposal = {
            "id": item["id"],
            "gold_chunk_ids": [document.metadata["chunk_id"] for document in selected],
            "gold_evidence": [document.page_content[:300] for document in selected],
            "scores": [round(float(scores[index]), 4) for index in order[:limit]],
            "candidates": len(candidates),
        }
        proposals.append(proposal)

        print(f"\n{item['id']} | кандидатов: {len(candidates)}")
        print(f"  вопрос: {item['question']}")
        for document, score in zip(selected, proposal["scores"]):
            print(f"  - {document.metadata['chunk_id']} | score={score}")
            print(f"    {document.page_content[:160].replace(chr(10), ' ')}")

    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
            for proposal in proposals:
                handle.write(json.dumps(proposal, ensure_ascii=False) + "\n")
        print(f"\nзаписано: {OUTPUT_PATH} ({len(proposals)} вопросов)")

    total = sum(len(proposal["gold_chunk_ids"]) for proposal in proposals)
    print(f"\nвсего gold-чанков: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
