"""Метрики retrieval и сохранение результатов экспериментов."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GOLDEN_PATH = Path("eval/golden/qa.jsonl")
GOLD_CHUNKS_PATH = Path("eval/golden/gold_chunks.jsonl")


def load_gold_chunk_map(path: str | Path = GOLD_CHUNKS_PATH) -> dict[str, list[str]]:
    path = Path(path)
    if not path.exists():
        return {}
    return {
        row["id"]: list(row["gold_chunk_ids"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_golden(path: str | Path = GOLDEN_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunk_key(document) -> str:
    """Стабильный ключ чанка внутри текущего корпуса."""
    meta = document.metadata
    if meta.get("chunk_id"):
        return str(meta["chunk_id"])
    return (
        f"{meta.get('article')}::{meta.get('section')}::"
        f"{meta.get('subsection', '')}::{meta.get('chunk')}"
    )


def gold_keys_for_item(item: dict, corpus: list) -> list[str]:
    articles = set(item["expected_articles"])
    sections = set(item["expected_sections"])
    return [
        chunk_key(document)
        for document in corpus
        if document.metadata.get("article") in articles
        and document.metadata.get("section") in sections
    ]


def article_rank(hits: list, expected_articles: list[str]) -> int | None:
    expected = set(expected_articles)
    for rank, hit in enumerate(hits, start=1):
        if hit.metadata.get("article") in expected:
            return rank
    return None


def section_rank(
    hits: list,
    expected_articles: list[str],
    expected_sections: list[str],
) -> int | None:
    articles = set(expected_articles)
    sections = set(expected_sections)
    for rank, hit in enumerate(hits, start=1):
        if (
            hit.metadata.get("article") in articles
            and hit.metadata.get("section") in sections
        ):
            return rank
    return None


def evaluate_retrieval(
    store,
    items: list[dict],
    corpus: list,
    k: int = 5,
    gold_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for item in items:
        hits = store.similarity_search(item["question"], k=k)
        top = hits[0].metadata if hits else {}
        if gold_map and item["id"] in gold_map:
            gold_keys = list(gold_map[item["id"]])
        else:
            gold_keys = gold_keys_for_item(item, corpus)
        retrieved_keys = [chunk_key(hit) for hit in hits]
        relevant = len(set(gold_keys) & set(retrieved_keys))

        rows.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "type": item["type"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "article_rank": article_rank(hits, item["expected_articles"]),
                "section_rank": section_rank(
                    hits,
                    item["expected_articles"],
                    item["expected_sections"],
                ),
                "top1_article": top.get("article"),
                "top1_section": top.get("section"),
                "gold_count": len(gold_keys),
                "relevant_retrieved": relevant,
                "recall_topk": relevant / len(gold_keys) if gold_keys else 0.0,
                "precision_topk": relevant / len(hits) if hits else 0.0,
                "gold_keys": gold_keys,
                "retrieved_keys": retrieved_keys,
            }
        )
    return pd.DataFrame(rows)


def _hit_at(series: pd.Series, k: int) -> float:
    values = series.apply(lambda rank: pd.notna(rank) and rank <= k)
    return float(values.mean())


def _reciprocal_rank(series: pd.Series) -> float:
    values = series.apply(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / float(rank)
    )
    return float(values.mean())


def _recall_at(row: pd.Series, k: int) -> float:
    gold = set(row["gold_keys"])
    if not gold:
        return 0.0
    retrieved = set(row["retrieved_keys"][:k])
    return len(gold & retrieved) / len(gold)


def _precision_at(row: pd.Series, k: int) -> float:
    retrieved = row["retrieved_keys"][:k]
    if not retrieved:
        return 0.0
    gold = set(row["gold_keys"])
    return len(gold & set(retrieved)) / len(retrieved)


def summarize(df: pd.DataFrame, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    summary: dict[str, float | int] = {"questions": int(len(df))}
    for k in ks:
        summary[f"article_hit@{k}"] = round(_hit_at(df["article_rank"], k), 4)
        summary[f"section_hit@{k}"] = round(_hit_at(df["section_rank"], k), 4)
        summary[f"recall@{k}"] = round(
            float(df.apply(lambda row: _recall_at(row, k), axis=1).mean()), 4
        )
        summary[f"precision@{k}"] = round(
            float(df.apply(lambda row: _precision_at(row, k), axis=1).mean()), 4
        )
    summary["mrr"] = round(_reciprocal_rank(df["article_rank"]), 4)
    return summary


def summarize_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic, group in df.groupby("topic", sort=True):
        rows.append(
            {
                "topic": topic,
                "n": len(group),
                "article_hit@1": round(_hit_at(group["article_rank"], 1), 3),
                "article_hit@3": round(_hit_at(group["article_rank"], 3), 3),
                "section_hit@3": round(_hit_at(group["section_rank"], 3), 3),
                "recall@5": round(
                    float(group.apply(lambda row: _recall_at(row, 5), axis=1).mean()), 3
                ),
                "precision@5": round(
                    float(group.apply(lambda row: _precision_at(row, 5), axis=1).mean()), 3
                ),
                "mrr": round(_reciprocal_rank(group["article_rank"]), 3),
            }
        )
    return pd.DataFrame(rows)


def save_results(
    df: pd.DataFrame,
    summary: dict,
    path: str | Path,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": df.to_dict(orient="records")}
    if extra:
        payload.update(extra)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
