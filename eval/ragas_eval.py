"""Оценка RAG-ответов метриками RAGAS."""

from __future__ import annotations

import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

from app.llm import get_llm
from app.rag.store import get_embeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def build_samples(rows: list[dict]) -> list:
    return [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=list(row["contexts"]),
            reference=row["reference"],
        )
        for row in rows
    ]


def records_from_frame(rows: list[dict], frame: pd.DataFrame) -> list[dict]:
    records = []
    for row, (_, scored) in zip(rows, frame.iterrows()):
        record = {"id": row["id"], "question": row["question"]}
        for name in METRIC_NAMES:
            value = scored.get(name)
            record[name] = (
                None if value is None or pd.isna(value) else round(float(value), 4)
            )
        records.append(record)
    return records


def summarize_frame(frame: pd.DataFrame) -> dict:
    summary = {"questions": int(len(frame))}
    for name in METRIC_NAMES:
        if name in frame:
            summary[name] = round(float(frame[name].mean(skipna=True)), 4)
    summary["failed"] = int(
        frame[[name for name in METRIC_NAMES if name in frame]].isna().any(axis=1).sum()
    )
    return summary


def evaluate_ragas(
    rows: list[dict],
    max_workers: int = 4,
    timeout: int = 240,
    max_retries: int = 3,
) -> dict:
    """Посчитать RAGAS-метрики для ответов RAG.

    rows: id, question, answer, contexts (list[str]), reference.
    """
    dataset = EvaluationDataset(samples=build_samples(rows))

    # DeepSeek не поддерживает n > 1, поэтому AnswerRelevancy считаем
    # с одной генерацией на вопрос.
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=LangchainLLMWrapper(get_llm()),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
        ),
        raise_exceptions=False,
        show_progress=False,
    )

    frame = result.to_pandas()
    return {
        "summary": summarize_frame(frame),
        "rows": records_from_frame(rows, frame),
    }
