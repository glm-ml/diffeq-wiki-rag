"""Метрики корректности выбора инструментов агентом."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def tool_metrics(
    expected_tools: Iterable[str],
    used_tools: Iterable[str],
) -> dict:
    expected = set(expected_tools)
    used = set(used_tools)
    true_positive = len(expected & used)

    precision = (
        true_positive / len(used)
        if used
        else (1.0 if not expected else 0.0)
    )
    recall = true_positive / len(expected) if expected else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": expected == used,
        "at_least_one_correct": true_positive > 0,
        "unnecessary_rate": (
            round(len(used - expected) / len(used), 4) if used else 0.0
        ),
        "missing_tools": sorted(expected - used),
        "extra_tools": sorted(used - expected),
    }


def sequence_accuracy(expected: list[str], used: list[str]) -> bool:
    return list(expected) == list(used)


def evaluate_tool_questions(
    items: list[dict],
    tools_by_id: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for item in items:
        used = tools_by_id.get(item["id"], [])
        metrics = tool_metrics(item["expected_tools"], used)
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_tools": item["expected_tools"],
            "used_tools": used,
            **metrics,
        }
        if item.get("expected_tool_sequence"):
            row["sequence_ok"] = sequence_accuracy(
                item["expected_tool_sequence"], used
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {
        "questions": len(frame),
        "tool_choice_accuracy": round(frame["at_least_one_correct"].mean(), 4),
        "exact_tool_set_accuracy": round(frame["exact_match"].mean(), 4),
        "tool_precision": round(frame["precision"].mean(), 4),
        "tool_recall": round(frame["recall"].mean(), 4),
        "tool_f1": round(frame["f1"].mean(), 4),
        "unnecessary_tool_rate": round(frame["unnecessary_rate"].mean(), 4),
    }
    if "sequence_ok" in frame:
        summary["sequence_accuracy"] = round(frame["sequence_ok"].mean(), 4)
    return frame, summary
