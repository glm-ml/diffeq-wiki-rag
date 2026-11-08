import pandas as pd

from eval.ragas_eval import (
    METRIC_NAMES,
    build_samples,
    records_from_frame,
    summarize_frame,
)


def _rows():
    return [
        {
            "id": "q1",
            "question": "Что такое вронскиан?",
            "answer": "Определитель [1].",
            "contexts": ["Вронскиан — определитель."],
            "reference": "Вронскиан — определитель матрицы.",
        },
        {
            "id": "q2",
            "question": "Что такое условие Липшица?",
            "answer": "Ограничение на скорость изменения.",
            "contexts": ["d(f(x), f(y)) <= K d(x, y)."],
            "reference": "Условие Липшица ограничивает расстояния.",
        },
    ]


def test_build_samples_maps_fields():
    samples = build_samples(_rows())
    assert len(samples) == 2
    assert samples[0].user_input == "Что такое вронскиан?"
    assert samples[0].retrieved_contexts == ["Вронскиан — определитель."]


def test_records_from_frame_keeps_order_and_rounds():
    frame = pd.DataFrame(
        {
            name: [1.0, 0.5] for name in METRIC_NAMES
        }
    )
    records = records_from_frame(_rows(), frame)

    assert [record["id"] for record in records] == ["q1", "q2"]
    assert records[1]["faithfulness"] == 0.5


def test_records_from_frame_marks_nan_as_none():
    frame = pd.DataFrame({name: [1.0, float("nan")] for name in METRIC_NAMES})
    records = records_from_frame(_rows(), frame)

    assert records[0]["context_recall"] == 1.0
    assert records[1]["context_recall"] is None


def test_summarize_frame_counts_failed_rows():
    frame = pd.DataFrame(
        {
            "faithfulness": [1.0, float("nan")],
            "answer_relevancy": [0.8, 0.6],
            "context_precision": [1.0, 0.5],
            "context_recall": [1.0, 0.0],
            "answer_correctness": [0.9, 0.7],
        }
    )
    summary = summarize_frame(frame)

    assert summary["questions"] == 2
    assert summary["failed"] == 1
    assert summary["answer_relevancy"] == 0.7
