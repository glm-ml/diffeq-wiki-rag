from eval.tool_metrics import evaluate_tool_questions, sequence_accuracy, tool_metrics


def test_exact_tool_choice():
    metrics = tool_metrics(["search_wikipedia"], ["search_wikipedia"])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["exact_match"] is True
    assert metrics["at_least_one_correct"] is True
    assert metrics["unnecessary_rate"] == 0.0


def test_missing_tool():
    metrics = tool_metrics(
        ["search_wikipedia", "run_python"],
        ["search_wikipedia"],
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.6667
    assert metrics["exact_match"] is False
    assert metrics["missing_tools"] == ["run_python"]


def test_extra_tool():
    metrics = tool_metrics(["run_python"], ["run_python", "web_search"])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["unnecessary_rate"] == 0.5
    assert metrics["extra_tools"] == ["web_search"]


def test_wrong_tool():
    metrics = tool_metrics(["run_python"], ["search_wikipedia"])
    assert metrics["f1"] == 0.0
    assert metrics["at_least_one_correct"] is False


def test_sequence_accuracy():
    assert sequence_accuracy(
        ["search_wikipedia", "run_python"],
        ["search_wikipedia", "run_python"],
    )
    assert not sequence_accuracy(
        ["search_wikipedia", "run_python"],
        ["run_python", "search_wikipedia"],
    )


def test_evaluate_tool_questions():
    items = [
        {
            "id": "q1",
            "question": "?",
            "expected_tools": ["run_python"],
            "expected_tool_sequence": ["run_python"],
        },
        {
            "id": "q2",
            "question": "?",
            "expected_tools": ["search_wikipedia", "run_python"],
            "expected_tool_sequence": ["search_wikipedia", "run_python"],
        },
    ]
    frame, summary = evaluate_tool_questions(
        items,
        {"q1": ["run_python"], "q2": ["search_wikipedia"]},
    )

    assert len(frame) == 2
    assert summary["tool_choice_accuracy"] == 1.0
    assert summary["exact_tool_set_accuracy"] == 0.5
    assert summary["tool_recall"] == 0.75
    assert summary["sequence_accuracy"] == 0.5
