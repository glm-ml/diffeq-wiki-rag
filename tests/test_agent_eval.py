from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.agent_eval import (
    passage_source_count,
    run_agent,
    tool_context,
    tool_sequence,
)


def _messages() -> list:
    return [
        HumanMessage(content="Что такое вронскиан?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_wikipedia",
                    "args": {"query": "вронскиан"},
                    "id": "1",
                }
            ],
        ),
        ToolMessage(
            content=(
                "[1] Wronskian — Definition\nURL: https://example.org/W\n"
                "Определитель.\n\n"
                "[2] Wronskian — Properties\nURL: https://example.org/W\n"
                "Свойство."
            ),
            name="search_wikipedia",
            tool_call_id="1",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_python",
                    "args": {"code": "print(1)"},
                    "id": "2",
                }
            ],
        ),
        ToolMessage(
            content="1",
            name="run_python",
            tool_call_id="2",
        ),
        AIMessage(content="Вронскиан — определитель [1]."),
    ]


def test_tool_sequence_keeps_order_and_duplicates():
    messages = _messages() + [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_wikipedia",
                    "args": {"query": "ещё"},
                    "id": "3",
                }
            ],
        )
    ]
    assert tool_sequence(messages) == [
        "search_wikipedia",
        "run_python",
        "search_wikipedia",
    ]


def test_tool_context_contains_observations():
    context = tool_context(_messages())
    assert "### search_wikipedia" in context
    assert "Определитель" in context
    assert "### run_python" in context


def test_passage_source_count_uses_search_only():
    assert passage_source_count(_messages()) == 2


def test_passage_source_count_zero_without_search():
    messages = [HumanMessage(content="?"), AIMessage(content="Ответ без поиска.")]
    assert passage_source_count(messages) == 0


def test_run_agent_reports_latency_and_trace():
    class FakeGraph:
        def invoke(self, state, config):
            return {
                "messages": _messages(),
                "iterations": 2,
            }

    result = run_agent(FakeGraph(), "Что такое вронскиан?", "thread-1")

    assert result["answer"] == "Вронскиан — определитель [1]."
    assert result["iterations"] == 2
    assert result["tool_sequence"] == ["search_wikipedia", "run_python"]
    assert result["n_sources"] == 2
    assert result["latency_ms"] >= 0
