from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import AgentGraph
from app.settings import get_settings


def _tool_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_wikipedia", "args": {"query": "Липшиц"}, "id": "1"}
        ],
    )


def test_should_continue_routes_to_tools():
    state = {"messages": [HumanMessage(content="?"), _tool_call_message()], "iterations": 1}
    assert AgentGraph.should_continue(state) == "tools"


def test_should_continue_stops_without_tool_calls():
    state = {"messages": [AIMessage(content="Готовый ответ")], "iterations": 1}
    assert AgentGraph.should_continue(state) == "__end__"


def test_should_continue_stops_at_iteration_limit(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "max_iterations", 2)
    state = {"messages": [_tool_call_message()], "iterations": 2}
    assert AgentGraph.should_continue(state) == "finalize"


def test_graph_has_finalize_node():
    graph = AgentGraph().build()
    assert "finalize" in graph.get_graph().nodes


def test_finalize_messages_closes_pending_tool_calls():
    from langchain_core.messages import ToolMessage

    state = {
        "messages": [HumanMessage(content="?"), _tool_call_message()],
        "iterations": 2,
    }
    messages = AgentGraph._finalize_messages(state)

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "1"
    assert "лимит итераций" in tool_messages[0].content


def test_graph_compiles_without_llm_call():
    graph = AgentGraph().build()
    assert graph is not None
