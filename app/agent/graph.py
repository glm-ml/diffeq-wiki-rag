"""Граф агента: LLM ↔ инструменты до финального ответа."""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

import structlog
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.prompt import FINALIZE_PROMPT, SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.llm import get_llm
from app.settings import get_settings

log = structlog.get_logger()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int


class AgentGraph:
    """Ленивая обёртка: LLM с инструментами создаётся один раз на процесс."""

    def __init__(self):
        self._llm = None
        self._final_llm = None

    def _llm_with_tools(self):
        if self._llm is None:
            self._llm = get_llm().bind_tools(TOOLS)
        return self._llm

    def _final_llm_instance(self):
        if self._final_llm is None:
            self._final_llm = get_llm()
        return self._final_llm

    def agent_node(self, state: AgentState) -> dict:
        started = time.perf_counter()
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = self._llm_with_tools().invoke(messages)
        log.info(
            "agent_step",
            iteration=state.get("iterations", 0) + 1,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tools=[call["name"] for call in getattr(response, "tool_calls", [])],
        )
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    def finalize_node(self, state: AgentState) -> dict:
        """Принудительный финальный ответ, когда лимит инструментов исчерпан."""
        started = time.perf_counter()
        messages = self._finalize_messages(state)
        response = self._final_llm_instance().invoke(messages)
        content = str(response.content).strip()
        if not content:
            response = AIMessage(
                content=(
                    "Не удалось завершить ответ в рамках лимита инструментов. "
                    "Уточните вопрос или попробуйте снова."
                )
            )
        log.info(
            "finalize_step",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return {"messages": [response]}

    @staticmethod
    def _finalize_messages(state: AgentState) -> list[AnyMessage]:
        """Закрыть висящие tool_calls, чтобы провайдер принял историю."""
        messages: list[AnyMessage] = [
            SystemMessage(content=FINALIZE_PROMPT),
            *state["messages"],
        ]
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            for call in last.tool_calls:
                messages.append(
                    ToolMessage(
                        content=(
                            "Инструмент не был вызван: достигнут лимит итераций. "
                            "Сформулируй ответ по уже собранной информации."
                        ),
                        tool_call_id=call.get("id", ""),
                        name=call["name"],
                    )
                )
        return messages

    @staticmethod
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tool_calls = bool(getattr(last, "tool_calls", None))
        if has_tool_calls:
            if state.get("iterations", 0) >= get_settings().max_iterations:
                return "finalize"
            return "tools"
        return END

    def build(self, checkpointer=None):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(TOOLS))
        graph.add_node("finalize", self.finalize_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self.should_continue)
        graph.add_edge("tools", "agent")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)


def build_agent_graph(checkpointer=None):
    return AgentGraph().build(checkpointer=checkpointer)
