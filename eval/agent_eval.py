"""Прогон агента и разбор его сообщений для answer-level оценки."""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

PASSAGE_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


def tool_sequence(messages: list[Any]) -> list[str]:
    """Имена инструментов в порядке вызовов, включая повторы."""
    sequence: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            sequence.append(call["name"])
    return sequence


def tool_context(messages: list[Any]) -> str:
    """Собрать наблюдения инструментов — контекст, который видел агент."""
    blocks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = message.name or "tool"
        blocks.append(f"### {name}\n{message.content}")
    return "\n\n".join(blocks)


def passage_source_count(messages: list[Any]) -> int:
    """Сколько пронумерованных пассажей вернул search_wikipedia."""
    highest = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "search_wikipedia":
            continue
        for value in PASSAGE_RE.findall(str(message.content)):
            highest = max(highest, int(value))
    return highest


def run_agent(graph, question: str, thread_id: str) -> dict:
    """Один прогон агента с замером latency и разбором трейса."""
    started = time.perf_counter()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "iterations": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    messages = result["messages"]

    return {
        "answer": str(messages[-1].content),
        "iterations": result.get("iterations", 0),
        "tool_sequence": tool_sequence(messages),
        "context": tool_context(messages),
        "n_sources": passage_source_count(messages),
        "latency_ms": latency_ms,
    }
