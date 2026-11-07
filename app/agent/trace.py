"""Извлечение шагов агента и источников из истории сообщений."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from app.schemas import Source, TraceStep


def build_trace(messages: list[AnyMessage]) -> tuple[list[TraceStep], list[str]]:
    steps: list[TraceStep] = []
    tools_used: list[str] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                tools_used.append(call["name"])
                steps.append(
                    TraceStep(
                        step=len(steps) + 1,
                        tool=call["name"],
                        arguments=call.get("args", {}),
                        observation="",
                    )
                )
        elif isinstance(message, ToolMessage):
            if steps and not steps[-1].observation:
                steps[-1].observation = str(message.content)[:600]

    return steps, tools_used


def extract_sources(messages: list[AnyMessage]) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content)
        if "Sources:" not in content:
            continue
        for line in content.split("Sources:", 1)[1].splitlines():
            line = line.strip().lstrip("- ").strip()
            if not line:
                continue
            title, _, url = line.partition(" | ")
            if not url:
                title, url = "Wikipedia", title
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            sources.append(Source(title=title, url=url, snippet=""))

    return sources
