"""CLI для ручной проверки агента: python -m app.agent.runner "вопрос"."""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.tools import set_pipeline_getter
from app.rag.pipeline import build_rag_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m app.agent.runner 'ваш вопрос'")
        raise SystemExit(1)

    pipeline = build_rag_pipeline()
    set_pipeline_getter(lambda: pipeline)
    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=" ".join(sys.argv[1:]))], "iterations": 0},
        config={"configurable": {"thread_id": "cli"}},
    )
    print("\n=== ОТВЕТ ===\n")
    print(result["messages"][-1].content)
    print(f"\n(итераций: {result['iterations']})")


if __name__ == "__main__":
    main()
