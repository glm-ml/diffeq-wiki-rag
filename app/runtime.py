"""Общее состояние процесса: RAG-конвейер и граф агента."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph
from app.agent.tools import set_pipeline_getter
from app.rag.pipeline import RagPipeline, build_rag_pipeline


class Runtime:
    def __init__(self) -> None:
        self.pipeline: RagPipeline | None = None
        self.graph = None
        self.checkpointer: MemorySaver | None = None

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and self.graph is not None

    def initialize(self) -> None:
        self.pipeline = build_rag_pipeline()
        set_pipeline_getter(lambda: self.pipeline)
        self.checkpointer = MemorySaver()
        self.graph = build_agent_graph(checkpointer=self.checkpointer)

    def reset(self) -> None:
        set_pipeline_getter(None)
        self.pipeline = None
        self.graph = None
        self.checkpointer = None


runtime = Runtime()
