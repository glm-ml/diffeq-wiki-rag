"""Pydantic-схемы HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=600)


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    thread_id: str | None = None


class TraceStep(BaseModel):
    step: int
    tool: str
    arguments: dict
    observation: str


class AgentResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[Source]
    trace: list[TraceStep]
    iterations: int
    guardrail: str | None = None


def sources_from_documents(documents, snippet_length: int = 180) -> list[Source]:
    return [
        Source(
            title=document.metadata.get("article", "Wikipedia"),
            url=document.metadata.get("source", ""),
            snippet=document.page_content[:snippet_length].strip(),
        )
        for document in documents
    ]
