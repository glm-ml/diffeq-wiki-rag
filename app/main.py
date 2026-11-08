"""FastAPI-приложение: REST API и Gradio-интерфейс."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from app.agent.guardrails import GuardrailError, check_input, check_output
from app.agent.trace import build_trace, extract_sources
from app.logging_config import configure_logging
from app.meta import SocialMetaMiddleware
from app.runtime import runtime
from app.schemas import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    sources_from_documents,
)
from app.ui import build_ui

configure_logging()

PREVIEW_PATH = Path(__file__).parent / "static" / "preview.png"


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.initialize()
    yield
    runtime.reset()


app = FastAPI(
    title="diffeq-wiki-rag",
    description="Agentic RAG по статьям Википедии о дифференциальных уравнениях",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(SocialMetaMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "ready": str(runtime.ready).lower()}


@app.get("/preview.png", include_in_schema=False)
def preview_image() -> FileResponse:
    return FileResponse(
        PREVIEW_PATH,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        question = check_input(payload.question)
    except GuardrailError as exc:
        raise HTTPException(status_code=422, detail=f"Вопрос отклонён: {exc}") from exc

    if runtime.pipeline is None:
        raise HTTPException(status_code=503, detail="Сервис ещё инициализируется")

    try:
        result = runtime.pipeline.run(question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM временно недоступна: {type(exc).__name__}",
        ) from exc

    return ChatResponse(
        answer=result.answer,
        sources=sources_from_documents(result.documents),
    )


@app.post("/agent", response_model=AgentResponse)
def agent_chat(payload: AgentRequest) -> AgentResponse:
    try:
        question = check_input(payload.question)
    except GuardrailError as exc:
        raise HTTPException(status_code=422, detail=f"Вопрос отклонён: {exc}") from exc

    if runtime.graph is None:
        raise HTTPException(status_code=503, detail="Сервис ещё инициализируется")

    thread_id = payload.thread_id or str(uuid.uuid4())
    try:
        result = runtime.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "iterations": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Агент временно недоступен: {type(exc).__name__}",
        ) from exc

    messages = result["messages"]
    answer = str(messages[-1].content)
    trace, tools_used = build_trace(messages)
    safe_answer, guardrail = check_output(answer, tools_used)

    return AgentResponse(
        answer=safe_answer,
        thread_id=thread_id,
        sources=extract_sources(messages),
        trace=trace,
        iterations=result.get("iterations", 0),
        guardrail=guardrail,
    )


# Gradio монтируется на "/" и перекрывает остальные GET-маршруты,
# поэтому API объявлен выше.
app = gr.mount_gradio_app(app, build_ui(), path="/")
