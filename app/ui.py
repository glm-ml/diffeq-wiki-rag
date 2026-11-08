"""Gradio-интерфейс: быстрый RAG и агентный режим со стримингом."""

from __future__ import annotations

import json
import time
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from app.agent.guardrails import GuardrailError, check_input, check_output
from app.agent.trace import build_trace, extract_sources
from app.runtime import runtime
from app.schemas import Source

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

EMPTY_SOURCES = "### Источники\n\n_Пока пусто_"
EMPTY_TIMINGS = "### Тайминги\n\n_Задайте вопрос_"


def _sources_from_documents(documents) -> str:
    if not documents:
        return EMPTY_SOURCES
    lines = ["### Источники", ""]
    for index, document in enumerate(documents, start=1):
        article = document.metadata.get("article", "Wikipedia")
        section = document.metadata.get("section", "")
        subsection = document.metadata.get("subsection", "")
        heading = " / ".join(part for part in (section, subsection) if part)
        snippet = document.page_content[:160].replace("\n", " ").strip()
        url = document.metadata.get("source", "")
        lines.append(f"**[{index}] {article}** — {heading}")
        lines.append(f"<{url}>")
        lines.append(f"> {snippet}…")
        lines.append("")
    return "\n".join(lines)


def _sources_from_agent(sources: list[Source]) -> str:
    if not sources:
        return "### Источники\n\n_Агент не обращался к корпусу_"
    lines = ["### Источники", ""]
    for index, source in enumerate(sources, start=1):
        lines.append(f"**[{index}] {source.title}**")
        lines.append(f"<{source.url}>")
        lines.append("")
    return "\n".join(lines)


def _timings(retrieval_ms: float, llm_ms: float | None, ttft_ms: float | None = None) -> str:
    lines = ["### Тайминги", "", f"- Поиск: **{retrieval_ms:.0f} мс**"]
    if ttft_ms is not None:
        lines.append(f"- Первый токен: **{ttft_ms:.0f} мс**")
    if llm_ms is not None:
        lines.append(f"- Генерация: **{llm_ms:.0f} мс**")
        lines.append(f"- Итого: **{retrieval_ms + llm_ms:.0f} мс**")
    return "\n".join(lines)


def _trace_markdown(steps) -> str:
    if not steps:
        return "_Без вызовов инструментов_"
    lines = ["### Шаги агента", ""]
    for step in steps:
        arguments = json.dumps(step.arguments, ensure_ascii=False)
        observation = step.observation[:160].replace("\n", " ")
        lines.append(f"**{step.step}. `{step.tool}`** · `{arguments}`")
        if observation:
            lines.append(f"> {observation}…")
        lines.append("")
    return "\n".join(lines)


def _append_error(history, message: str, error: str):
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"⚠️ {error}"},
    ]


def rag_stream(message: str, history: list):
    """Быстрый режим: поиск -> стриминг ответа."""
    if not message or not message.strip():
        yield history, "", EMPTY_TIMINGS, EMPTY_SOURCES, "_—_", "💤 Жду вопрос", ""
        return

    try:
        question = check_input(message)
    except GuardrailError as exc:
        history = _append_error(history, message, f"Запрос отклонён: {exc}")
        yield history, "", "_Отклонён_", EMPTY_SOURCES, "_—_", "🛡 Guardrail", ""
        return

    if runtime.pipeline is None:
        history = _append_error(history, message, "Сервис ещё инициализируется")
        yield history, "", "_Нет данных_", EMPTY_SOURCES, "_—_", "⏳ Инициализация", ""
        return

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]

    started = time.perf_counter()
    documents = runtime.pipeline.retrieve(question)
    retrieval_ms = (time.perf_counter() - started) * 1000
    sources = _sources_from_documents(documents)
    yield (
        history,
        "",
        _timings(retrieval_ms, None),
        sources,
        "_Быстрый режим: без инструментов_",
        "✍️ Генерирую ответ…",
        "",
    )

    generation_started = time.perf_counter()
    ttft_ms = None
    accumulated = ""
    try:
        for chunk in runtime.pipeline.stream_answer(question, documents):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - generation_started) * 1000
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield (
                history,
                "",
                _timings(
                    retrieval_ms,
                    (time.perf_counter() - generation_started) * 1000,
                    ttft_ms,
                ),
                sources,
                "_Быстрый режим: без инструментов_",
                f"✍️ {len(accumulated)} символов…",
                "",
            )
    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ LLM временно недоступна ({type(exc).__name__}). "
            "Попробуйте через 30–60 секунд."
        )
        yield history, "", _timings(retrieval_ms, None), sources, "_—_", "❌ Ошибка LLM", ""
        return

    total_ms = (time.perf_counter() - started) * 1000
    yield (
        history,
        "",
        _timings(retrieval_ms, total_ms - retrieval_ms, ttft_ms),
        sources,
        "_Быстрый режим: без инструментов_",
        f"✅ Готово за {total_ms / 1000:.1f} с",
        "",
    )


def agent_stream(message: str, history: list, thread_id_state: str):
    """Агентный режим: обновления графа и токены финального ответа."""
    if not message or not message.strip():
        yield history, "", EMPTY_TIMINGS, EMPTY_SOURCES, "_—_", "💤 Жду вопрос", thread_id_state
        return

    try:
        question = check_input(message)
    except GuardrailError as exc:
        history = _append_error(history, message, f"Запрос отклонён: {exc}")
        yield history, "", "_Отклонён_", EMPTY_SOURCES, "_—_", "🛡 Guardrail", thread_id_state
        return

    if runtime.graph is None:
        history = _append_error(history, message, "Сервис ещё инициализируется")
        yield history, "", "_Нет данных_", EMPTY_SOURCES, "_—_", "⏳ Инициализация", thread_id_state
        return

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    thread_id = thread_id_state or str(uuid.uuid4())

    started = time.perf_counter()
    yield history, "", EMPTY_TIMINGS, EMPTY_SOURCES, "_—_", "🤔 Агент выбирает инструмент…", thread_id

    accumulated = ""
    collected = []
    tools_used: list[str] = []
    iterations = 0

    try:
        for mode, payload in runtime.graph.stream(
            {"messages": [HumanMessage(content=question)], "iterations": 0},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                node, update = next(iter(payload.items()))
                node_messages = update.get("messages", []) or []
                collected.extend(node_messages)

                if node == "agent" and node_messages:
                    iterations += 1
                    calls = getattr(node_messages[-1], "tool_calls", None) or []
                    if calls:
                        tool = calls[0]["name"]
                        tools_used.append(tool)
                        status = {
                            "search_wikipedia": "📚 Ищу в статьях Википедии…",
                            "run_python": "🧮 Считаю в Python…",
                            "web_search": "🌐 Ищу в интернете…",
                        }.get(tool, f"🛠 Вызываю {tool}…")
                        yield history, "", EMPTY_TIMINGS, EMPTY_SOURCES, "_—_", status, thread_id
                elif node == "tools" and node_messages:
                    name = getattr(node_messages[-1], "name", "tool")
                    size = len(str(getattr(node_messages[-1], "content", "")))
                    yield (
                        history,
                        "",
                        EMPTY_TIMINGS,
                        EMPTY_SOURCES,
                        "_—_",
                        f"🔍 Разбираю результат {name} ({size} симв.)…",
                        thread_id,
                    )

            elif mode == "messages":
                chunk, metadata = payload
                text = getattr(chunk, "content", "") or ""
                is_final_node = metadata.get("langgraph_node") == "agent"
                is_tool_call = bool(getattr(chunk, "tool_call_chunks", None))
                if text and is_final_node and not is_tool_call:
                    accumulated += text
                    history[-1]["content"] = accumulated
                    yield (
                        history,
                        "",
                        EMPTY_TIMINGS,
                        EMPTY_SOURCES,
                        "_—_",
                        f"✍️ {len(accumulated)} символов…",
                        thread_id,
                    )
    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ Агент недоступен ({type(exc).__name__}). "
            "Попробуйте через 30–60 секунд."
        )
        yield history, "", EMPTY_TIMINGS, EMPTY_SOURCES, "_—_", "❌ Ошибка агента", thread_id
        return

    if not accumulated:
        for item in reversed(collected):
            if getattr(item, "content", None) and not getattr(item, "tool_calls", None):
                accumulated = str(item.content)
                break

    steps, used = build_trace(collected)
    if not tools_used:
        tools_used = used
    safe_answer, guardrail = check_output(accumulated, tools_used)
    history[-1]["content"] = safe_answer

    sources = _sources_from_agent(extract_sources(collected))
    total_ms = (time.perf_counter() - started) * 1000
    timings = (
        "### Тайминги\n\n"
        f"- Итого: **{total_ms:.0f} мс**\n"
        f"- Итераций: **{iterations}**\n"
        f"- Вызовов инструментов: **{len(tools_used)}**"
    )
    status = (
        f"✅ Готово за {total_ms / 1000:.1f} с · "
        f"инструментов: {len(tools_used)}"
    )
    if guardrail:
        status += f" · guardrail: {guardrail}"
    yield history, "", timings, sources, _trace_markdown(steps), status, thread_id


def route(message: str, history: list, mode: str, thread_id_state: str):
    if mode == "Агент":
        yield from agent_stream(message, history, thread_id_state)
    else:
        for history_out, textbox, timings, sources, trace, status, _ in rag_stream(
            message, history
        ):
            yield history_out, textbox, timings, sources, trace, status, thread_id_state


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Дифференциальные уравнения — RAG", fill_height=True) as demo:
        gr.Markdown(
            "# Дифференциальные уравнения — RAG\n"
            "Ответы по разделам Википедии: условие Липшица, линейные ДУ "
            "n-го порядка, теорема Пикара–Линделёфа, вронскиан."
        )
        status_line = gr.Markdown("💤 Жду вопрос", elem_id="status-line")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=520,
                    show_label=False,
                    latex_delimiters=LATEX_DELIMITERS,
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Например: почему условие Липшица гарантирует единственность решения?",
                        scale=4,
                        show_label=False,
                    )
                    send = gr.Button("Отправить", scale=1, variant="primary")

            with gr.Column(scale=1, elem_id="side-panel"):
                mode = gr.Radio(
                    choices=["Быстрый RAG", "Агент"],
                    value="Быстрый RAG",
                    label="Режим",
                )
                timings = gr.Markdown(EMPTY_TIMINGS)
                sources = gr.Markdown(EMPTY_SOURCES)
                with gr.Accordion("Что сделал агент", open=False):
                    trace = gr.Markdown("_Включите режим «Агент»_")

        thread_id_state = gr.State("")

        inputs = [message, chatbot, mode, thread_id_state]
        outputs = [chatbot, message, timings, sources, trace, status_line, thread_id_state]

        message.submit(route, inputs, outputs)
        send.click(route, inputs, outputs)

    return demo
