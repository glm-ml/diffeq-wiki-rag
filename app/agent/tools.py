"""Инструменты агента: поиск по корпусу, вычисления, веб-поиск."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from typing import Callable

from langchain_core.tools import tool

from app.rag.pipeline import RagPipeline
from app.settings import get_settings

_pipeline_getter: Callable[[], RagPipeline] | None = None


def set_pipeline_getter(getter: Callable[[], RagPipeline]) -> None:
    """Связать инструменты с уже собранным RAG-конвейером."""
    global _pipeline_getter
    _pipeline_getter = getter


def _pipeline() -> RagPipeline:
    if _pipeline_getter is None:
        raise RuntimeError("RAG-конвейер не инициализирован")
    return _pipeline_getter()


@tool
def search_wikipedia(query: str) -> str:
    """Найти релевантные пассажи в статьях Википедии о дифференциальных уравнениях.

    Возвращает пронумерованные фрагменты и URL. Не генерирует ответ —
    синтез ответа остаётся за агентом.

    Args:
        query: вопрос или ключевые слова на русском либо английском.
    """
    documents = _pipeline().retrieve(query)
    if not documents:
        return "Ничего не найдено в корпусе."

    blocks: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = metadata.get("article", "Wikipedia")
        section = metadata.get("section", "")
        url = metadata.get("source", "")
        heading = " / ".join(
            part
            for part in (section, metadata.get("subsection", ""))
            if part
        )
        blocks.append(
            f"[{index}] {title} — {heading}\n"
            f"URL: {url}\n"
            f"{document.page_content}"
        )
        if url and url not in seen:
            seen.add(url)
            sources.append(f"- {title} | {url}")

    return "\n\n".join(blocks) + "\n\nSources:\n" + "\n".join(sources)


@tool
def run_python(code: str) -> str:
    """Выполнить Python-код в отдельном процессе и вернуть stdout.

    Используй для арифметики, проверки формул и небольших расчётов.
    Результат выводи через print().

    Args:
        code: исходный код на Python.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            cwd=tempfile.mkdtemp(prefix="diffeq-repl-"),
        )
    except subprocess.TimeoutExpired:
        return "Ошибка: выполнение превысило 10 секунд."

    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return f"Ошибка выполнения:\n{error[:1500]}"
    return output[:3000] or "(код выполнен без вывода — используйте print())"


@tool
def web_search(query: str) -> str:
    """Найти информацию в интернете через DuckDuckGo.

    Args:
        query: поисковый запрос.
    """
    if not get_settings().enable_web_search:
        return "Веб-поиск отключён в настройках."

    try:
        from ddgs import DDGS

        with DDGS() as engine:
            results = list(engine.text(query, max_results=3))
    except Exception as exc:  # сеть и rate-limit не должны ронять агента
        return f"Веб-поиск недоступен: {type(exc).__name__}"

    if not results:
        return "Ничего не найдено."

    lines = []
    for item in results:
        title = item.get("title", "")
        body = item.get("body", "")
        url = item.get("href", "")
        lines.append(f"- {title}: {body}\n  {url}")
    return "\n".join(lines)


TOOLS = [search_wikipedia, run_python, web_search]
