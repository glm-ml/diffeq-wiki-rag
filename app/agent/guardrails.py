"""Проверки входа и выхода агента."""

from __future__ import annotations

import re

from app.settings import get_settings

_INJECTION_PATTERNS = (
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior) instructions",
    r"system prompt",
    r"<\|im_(start|end)\|>",
    r"игнорируй (все )?(предыдущие|прошлые) инструкции",
    r"забудь (все )?(предыдущие|прошлые) инструкции",
    r"покажи (системный )?промпт",
)
_COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_DECLARATIVE_MARKERS = (
    "согласно",
    "по документации",
    "в корпусе сказано",
    "according to",
    "the documentation says",
)


class GuardrailError(ValueError):
    """Запрос не прошёл входную проверку."""


def check_input(question: str) -> str:
    text = question.strip()
    if not 2 <= len(text) <= 600:
        raise GuardrailError("длина вопроса должна быть от 2 до 600 символов")
    if _CONTROL.search(text):
        raise GuardrailError("в вопросе есть управляющие символы")
    for pattern in _COMPILED:
        if pattern.search(text):
            raise GuardrailError("запрос похож на попытку prompt injection")
    return text


def check_output(answer: str, tools_used: list[str]) -> tuple[str, str | None]:
    """Обрезать слишком длинный ответ и проверить ссылки на корпус."""
    limit = get_settings().max_output_chars
    if len(answer) > limit:
        return answer[:limit] + "\n\n…(ответ обрезан)", "output_too_long"

    lowered = answer.casefold()
    claims_corpus = any(marker in lowered for marker in _DECLARATIVE_MARKERS)
    used_corpus = "search_wikipedia" in tools_used
    if claims_corpus and not used_corpus:
        return (
            "Не могу сослаться на корпус, не обратившись к нему. "
            "Уточните вопрос — я сначала найду нужный раздел статьи.",
            "claim_without_retrieval",
        )
    return answer, None
