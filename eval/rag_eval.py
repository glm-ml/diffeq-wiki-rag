"""Генерация ответов и LLM-судья для answer-level экспериментов."""

from __future__ import annotations

import json
import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm
from app.rag.prompt import HUMAN_PROMPT, SYSTEM_PROMPT

JUDGE_PROMPT = """Ты — строгий оценщик RAG-ответов.

Вопрос: {question}
Эталонный ответ: {reference}
Ключевые факты: {key_facts}

Контекст, который был у системы:
{context}

Ответ системы:
{answer}

Оцени по шкале 1–5:
- correctness: насколько ответ соответствует эталону и ключевым фактам;
- faithfulness: все ли утверждения ответа подтверждаются контекстом
  (5 — полностью, 1 — много выдумок);
- citations: корректны ли ссылки [n] и указывают ли они на релевантные источники
  (5 — все корректны, 1 — ссылок нет или они неверные).

Верни только JSON без markdown:
{{"correctness": 1, "faithfulness": 1, "citations": 1, "comment": "кратко"}}
"""


def build_answer_chain(llm=None):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return prompt | (llm or get_llm()) | StrOutputParser()


def answer_with_context(chain, question: str, context: str) -> str:
    return chain.invoke({"question": question, "context": context})


def citation_stats(answer: str, n_sources: int) -> dict:
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    valid = [value for value in citations if 1 <= value <= n_sources]
    return {
        "citations": len(citations),
        "valid_citations": len(valid),
        "has_citations": bool(citations),
        "valid_ratio": (len(valid) / len(citations)) if citations else 0.0,
    }


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def judge_answer(
    question: str,
    reference: str,
    key_facts: list[str],
    context: str,
    answer: str,
    llm=None,
) -> dict | None:
    judge_llm = llm or get_llm()
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        key_facts="; ".join(key_facts),
        context=context,
        answer=answer,
    )

    for _ in range(3):
        try:
            response = judge_llm.invoke(prompt)
            parsed = _parse_json(str(response.content))
            if parsed is None:
                raise ValueError("не JSON")
            result = {
                "correctness": int(parsed["correctness"]),
                "faithfulness": int(parsed["faithfulness"]),
                "citations": int(parsed["citations"]),
                "comment": str(parsed.get("comment", "")),
            }
            for key in ("correctness", "faithfulness", "citations"):
                result[key] = max(1, min(5, result[key]))
            return result
        except Exception:
            time.sleep(1.5)

    return None
