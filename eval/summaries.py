"""Генерация кратких саммари разделов для эксперимента с retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.llm import get_llm

CACHE_PATH = Path("eval/results/section_summaries.json")
MAX_SECTION_CHARS = 3000


def collect_sections(corpus: list) -> dict[str, dict]:
    """Сгруппировать чанки по (article, section) и собрать текст раздела."""
    sections: dict[str, dict] = {}
    for document in corpus:
        meta = document.metadata
        key = f"{meta['article']}::{meta['section']}"
        entry = sections.setdefault(
            key,
            {
                "article": meta["article"],
                "section": meta["section"],
                "parts": [],
            },
        )
        if sum(len(part) for part in entry["parts"]) < MAX_SECTION_CHARS:
            entry["parts"].append(document.page_content)

    for entry in sections.values():
        entry["text"] = "\n\n".join(entry.pop("parts"))[:MAX_SECTION_CHARS]
    return sections


def _prompt(entry: dict, language: str) -> str:
    if language == "en":
        instruction = (
            "Write 1-2 sentences (maximum 40 words) describing what this section "
            "is about and what distinguishes it from other sections of the same "
            "article. Use key terms. Answer only with the summary."
        )
    else:
        instruction = (
            "Напиши 1–2 предложения (максимум 40 слов) о том, чему посвящён этот "
            "раздел и чем он отличается от других разделов этой же статьи. "
            "Используй ключевые термины. Верни только саммари."
        )

    return f"""{instruction}

Article: {entry['article']}
Section: {entry['section']}

Section text:
{entry['text']}
"""


def load_cache(path: str | Path = CACHE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: str | Path = CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_section_summaries(
    sections: dict[str, dict],
    languages: tuple[str, ...] = ("en", "ru"),
    path: str | Path = CACHE_PATH,
) -> dict:
    """Сгенерировать саммари и закэшировать их в JSON."""
    cache = load_cache(path)
    llm = get_llm()

    for key, entry in sections.items():
        record = cache.setdefault(
            key,
            {"article": entry["article"], "section": entry["section"]},
        )
        for language in languages:
            field = f"summary_{language}"
            if record.get(field):
                continue

            last_error = None
            for _ in range(3):
                try:
                    response = llm.invoke(_prompt(entry, language))
                    text = str(response.content).strip()
                    if not text:
                        raise ValueError("пустой ответ")
                    record[field] = text
                    last_error = None
                    break
                except Exception as exc:  # сеть/rate-limit
                    last_error = exc
                    time.sleep(1.5)

            if last_error is not None:
                raise RuntimeError(f"{key}/{language}: {last_error}") from last_error

            save_cache(cache, path)
            print(f"{key} [{language}]: {record[field][:90]}...")

    return cache


def summary_for(cache: dict, article: str, section: str, language: str) -> str:
    record = cache.get(f"{article}::{section}", {})
    return record.get(f"summary_{language}", "").strip()
