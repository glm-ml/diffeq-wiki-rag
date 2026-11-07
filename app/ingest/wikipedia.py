"""Загрузка тематических разделов статей Википедии.

Модуль не тянет весь HTML-документ в корпус: из страницы берутся только
содержательные разделы (служебные References / See also / External links
отбрасываются), а формулы из MathML-разметки превращаются в читаемый текст.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml
from bs4 import BeautifulSoup, Tag

USER_AGENT = "diffeq-wiki-rag/0.1 (RAG portfolio project; glebunc@yandex.ru)"

SKIPPED_SECTIONS = {
    "see also",
    "references",
    "citations",
    "sources",
    "notes",
    "footnotes",
    "external links",
    "further reading",
    "bibliography",
    "works cited",
    "notes and references",
    "explanatory notes",
    "general references",
}

_NOISE_SELECTORS = (
    "script",
    "style",
    "sup.reference",
    "span.mw-editsection",
    "span.mwe-math-mathml-a11y",
    "div.navbox",
    "table.navbox",
    "div.navbox-styles",
    "div.vertical-navbox",
    "table.vertical-navbox",
    "div.sidebar",
    "table.sidebar",
    "div.metadata",
    "div.mbox",
    "div.hatnote",
    "div.reflist",
    "div.mw-references-wrap",
    "ol.references",
    "div.sistersitebox",
    "div.side-box",
    "table.infobox",
    "div.infobox",
    "div.mw-empty-elt",
)

_MATH_PREFIX = re.compile(r"^\{\\displaystyle\s*")
_SPACES = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?%)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ArticleSpec:
    title: str
    sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class WikiSection:
    article: str
    section: str
    subsection: str
    url: str
    text: str


def load_articles(path: str | Path = "data/articles.yaml") -> tuple[str, list[ArticleSpec]]:
    """Прочитать список статей. Возвращает (язык, список спецификаций)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    language = str(payload.get("language", "en"))
    specs = []
    for item in payload.get("articles", []):
        if isinstance(item, str):
            specs.append(ArticleSpec(title=item))
            continue
        specs.append(
            ArticleSpec(
                title=item["title"],
                sections=tuple(item.get("sections", ())),
            )
        )
    return language, specs


def fetch_article_html(
    spec: ArticleSpec,
    language: str = "en",
    client: httpx.Client | None = None,
) -> str:
    """Получить HTML отрендеренной статьи через MediaWiki API."""
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
    try:
        response = client.get(
            f"https://{language}.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": spec.title,
                "prop": "text",
                "format": "json",
                "formatversion": "2",
                "redirects": "1",
                "disableeditsection": "1",
                "disabletoc": "1",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(f"Wikipedia API: {payload['error'].get('info', 'unknown error')}")
        return payload["parse"]["text"]
    finally:
        if owns_client:
            client.close()


def extract_sections(
    html: str,
    spec: ArticleSpec,
    language: str = "en",
) -> list[WikiSection]:
    """Разбить HTML статьи на разделы и подразделы, сохранив якоря."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("div.mw-parser-output") or soup.body or soup
    if root is None:
        return []

    for selector in _NOISE_SELECTORS:
        for node in root.select(selector):
            node.decompose()

    _expand_math(root)

    page_url = (
        f"https://{language}.wikipedia.org/wiki/"
        f"{quote(spec.title.replace(' ', '_'))}"
    )
    only = {name.casefold() for name in spec.sections}
    result: list[WikiSection] = []

    # Введение до первого заголовка.
    lead = _collect_children_until_heading(root)
    if lead:
        result.append(
            WikiSection(
                article=spec.title,
                section="Overview",
                subsection="",
                url=page_url,
                text=lead,
            )
        )

    h2, h3 = "", ""
    for heading in root.find_all(["h2", "h3", "h4"]):
        level = int(heading.name[1])
        title = _heading_text(heading)
        if not title:
            continue

        if level == 2:
            h2, h3 = title, ""
        elif level == 3:
            h3 = title
        else:
            h3 = f"{h3} → {title}" if h3 else title

        if h2.casefold() in SKIPPED_SECTIONS:
            continue
        if only and h2.casefold() not in only:
            continue

        wrapper = heading.find_parent(
            "div", class_=lambda value: bool(value) and "mw-heading" in value.split()
        ) or heading
        text = _collect_until_heading(wrapper)
        if len(text) < 40:
            continue

        anchor = heading.get("id") or ""
        result.append(
            WikiSection(
                article=spec.title,
                section=h2 or "Overview",
                subsection=h3,
                url=f"{page_url}#{quote(anchor)}" if anchor else page_url,
                text=text,
            )
        )

    return result


def _heading_text(heading: Tag) -> str:
    return _normalize(heading.get_text(" ", strip=True))


def _is_heading_node(node) -> bool:
    if not isinstance(node, Tag):
        return False
    if node.name in {"h2", "h3", "h4"}:
        return True
    classes = node.get("class") or []
    if node.name == "div" and any("mw-heading" in value for value in classes):
        return node.find(["h2", "h3", "h4"]) is not None
    return False


def _collect_until_heading(start) -> str:
    return _collect_nodes(start.next_siblings)


def _collect_children_until_heading(root) -> str:
    return _collect_nodes(root.children)


def _collect_nodes(nodes) -> str:
    blocks: list[str] = []
    for node in nodes:
        if _is_heading_node(node):
            break
        text = _node_text(node)
        if text:
            blocks.append(text)
    return _normalize("\n\n".join(blocks))


def _node_text(node) -> str:
    """Извлечь текст блока, не разрывая инлайн-элементы на отдельные строки."""
    if not isinstance(node, Tag):
        return str(node).strip()

    if node.name in {"ul", "ol"}:
        items = [
            f"• {item.get_text(' ', strip=True)}"
            for item in node.find_all("li", recursive=False)
            if item.get_text(" ", strip=True)
        ]
        return "\n".join(items)

    if node.name == "dl":
        items = [
            child.get_text(" ", strip=True)
            for child in node.find_all(["dt", "dd"], recursive=False)
            if child.get_text(" ", strip=True)
        ]
        return "\n".join(items)

    if node.name == "table":
        rows = []
        for row in node.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    return node.get_text(" ", strip=True)


def _expand_math(root: Tag) -> None:
    """Заменить картинки формул на их alt-текст (там лежит исходный LaTeX)."""
    for image in root.find_all("img"):
        alt = (image.get("alt") or "").strip()
        if not alt:
            continue
        classes = image.get("class") or []
        if any(value.startswith("mwe-math-fallback") for value in classes):
            alt = _clean_math(alt)
        image.replace_with(f" {alt} ")


def _clean_math(value: str) -> str:
    value = _MATH_PREFIX.sub("", value.strip())
    # Префикс {\displaystyle ...} содержит открывающую скобку, поэтому
    # после его удаления может остаться лишняя закрывающая.
    opens = value.count("{")
    closes = value.count("}")
    while closes > opens and value.endswith("}"):
        value = value[:-1].rstrip()
        closes -= 1
    # Снимаем внешние парные скобки у display-формул.
    while value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    return value.strip()


def _normalize(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
