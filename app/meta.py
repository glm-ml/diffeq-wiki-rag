"""Open Graph и Twitter meta-теги для превью в мессенджерах."""

from __future__ import annotations

import html as html_module
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.settings import get_settings

TITLE = "diffeq-wiki-rag — агентный RAG по дифференциальным уравнениям"
DESCRIPTION = (
    "Ответы по статьям Википедии со ссылками, формулами "
    "и трейсом инструментов агента."
)

META_TAG_RE = re.compile(
    r'<meta\s+(?:property|name)="(?:og|twitter):[^"]*"[^>]*?>',
    re.IGNORECASE | re.DOTALL,
)


def preview_meta(base_url: str) -> str:
    base = base_url.rstrip("/")
    image = f"{base}/preview.png"
    title = html_module.escape(TITLE)
    description = html_module.escape(DESCRIPTION)
    return "\n".join(
        [
            '<meta property="og:type" content="website" />',
            f'<meta property="og:url" content="{base}/" />',
            f'<meta property="og:title" content="{title}" />',
            f'<meta property="og:description" content="{description}" />',
            f'<meta property="og:image" content="{image}" />',
            '<meta property="og:image:width" content="1200" />',
            '<meta property="og:image:height" content="630" />',
            '<meta name="twitter:card" content="summary_large_image" />',
            f'<meta name="twitter:title" content="{title}" />',
            f'<meta name="twitter:description" content="{description}" />',
            f'<meta name="twitter:image" content="{image}" />',
        ]
    )


def rewrite_meta(html: str, base_url: str) -> str:
    """Убрать стандартные теги Gradio и вставить свои."""
    if "</head>" not in html:
        return html
    cleaned = META_TAG_RE.sub("", html)
    return cleaned.replace("</head>", preview_meta(base_url) + "\n</head>", 1)


class SocialMetaMiddleware(BaseHTTPMiddleware):
    """Подменяет meta-теги только в HTML-ответе главной страницы."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.method != "GET" or request.url.path not in ("/", "/index.html"):
            return response

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return response
        if response.headers.get("content-encoding", "identity") != "identity":
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        text = rewrite_meta(text, get_settings().public_base_url)
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        return Response(
            content=text,
            status_code=response.status_code,
            headers=headers,
            media_type="text/html",
        )
