from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from app.meta import SocialMetaMiddleware, preview_meta, rewrite_meta


def test_middleware_rewrites_root_html():
    app = FastAPI()
    app.add_middleware(SocialMetaMiddleware)

    @app.get("/", response_class=HTMLResponse)
    def root():
        return "<html><head><meta property=\"og:image\" content=\"\" /></head></html>"

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "https://195.62.49.192.nip.io/preview.png" in response.text
    assert 'content=""' not in response.text


def test_rewrite_meta_removes_gradio_defaults():
    html = """<html><head>
    <meta property="og:title" content="Gradio" />
    <meta
        property="og:image"
        content="https://raw.githubusercontent.com/gradio-app/gradio/main/header.jpg"
    />
    <meta name="twitter:image" content="https://raw.githubusercontent.com/gradio-app/gradio/main/header.jpg" />
    </head><body></body></html>"""

    result = rewrite_meta(html, "https://example.test")

    assert "raw.githubusercontent.com" not in result
    assert 'content="Gradio"' not in result
    assert 'content="https://example.test/preview.png"' in result
    assert "diffeq-wiki-rag" in result


def test_rewrite_meta_drops_empty_images():
    html = '<html><head><meta property="og:image" content="" /></head></html>'

    result = rewrite_meta(html, "https://example.test")

    assert 'content=""' not in result
    assert "https://example.test/preview.png" in result


def test_rewrite_meta_without_head_returns_input():
    html = "<html><body>no head</body></html>"
    assert rewrite_meta(html, "https://example.test") == html


def test_preview_meta_escapes_html():
    result = preview_meta("https://example.test/")
    assert 'content="https://example.test/preview.png"' in result
    assert 'og:url" content="https://example.test/"' in result
