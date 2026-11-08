from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_preview_image_is_png():
    response = client.get("/preview.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_chat_rejects_too_short_question():
    response = client.post("/chat", json={"question": "?"})
    assert response.status_code == 422


def test_chat_returns_answer_and_sources(monkeypatch):
    from app.rag.pipeline import RagResult
    from app.runtime import runtime
    from langchain_core.documents import Document

    class FakePipeline:
        def run(self, question):
            return RagResult(
                answer="Условие Липшица [1].",
                documents=[
                    Document(
                        page_content="|f(x)-f(y)| <= L|x-y|",
                        metadata={
                            "article": "Lipschitz continuity",
                            "section": "Definition",
                            "source": "https://en.wikipedia.org/wiki/Lipschitz_continuity#Definition",
                        },
                    )
                ],
            )

    monkeypatch.setattr(runtime, "pipeline", FakePipeline())
    response = client.post("/chat", json={"question": "Что такое условие Липшица?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Условие Липшица")
    assert payload["sources"][0]["title"] == "Lipschitz continuity"
    assert payload["sources"][0]["url"].endswith("#Definition")


def test_agent_returns_trace(monkeypatch):
    from app.runtime import runtime
    from langchain_core.messages import AIMessage, ToolMessage

    class FakeGraph:
        def invoke(self, state, config):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_wikipedia",
                                "args": {"query": "вронскиан"},
                                "id": "call-1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=(
                            "Вронскиан — определитель [1].\n\nSources:\n"
                            "- Wronskian | https://en.wikipedia.org/wiki/Wronskian"
                        ),
                        name="search_wikipedia",
                        tool_call_id="call-1",
                    ),
                    AIMessage(content="Вронскиан — определитель [1]."),
                ],
                "iterations": 2,
            }

    monkeypatch.setattr(runtime, "graph", FakeGraph())
    response = client.post("/agent", json={"question": "Что такое вронскиан?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["iterations"] == 2
    assert payload["thread_id"]
    assert payload["trace"][0]["tool"] == "search_wikipedia"
    assert payload["sources"][0]["url"] == "https://en.wikipedia.org/wiki/Wronskian"
