from langchain_core.documents import Document

from app.agent import tools as agent_tools


class FakePipeline:
    def retrieve(self, question):
        return [
            Document(
                page_content="Условие Липшица",
                metadata={
                    "article": "Lipschitz continuity",
                    "section": "Definitions",
                    "source": "https://en.wikipedia.org/wiki/Lipschitz_continuity#Definitions",
                },
            ),
            Document(
                page_content="Тот же источник",
                metadata={
                    "article": "Lipschitz continuity",
                    "section": "Definitions",
                    "source": "https://en.wikipedia.org/wiki/Lipschitz_continuity#Definitions",
                },
            ),
        ]

    def run(self, question):
        raise AssertionError("search_wikipedia не должен вызывать генерацию ответа")


def test_search_wikipedia_returns_passages_not_answer(monkeypatch):
    agent_tools.set_pipeline_getter(lambda: FakePipeline())
    output = agent_tools.search_wikipedia.invoke({"query": "Липшиц"})

    assert "[1] Lipschitz continuity" in output
    assert "Условие Липшица" in output
    assert "URL:" in output
    assert "Sources:" in output
    assert output.count("Sources:") == 1
    sources_block = output.rsplit("Sources:", 1)[1]
    assert sources_block.count(
        "https://en.wikipedia.org/wiki/Lipschitz_continuity#Definitions"
    ) == 1


def test_run_python_executes_code():
    assert agent_tools.run_python.invoke({"code": "print(2 + 2)"}) == "4"


def test_run_python_handles_cyrillic_output():
    output = agent_tools.run_python.invoke({"code": 'print("корни: 1 и 2")'})
    assert output == "корни: 1 и 2"


def test_run_python_returns_error():
    output = agent_tools.run_python.invoke({"code": "1 / 0"})
    assert "Ошибка" in output


def test_web_search_can_be_disabled(monkeypatch):
    from app.settings import get_settings

    monkeypatch.setattr(get_settings(), "enable_web_search", False)
    assert "отключён" in agent_tools.web_search.invoke({"query": "что нового"})
