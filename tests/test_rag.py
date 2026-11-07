from langchain_core.documents import Document

from app.rag.pipeline import RagPipeline, format_context


def _documents():
    return [
        Document(
            page_content="|f(x)-f(y)| <= L|x-y|",
            metadata={
                "article": "Lipschitz continuity",
                "section": "Definition",
                "subsection": "Local Lipschitz",
                "source": "https://en.wikipedia.org/wiki/Lipschitz_continuity#Definition",
            },
        ),
        Document(
            page_content="W = det(...)",
            metadata={
                "article": "Wronskian",
                "section": "Definition",
                "subsection": "",
                "source": "https://en.wikipedia.org/wiki/Wronskian",
            },
        ),
    ]


def test_format_context_numbers_sources():
    context = format_context(_documents())
    assert context.startswith("[1] Lipschitz continuity — Definition / Local Lipschitz")
    assert "[2] Wronskian — Definition" in context
    assert "URL: https://en.wikipedia.org/wiki/Wronskian" in context


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, question):
        return self.documents


class FakeChain:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return "Ответ [1]."

    def stream(self, payload):
        self.calls.append(payload)
        yield "Ответ "
        yield "[1]."


def test_pipeline_run_retrieves_and_answers():
    chain = FakeChain()
    pipeline = RagPipeline(FakeRetriever(_documents()), chain)

    result = pipeline.run("Что такое условие Липшица?")

    assert result.answer == "Ответ [1]."
    assert len(result.documents) == 2
    assert "Lipschitz continuity" in chain.calls[0]["context"]
    assert chain.calls[0]["question"] == "Что такое условие Липшица?"


def test_pipeline_streams_answer():
    pipeline = RagPipeline(FakeRetriever(_documents()), FakeChain())
    chunks = list(pipeline.stream_answer("Вопрос", _documents()))
    assert "".join(chunks) == "Ответ [1]."
