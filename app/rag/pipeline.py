"""RAG-конвейер: поиск чанков и генерация ответа со ссылками."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.llm import get_llm
from app.rag.prompt import HUMAN_PROMPT, SYSTEM_PROMPT
from app.rag.store import build_vectorstore
from app.settings import get_settings


@dataclass
class RagResult:
    answer: str
    documents: list[Document]


def format_context(documents: list[Document]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = metadata.get("article", "Wikipedia")
        section = metadata.get("section", "")
        subsection = metadata.get("subsection", "")
        heading = " / ".join(part for part in (section, subsection) if part)
        blocks.append(
            f"[{index}] {title} — {heading}\n"
            f"URL: {metadata.get('source', '')}\n"
            f"{document.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


class RagPipeline:
    def __init__(self, retriever: Runnable, chain: Runnable):
        self.retriever = retriever
        self.chain = chain

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def answer(self, question: str, documents: list[Document]) -> str:
        return self.chain.invoke(
            {"context": format_context(documents), "question": question}
        )

    def stream_answer(self, question: str, documents: list[Document]):
        yield from self.chain.stream(
            {"context": format_context(documents), "question": question}
        )

    def run(self, question: str) -> RagResult:
        documents = self.retrieve(question)
        return RagResult(answer=self.answer(question, documents), documents=documents)


def build_rag_pipeline(
    retriever: Runnable | None = None,
    llm: Runnable | None = None,
) -> RagPipeline:
    settings = get_settings()
    if retriever is None:
        store = build_vectorstore()
        retriever = store.as_retriever(search_kwargs={"k": settings.top_k})
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    chain = prompt | (llm or get_llm()) | StrOutputParser()
    return RagPipeline(retriever=retriever, chain=chain)
