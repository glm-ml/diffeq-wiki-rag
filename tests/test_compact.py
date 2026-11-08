from langchain_core.documents import Document

from app.rag.compact import compact_documents, split_sentences


def fake_scorer(question: str, sentences: list[str]) -> list[float]:
    return [1.0 if "important" in sentence.lower() else 0.0 for sentence in sentences]


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("No punctuation") == ["No punctuation"]


def test_compact_keeps_best_sentence_and_metadata():
    documents = [
        Document(
            page_content="important fact one. filler sentence. important fact two.",
            metadata={"chunk_id": "a", "article": "A"},
        ),
        Document(
            page_content="filler only. important detail.",
            metadata={"chunk_id": "b", "article": "B"},
        ),
    ]

    compacted, stats = compact_documents(
        documents,
        "important question",
        fake_scorer,
        max_tokens=10,
    )

    assert len(compacted) == 2
    assert "important" in compacted[0].page_content
    assert "important" in compacted[1].page_content
    assert compacted[0].metadata["chunk_id"] == "a"
    assert compacted[1].metadata["chunk_id"] == "b"
    assert compacted[0].metadata["original_content"].startswith("important fact one")
    assert stats.before_tokens > stats.after_tokens
    assert stats.documents == 2


def test_compact_keeps_one_sentence_per_document_even_if_over_budget():
    documents = [
        Document(page_content="important but very long sentence " * 20, metadata={"chunk_id": "a"}),
        Document(page_content="filler.", metadata={"chunk_id": "b"}),
    ]

    compacted, stats = compact_documents(
        documents,
        "question",
        fake_scorer,
        max_tokens=1,
    )

    assert len(compacted) == 2
    assert all(document.page_content for document in compacted)
    assert stats.selected_sentences >= 2
