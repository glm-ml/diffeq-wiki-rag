from types import SimpleNamespace

from app.ingest import ensure_index as module


class FakeClient:
    def __init__(self, count: int):
        self._count = count
        self.closed = False

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def count(self, collection_name):
        return SimpleNamespace(count=self._count)

    def close(self):
        self.closed = True


class FakeStore:
    def __init__(self):
        self.added = []

    def add_documents(self, documents, batch_size=64):
        self.added.append((documents, batch_size))


def test_ensure_index_skips_filled_collection(monkeypatch):
    client = FakeClient(count=132)
    monkeypatch.setattr(module, "open_client", lambda: client)
    monkeypatch.setattr(module, "ensure_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "load_corpus",
        lambda path: (_ for _ in ()).throw(AssertionError("не должен читать корпус")),
    )

    assert module.ensure_index() == 132
    assert client.closed is True


def test_ensure_index_fills_empty_collection(monkeypatch):
    client = FakeClient(count=0)
    store = FakeStore()
    documents = [object(), object()]
    monkeypatch.setattr(module, "open_client", lambda: client)
    monkeypatch.setattr(module, "ensure_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_corpus", lambda path: documents)
    monkeypatch.setattr(module, "build_vectorstore", lambda client: store)

    assert module.ensure_index() == 2
    assert store.added == [(documents, 64)]
    assert client.closed is True
