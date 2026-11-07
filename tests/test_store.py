from qdrant_client import QdrantClient

from app.rag.store import ensure_collection
from app.settings import get_settings


def test_ensure_collection_creates_collection():
    client = QdrantClient(location=":memory:")
    ensure_collection(client)

    assert client.collection_exists(get_settings().collection_name)
    info = client.get_collection(get_settings().collection_name)
    assert info.config.params.vectors.size == get_settings().embedding_dim


def test_ensure_collection_recreates_when_requested():
    from qdrant_client.models import PointStruct

    client = QdrantClient(location=":memory:")
    ensure_collection(client)
    client.upsert(
        collection_name=get_settings().collection_name,
        points=[
            PointStruct(
                id=1,
                vector=[0.0] * get_settings().embedding_dim,
                payload={},
            )
        ],
    )

    ensure_collection(client, recreate=True)

    assert client.count(get_settings().collection_name).count == 0
