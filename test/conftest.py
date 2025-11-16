import pytest
from tajweed_embeddings.embedder.tajweed_embedder import TajweedEmbedder

@pytest.fixture
def emb():
    return TajweedEmbedder()