import pytest
from tajweed_embedder import TajweedEmbedder

@pytest.fixture
def emb():
    return TajweedEmbedder()