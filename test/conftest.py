"""Shared pytest fixtures for tajweed embedder tests."""

import pytest
from tajweed_embeddings.embedder.tajweed_embedder import TajweedEmbedder


@pytest.fixture
def emb():
    """Return a fresh TajweedEmbedder instance."""
    return TajweedEmbedder()
