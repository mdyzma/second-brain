"""Tests for the Embedder class."""

import pytest

from second_brain.ltm.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance (model loads once for all tests in module)."""
    return Embedder()


class TestEmbedder:
    def test_embed_returns_correct_dimensions(self, embedder):
        result = embedder.embed("hello world")
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_embed_batch_returns_list_of_vectors(self, embedder):
        results = embedder.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 384
        assert len(results[1]) == 384

    def test_lazy_loading(self):
        e = Embedder()
        assert e._model is None
        _ = e.embed("trigger load")
        assert e._model is not None

    def test_dimensions_property(self):
        e = Embedder()
        assert e.dimensions == 384
