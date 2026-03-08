"""Embedding generation using sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from second_brain.config.settings import EmbeddingSettings


class Embedder:
    """Generates vector embeddings from text."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings or EmbeddingSettings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model to avoid startup penalty when not needed."""
        if self._model is None:
            self._model = SentenceTransformer(self._settings.model)
        return self._model

    @property
    def dimensions(self) -> int:
        return self._settings.dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return [e.tolist() for e in embeddings]
