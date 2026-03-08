"""High-level retrieval interface combining embedding + vector search."""

from __future__ import annotations

from dataclasses import dataclass

from second_brain.config.settings import AppSettings
from second_brain.ltm.db_manager import DatabaseManager, MemoryRecord
from second_brain.ltm.embedder import Embedder


@dataclass
class RetrievalResult:
    """A search result with formatted context string."""

    memories: list[MemoryRecord]
    context_text: str


class Retriever:
    """Orchestrates embed-then-search for the LTM."""

    def __init__(
        self,
        db: DatabaseManager,
        embedder: Embedder,
        settings: AppSettings | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._settings = settings or AppSettings()

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Embed the query, search LTM, return formatted results."""
        top_k = top_k or self._settings.retrieval_top_k
        query_vec = self._embedder.embed(query)
        memories = self._db.search_by_vector(
            query_embedding=query_vec,
            top_k=top_k,
            similarity_threshold=self._settings.retrieval_similarity_threshold,
        )
        context_text = self._format_context(memories)
        return RetrievalResult(memories=memories, context_text=context_text)

    @staticmethod
    def _format_context(memories: list[MemoryRecord]) -> str:
        """Format memories into a context block for the LLM prompt."""
        if not memories:
            return ""
        parts = []
        for i, mem in enumerate(memories, 1):
            source = mem.source_file or mem.category or "unknown"
            parts.append(
                f"[{i}] (source: {source}, similarity: {mem.similarity:.2f})\n{mem.content}"
            )
        return "\n\n---\n\n".join(parts)
