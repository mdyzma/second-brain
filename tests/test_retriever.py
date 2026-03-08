"""Tests for the Retriever class."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from second_brain.config.settings import AppSettings
from second_brain.ltm.db_manager import DatabaseManager, MemoryRecord
from second_brain.ltm.retriever import Retriever


class TestRetriever:
    def test_search_combines_embed_and_db(self, mock_embedder):
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.search_by_vector.return_value = [
            MemoryRecord(
                id=uuid.uuid4(),
                content="Test memory content",
                category="note",
                agent_name="System",
                source_file="/test/file.md",
                created_at=datetime.now(),
                similarity=0.85,
            )
        ]

        settings = AppSettings(retrieval_top_k=3, retrieval_similarity_threshold=0.2)
        retriever = Retriever(db=mock_db, embedder=mock_embedder, settings=settings)
        result = retriever.search("test query")

        mock_embedder.embed.assert_called_once_with("test query")
        mock_db.search_by_vector.assert_called_once()
        assert len(result.memories) == 1
        assert "Test memory content" in result.context_text

    def test_search_empty_results(self, mock_embedder):
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.search_by_vector.return_value = []

        retriever = Retriever(db=mock_db, embedder=mock_embedder)
        result = retriever.search("no results query")

        assert result.memories == []
        assert result.context_text == ""

    def test_format_context_numbers_results(self):
        memories = [
            MemoryRecord(
                id=uuid.uuid4(),
                content="First memory",
                source_file="/path/a.md",
                similarity=0.9,
            ),
            MemoryRecord(
                id=uuid.uuid4(),
                content="Second memory",
                category="chat",
                similarity=0.7,
            ),
        ]
        text = Retriever._format_context(memories)
        assert "[1]" in text
        assert "[2]" in text
        assert "First memory" in text
        assert "Second memory" in text
        assert "0.90" in text
        assert "0.70" in text
