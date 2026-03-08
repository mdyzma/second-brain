"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from second_brain.config.settings import AppSettings, DatabaseSettings, EmbeddingSettings
from second_brain.ltm.embedder import Embedder


@pytest.fixture
def test_db_settings():
    """Database settings pointing to a test database."""
    return DatabaseSettings(
        host="localhost",
        port=5432,
        name="secondbraindb_test",
        user="midy",
        password="testpassword",
    )


@pytest.fixture
def test_settings(test_db_settings):
    """Full app settings for testing."""
    return AppSettings(
        anthropic_api_key="test-key",
        db=test_db_settings,
        embedding=EmbeddingSettings(model="all-MiniLM-L6-v2", dimensions=384),
    )


@pytest.fixture
def mock_embedder():
    """Embedder that returns deterministic vectors without loading the model."""
    embedder = MagicMock(spec=Embedder)
    embedder.dimensions = 384
    embedder.embed.return_value = [0.1] * 384
    embedder.embed_batch.return_value = [[0.1] * 384]
    return embedder
