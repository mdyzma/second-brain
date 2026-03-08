"""Tests for configuration loading."""

from second_brain.config.settings import AppSettings, DatabaseSettings, EmbeddingSettings


class TestDatabaseSettings:
    def test_dsn_format(self):
        s = DatabaseSettings(host="h", port=5432, name="db", user="u", password="p")
        assert s.dsn == "host=h port=5432 dbname=db user=u password=p"

    def test_url_format(self):
        s = DatabaseSettings(host="h", port=5432, name="db", user="u", password="p")
        assert s.url == "postgresql://u:p@h:5432/db"

    def test_default_host(self):
        s = DatabaseSettings(password="p")
        assert s.host == "192.168.88.231"

    def test_default_port(self):
        s = DatabaseSettings(password="p")
        assert s.port == 5432


class TestEmbeddingSettings:
    def test_defaults(self):
        s = EmbeddingSettings()
        assert s.model == "all-MiniLM-L6-v2"
        assert s.dimensions == 384


class TestAppSettings:
    def test_defaults(self):
        s = AppSettings()
        assert s.retrieval_top_k == 5
        assert s.retrieval_similarity_threshold == 0.3

    def test_nested_settings(self):
        s = AppSettings()
        assert s.embedding.dimensions == 384
        assert s.llm.provider == "anthropic"
