"""Application configuration using Pydantic BaseSettings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "192.168.88.231"
    port: int = 5432
    name: str = "secondbraindb"
    user: str = "midy"
    password: str = ""

    @property
    def dsn(self) -> str:
        """Connection string in psycopg key=value format."""
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.name} user={self.user} password={self.password}"
        )

    @property
    def url(self) -> str:
        """Standard PostgreSQL URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    vault_path: Path = Path.home() / "Documents" / "Obsidian"
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.3

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def get_settings() -> AppSettings:
    """Factory function for settings. Enables test overrides."""
    return AppSettings()
