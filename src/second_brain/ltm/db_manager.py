"""Database operations for Long-Term Memory (agent_memories table)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from second_brain.config.settings import DatabaseSettings


@dataclass
class MemoryRecord:
    """A single memory row from the agent_memories table."""

    id: uuid.UUID
    content: str
    category: str | None = None
    agent_name: str | None = None
    source_file: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    similarity: float | None = None  # Populated by search results only


class DatabaseManager:
    """Manages connections and CRUD for the agent_memories table."""

    TABLE = "agent_memories"

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self._settings = settings or DatabaseSettings()
        self._conn: psycopg.Connection | None = None

    def connect(self) -> psycopg.Connection:
        """Open (or return existing) database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._settings.dsn)
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DatabaseManager:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def insert_memory(
        self,
        content: str,
        embedding: list[float],
        category: str = "note",
        agent_name: str = "System",
        source_file: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Insert a single memory record. Returns the new UUID."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.TABLE}
                    (agent_name, category, source_file, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING id
                """,
                (
                    agent_name,
                    category,
                    source_file,
                    content,
                    embedding,
                    psycopg.types.json.Jsonb(metadata or {}),
                ),
            )
            row = cur.fetchone()
            conn.commit()
            assert row is not None
            return row[0]

    def search_by_vector(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[MemoryRecord]:
        """Vector similarity search using cosine distance.

        Returns MemoryRecords sorted by similarity (highest first).
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content, category, agent_name, source_file,
                       metadata, created_at,
                       1 - (embedding <=> %s) AS similarity
                FROM {self.TABLE}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            sim = float(row[7])
            if sim >= similarity_threshold:
                results.append(
                    MemoryRecord(
                        id=row[0],
                        content=row[1],
                        category=row[2],
                        agent_name=row[3],
                        source_file=row[4],
                        metadata=row[5] or {},
                        created_at=row[6],
                        similarity=sim,
                    )
                )
        return results

    def count(self) -> int:
        """Return total number of memories."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
            row = cur.fetchone()
            assert row is not None
            return row[0]

    def delete_by_source(self, source_file: str) -> int:
        """Delete all memories from a given source file. Returns count deleted."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.TABLE} WHERE source_file = %s",
                (source_file,),
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted
