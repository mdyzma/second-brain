"""Obsidian vault file watcher that syncs .md files to LTM."""

from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from second_brain.config.settings import get_settings
from second_brain.ltm.db_manager import DatabaseManager
from second_brain.ltm.embedder import Embedder


class NoteHandler(FileSystemEventHandler):
    """Watches for .md file changes and syncs them to the database."""

    def __init__(self, db: DatabaseManager, embedder: Embedder) -> None:
        self._db = db
        self._embedder = embedder

    def on_modified(self, event) -> None:  # noqa: ANN001
        if event.src_path.endswith(".md"):
            self.process_note(event.src_path)

    def process_note(self, file_path: str) -> None:
        """Read, chunk, embed, and store a markdown file."""
        content = Path(file_path).read_text(encoding="utf-8")
        chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 20]

        if not chunks:
            return

        # Clear old version of this note
        self._db.delete_by_source(file_path)

        # Batch embed all chunks at once
        embeddings = self._embedder.embed_batch(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            self._db.insert_memory(
                content=chunk,
                embedding=embedding,
                category="note",
                agent_name="System",
                source_file=file_path,
            )
        print(f"Synced: {Path(file_path).name} ({len(chunks)} chunks)")


def run_sync() -> None:
    """Start the Obsidian vault file watcher."""
    settings = get_settings()
    db = DatabaseManager(settings.db)
    db.connect()
    embedder = Embedder(settings.embedding)

    handler = NoteHandler(db=db, embedder=embedder)
    observer = Observer()
    observer.schedule(handler, str(settings.vault_path), recursive=True)
    print(f"Watching Obsidian vault at {settings.vault_path}...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()
        db.close()


if __name__ == "__main__":
    run_sync()
