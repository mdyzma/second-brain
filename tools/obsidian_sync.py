import os
import time
import psycopg
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
VAULT_PATH = "/path/to/your/obsidian/vault"
DB_CONFIG = "host=192.168.88.231 dbname=secondbraindb user=midy password=your_password"
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class NoteHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".md"):
            self.process_note(event.src_path)

    def process_note(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking (by paragraph) to keep vectors precise
        chunks = [c for c in content.split("\n\n") if len(c) > 20]
        
        with psycopg.connect(DB_CONFIG) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # 1. Clear old version of this note
                cur.execute("DELETE FROM agent_memories WHERE source_file = %s", (file_path,))
                
                # 2. Insert new chunks
                for chunk in chunks:
                    embedding = MODEL.encode(chunk).tolist()
                    cur.execute("""
                        INSERT INTO agent_memories (agent_name, category, source_file, content, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, ("System", "note", file_path, chunk, embedding))
                conn.commit()
        print(f"🔄 Synced: {Path(file_path).name}")

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(NoteHandler(), VAULT_PATH, recursive=True)
    print(f"👀 Watching Obsidian vault at {VAULT_PATH}...")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
