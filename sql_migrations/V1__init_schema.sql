CREATE EXTENSION IF NOT EXISTS vector;

-- LTM: agent_memories table (existing on Proxmox, created here for Docker dev)
CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT,
    agent_name TEXT,
    source_file TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_memories_embedding
    ON agent_memories USING hnsw (embedding vector_cosine_ops);

-- STM: short_term_memory table (for Phase 2, created now to avoid future migration)
CREATE TABLE IF NOT EXISTS short_term_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'CONSOLIDATED')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stm_status ON short_term_memory (status);
CREATE INDEX IF NOT EXISTS idx_stm_created ON short_term_memory (created_at);
