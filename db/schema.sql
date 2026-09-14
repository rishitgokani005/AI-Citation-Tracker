-- Schema for AI Citation Tracker SQLite Database

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL,
    brand_name TEXT NOT NULL,
    position INTEGER NOT NULL,
    sentiment TEXT NOT NULL,
    FOREIGN KEY(response_id) REFERENCES responses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL,
    source_url TEXT,
    source_domain TEXT NOT NULL,
    FOREIGN KEY(response_id) REFERENCES responses(id) ON DELETE CASCADE
);

-- Indexing for performance
CREATE INDEX IF NOT EXISTS idx_responses_prompt_platform ON responses(prompt_id, platform);
CREATE INDEX IF NOT EXISTS idx_mentions_response_id ON mentions(response_id);
CREATE INDEX IF NOT EXISTS idx_citations_response_id ON citations(response_id);
