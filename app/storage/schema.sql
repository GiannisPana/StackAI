CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    filename       TEXT NOT NULL,
    sha256         TEXT NOT NULL UNIQUE,
    num_pages      INTEGER NOT NULL,
    num_chunks     INTEGER NOT NULL,
    ocr_pages      INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL CHECK(status IN ('processing','ready','failed')),
    is_deleted     INTEGER NOT NULL DEFAULT 0,
    deleted_at     TEXT,
    ingested_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status, is_deleted);

CREATE TABLE IF NOT EXISTS chunks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id         INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    page           INTEGER NOT NULL,
    bbox_x0        REAL,
    bbox_y0        REAL,
    bbox_x1        REAL,
    bbox_y1        REAL,
    text           TEXT NOT NULL,
    token_count    INTEGER NOT NULL,
    section_title  TEXT,
    embedding_row  INTEGER UNIQUE,
    source         TEXT NOT NULL CHECK(source IN ('pdf_text','ocr'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_row ON chunks(embedding_row);
