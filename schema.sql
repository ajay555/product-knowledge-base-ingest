-- ─────────────────────────────────────────────────────────────────────────────
--  PDF Q&A Knowledge Base Schema
--  Run once, or let ingest.py --setup-schema apply this automatically.
-- ─────────────────────────────────────────────────────────────────────────────

-- pgvector extension (must be enabled before creating vector columns)
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── documents ───────────────────────────────────────────────────────────────
-- One row per source PDF file.
CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT        NOT NULL UNIQUE,   -- original filename
    file_path       TEXT        NOT NULL,          -- path at ingestion time
    total_pages     INT         NOT NULL,
    file_hash       TEXT        NOT NULL,          -- SHA-256 of the file bytes
                                                   --   used for incremental detection
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB       NOT NULL DEFAULT '{}'
);

-- ─── text_chunks ─────────────────────────────────────────────────────────────
-- Chunked text extracted from each page, with a pgvector embedding.
CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_num        INT         NOT NULL,   -- 1-based
    chunk_index     INT         NOT NULL,   -- order within the page

    -- Raw extracted text for this chunk
    text            TEXT        NOT NULL,

    -- Vector embedding (OpenAI text-embedding-3-small = 1536 dims)
    embedding       VECTOR(1536),

    -- Structural context
    section_heading TEXT,                  -- nearest heading above this chunk (if detected)
    bbox            JSONB,                 -- {"x0":f, "y0":f, "x1":f, "y1":f} on the page

    -- Denormalised document info (avoids JOIN in hot query path)
    doc_filename    TEXT        NOT NULL,
    doc_total_pages INT         NOT NULL,

    -- Arbitrary extra metadata for future use
    metadata        JSONB       NOT NULL DEFAULT '{}',

    UNIQUE (doc_id, page_num, chunk_index)
);

-- ─── page_images ─────────────────────────────────────────────────────────────
-- Every image extracted from a PDF page.
CREATE TABLE IF NOT EXISTS page_images (
    image_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_num        INT         NOT NULL,   -- 1-based
    image_index     INT         NOT NULL,   -- order within the page

    -- Storage
    blob_url        TEXT        NOT NULL,   -- Vercel Blob public URL
    blob_pathname   TEXT        NOT NULL,   -- Vercel Blob pathname (for deletion)

    -- Image properties
    width           INT,
    height          INT,
    image_format    TEXT,                  -- PNG / JPEG / WEBP …

    -- Positional context on the page
    bbox            JSONB,                 -- {"x0":f, "y0":f, "x1":f, "y1":f}

    -- Text within ~60px of the image (caption / label detection)
    caption_text    TEXT,

    -- Denormalised for fast lookup without JOIN
    doc_filename    TEXT        NOT NULL,

    metadata        JSONB       NOT NULL DEFAULT '{}',

    UNIQUE (doc_id, page_num, image_index)
);

-- ─── Indexes ─────────────────────────────────────────────────────────────────

-- Fast lookup of all chunks / images belonging to a document
CREATE INDEX IF NOT EXISTS idx_chunks_doc
    ON text_chunks (doc_id);

-- Fast lookup of images on the same page as a matching chunk (used at query time)
CREATE INDEX IF NOT EXISTS idx_chunks_doc_page
    ON text_chunks (doc_id, page_num);

CREATE INDEX IF NOT EXISTS idx_images_doc_page
    ON page_images (doc_id, page_num);

-- ANN vector search using HNSW (cosine distance)
-- m=16, ef_construction=64 are safe defaults; tune upward for larger corpora
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON text_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
