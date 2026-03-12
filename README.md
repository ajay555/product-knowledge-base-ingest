# PDF Ingestion Pipeline

Parses PDFs in `customer_data/`, extracts text chunks + images, generates
vector embeddings, and loads everything into Neon (pgvector) + Vercel Blob.

## Quick start

```bash
cd ingestion/

# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit .env — fill in your OpenAI API key (other values are pre-filled)
#    OPENAI_API_KEY=sk-...

# 3. First run — apply schema and load all PDFs
python ingest.py --setup-schema --full-reload

# 4. Subsequent runs — only process new/changed PDFs
python ingest.py
```

## CLI flags

| Flag | Description |
|------|-------------|
| *(none)* | Incremental mode — skip PDFs whose SHA-256 hash hasn't changed |
| `--full-reload` | **Destructive.** Clears all DB rows + Vercel blobs, then re-ingests everything |
| `--setup-schema` | Applies `schema.sql` to the database (safe to run repeatedly — uses `IF NOT EXISTS`) |
| `--dry-run` | Parse PDFs and log what would happen; no DB writes, no blob uploads |
| `--source-dir PATH` | Override `PDF_SOURCE_DIR` from `.env` |
| `--verbose` | Enable DEBUG-level logging |

## Database schema

Three tables in Neon PostgreSQL:

| Table | Purpose |
|-------|---------|
| `documents` | One row per PDF — filename, page count, SHA-256 hash, processed timestamp |
| `text_chunks` | Chunked page text with 1536-dim vector embedding + page / section metadata |
| `page_images` | Extracted images with Vercel Blob URL + bounding box + caption text |

**Querying at runtime (Q&A frontend):**

```sql
-- 1. Find top-k most relevant chunks
SELECT chunk_id, doc_filename, page_num, text, section_heading
FROM   text_chunks
ORDER  BY embedding <=> %s::vector   -- cosine distance to query embedding
LIMIT  5;

-- 2. Pull images from the same pages
SELECT blob_url, caption_text, width, height
FROM   page_images
WHERE  doc_id = %s AND page_num = ANY(%s)
ORDER  BY page_num, image_index;
```

## File structure

```
ingestion/
├── .env               ← credentials + config (DO NOT commit to git)
├── ingest.py          ← main orchestrator / CLI entry point
├── pdf_parser.py      ← PyMuPDF text chunking + image extraction
├── blob_store.py      ← Vercel Blob upload / delete helpers
├── embeddings.py      ← batched OpenAI embedding generation
├── schema.sql         ← CREATE TABLE / INDEX statements
└── requirements.txt
```

## Notes on image extraction

- Images smaller than `MIN_IMAGE_WIDTH × MIN_IMAGE_HEIGHT` (default 80×80 px) are skipped — this filters out bullets, dividers, and decorative elements.
- Every image is converted to PNG before upload for consistency.
- The `caption_text` field contains the nearest text block within 60 pts of the image bounding box — useful for displaying image labels in the UI.
- `bbox` (JSON) records the image position on the page; can be used to render a page-level heatmap or highlight in a PDF viewer.

## Incremental re-ingestion

When you add a new PDF or replace an existing one, just run `python ingest.py`.
The script computes a SHA-256 hash of each file and skips anything already loaded
with a matching hash. A changed file is fully removed (DB rows + blobs) then
re-ingested from scratch.
